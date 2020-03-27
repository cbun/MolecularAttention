import argparse
import pickle
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from features.datasets import ImageDataset, funcs, get_properety_function, ImageDatasetPreLoaded
from features.generateFeatures import MORDRED_SIZE
from metrics import trackers
from models import imagemodel
from sklearn.utils import class_weight
from scipy import stats
print(torch.cuda.current_device())
if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())

print(torch.cuda.is_available())
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_optimizer(c):
    if c == 'sgd':
        return torch.optim.SGD
    elif c == 'adam':
        return torch.optim.Adam
    elif c == 'adamw':
        return torch.optim.AdamW

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='smiles input file')
    parser.add_argument('--precomputed_values', type=str, required=False, default=None,
                        help='separated dock scores for trainings')
    parser.add_argument('--precomputed_images', type=str, required=False, default=None,
                        help='precomputed images for trainings')
    parser.add_argument('-w', type=int, default=8, help='number of workers for data loaders to use.')
    parser.add_argument('-b', type=int, default=64, help='batch size to use')
    parser.add_argument('-o', type=str, default='saved_models/model.pt', help='name of file to save model to')
    parser.add_argument('-r', type=int, default=32, help='random seed for splitting.')
    parser.add_argument('-pb', action='store_true')
    parser.add_argument('-g', type=int, default=1, help='use multiple GPUs')
    parser.add_argument('-t', type=int, default=1, help='number of tasks')
    parser.add_argument('--nheads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--metric_plot_prefix', default=None, type=str, help='prefix for graphs for performance')
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer to use',
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning to use')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs to use')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--classifacation', action='store_true')
    parser.add_argument('--ensemble_eval', action='store_true')
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--width', default=256, type=int, help='rep size')
    parser.add_argument('--depth', default=2, type=int, help='num linear layers')
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--forward', action='store_true')
    parser.add_argument('--freeze', action='store_true')

    args = parser.parse_args()
    if args.metric_plot_prefix is None:
        args.metric_plot_prefix = "".join(args.o.split(".")[:-1]) + "_"
    args.optimizer = get_optimizer(args.optimizer)
    print(args)
    return args

def run_eval(model, train_loader, ordinal=False, classifacation=False, enseml=True, tasks=1):
    with torch.no_grad():
        model.eval()
        if classifacation:
            tracker =  trackers.PytorchHistory(
                metric=metrics.roc_auc_score, metric_name='roc-auc')
        else:
            tracker =  trackers.PytorchHistory()

        train_loss = 0
        test_loss = 0
        train_iters = 0
        test_iters = 0
        preds = []
        values = []
        predss = []
        valuess = []
        model.eval()

        for i in range(25 if enseml else 1):
            for i, (drugfeats, value) in enumerate(train_loader):
                drugfeats, value = drugfeats.to(device), value.to(device)
                pred, attn = model(drugfeats)
                mse_loss = torch.nn.functional.l1_loss(pred, value).mean()
                test_loss += mse_loss.item()
                test_iters += 1
                tracker.track_metric(pred.detach().cpu().numpy(), value.detach().cpu().numpy())
                valuess.append(value.cpu().detach().numpy().flatten())
                predss.append(pred.detach().cpu().numpy().flatten())

            preds.append(np.concatenate(predss, axis=0))
            values.append(np.concatenate(valuess, axis=0))
        preds = np.stack(preds)
        values = np.stack(values)
        print(preds.shape, values.shape)
        preds = np.mean(preds, axis=0)
        values = np.mean(values, axis=0)

        if ordinal:
            preds, values = np.round(preds), np.round(values)
            incorrect = 0
            for i in range(preds.shape[0]):
                if values[i] != preds[i]:
                    print("incorrect at", i)
                    incorrect += 1
            print("total incorrect", incorrect, incorrect / preds.shape[0])

        tracker.log_loss(test_loss / test_iters, train=False)
        tracker.log_metric(internal=True, train=False)

        print("val", test_loss / test_iters, 'r2', tracker.get_last_metric(train=False))
        print("avg ensmelb r2, mae", metrics.r2_score(values, preds), metrics.mean_absolute_error(values, preds))
        print(preds)
        print(values)
        print("Pearson corr", stats.pearsonr(preds,values))
    return model, tracker


def trainer(model, optimizer, train_loader, test_loader, epochs=5, gpus=1, tasks=1, classifacation=False, mae=False,
            pb=True, out="model.pt", cyclic=False, verbose=True):
    device = next(model.parameters()).device

    if classifacation:
        tracker = trackers.ComplexPytorchHistory() if tasks > 1 else trackers.PytorchHistory(
            metric=metrics.roc_auc_score, metric_name='roc-auc')
    else:
        tracker = trackers.ComplexPytorchHistory() if tasks > 1 else trackers.PytorchHistory()

    earlystopping = EarlyStopping(patience=50, delta=1e-5)
    if cyclic:
        lr_red = CosineAnnealingWarmRestarts(optimizer, T_0=20)
    else:
        lr_red = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, cooldown=0, verbose=verbose,
                                   threshold=1e-4,
                                   min_lr=1e-8)
    print("start epoch loop")    
    for epochnum in range(epochs):
        train_loss = 0
        test_loss = 0
        train_iters = 0
        test_iters = 0
        model.train()
        if pb:
            gen = tqdm(enumerate(train_loader))
        else:
            gen = enumerate(train_loader)
        for i, (drugfeats, value) in gen:
            optimizer.zero_grad()
            drugfeats, value = drugfeats.to(device), value.to(device)
            pred, attn = model(drugfeats)
            if classifacation:
                # compute and pass class weighting 
                cw = class_weight.compute_class_weight('balanced', np.array([0,1]),value)
                cw = torch.FloatTensor(cw).cuda()
                mse_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, value, weight=cw).mean()
            elif mae:
                mse_loss = torch.nn.functional.l1_loss(pred, value).mean()
            else:
                mse_loss = torch.nn.functional.mse_loss(pred, value).mean()
            mse_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += mse_loss.item()
            train_iters += 1
            tracker.track_metric(pred=pred.detach().cpu().numpy(), value=value.detach().cpu().numpy())

        tracker.log_loss(train_loss / train_iters, train=True)
        tracker.log_metric(internal=True, train=True)

        model.eval()
        with torch.no_grad():
            for i, (drugfeats, value) in enumerate(test_loader):
                drugfeats, value = drugfeats.to(device), value.to(device)
                pred, attn = model(drugfeats)

                if classifacation:
                    mse_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, value).mean()
                elif mae:
                    mse_loss = torch.nn.functional.l1_loss(pred, value).mean()
                else:
                    mse_loss = torch.nn.functional.mse_loss(pred, value).mean()
                test_loss += mse_loss.item()
                test_iters += 1
                tracker.track_metric(pred.detach().cpu().numpy(), value.detach().cpu().numpy())
        tracker.log_loss(test_loss / train_iters, train=False)
        tracker.log_metric(internal=True, train=False)

        lr_red.step(test_loss / test_iters)
        earlystopping(test_loss / test_iters)
        if verbose:
            print("Epoch", epochnum, train_loss / train_iters, test_loss / test_iters, tracker.metric_name,
                  tracker.get_last_metric(train=True), tracker.get_last_metric(train=False))

        if out is not None:
            if gpus == 1:
                state = model.state_dict()
                heads = model.nheads
            else:
                state = model.module.state_dict()
                heads = model.module.nheads
            torch.save({'model_state': state,
                        'opt_state': optimizer.state_dict(),
                        'history': tracker,
                        'nheads': heads,
                        'ntasks': tasks}, out)
        if earlystopping.early_stop:
            break
    return model, tracker

def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(smiles)
    return smiles if mol is not None else None


def load_data_models(fname, random_seed, workers, batch_size, precomputed_values, precomputed_images,  pname='logp', nheads=1,
                      eval=False, tasks=1, gpus=1, rotate=False,
                     classifacation=False, ensembl=False, dropout=0, intermediate_rep=None, depth=2, pretrain=True, forward=False, freeze=False):
    df = pd.read_csv(fname, header=0)
    smiles=list(df.iloc[:,0].values)
    features = np.load(precomputed_values).astype(np.float32).reshape(-1,1)
    precomputed_images = np.load(precomputed_images)
    print(precomputed_images[0].shape)
	
    assert(features.shape[0] == len(smiles))
    assert(precomputed_images.shape[0] == len(smiles))

    
    features=np.absolute(features)
    train_idx, test_idx, train_smiles, test_smiles = train_test_split(list(range(len(smiles))), smiles,
                                                                      test_size=0.2, random_state=random_seed)

    if eval:
        print("Assume data is already scaled correctly")
        test_dataset = ImageDatasetPreLoaded(smiles, features,
                                         property_func=get_properety_function(pname),
                                         values=tasks, rot=0, images=precomputed_images)
        test_loader = DataLoader(test_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size,
                             shuffle=(not eval))
        train_loader=None
    else:
        train_idx, test_idx, train_smiles, test_smiles = train_test_split(list(range(len(smiles))), smiles,
                                                                      test_size=0.2, random_state=random_seed, shuffle=True)
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        
        train_features = features[train_idx]
        test_features = features[test_idx]
        train_dataset = ImageDatasetPreLoaded(train_smiles, train_features,
                                          property_func=get_properety_function(pname),
                                          values=tasks, rot=359, images=[precomputed_images[i] for i in train_idx])
        train_loader = DataLoader(train_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size,
                                  shuffle=True)
        test_dataset = ImageDatasetPreLoaded(test_smiles, test_features, property_func=get_properety_function(pname),
                                         values=tasks, rot=359, images=[precomputed_images[i] for i in test_idx])
        test_loader = DataLoader(test_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size,
                                  shuffle=True)
    
    
    if intermediate_rep is None:
        model = imagemodel.ImageModel(nheads=nheads, outs=tasks, classifacation=classifacation, dr=dropout, linear_layers=depth, pretrain=pretrain, forward=forward, freeze=freeze)
    else:
        model = imagemodel.ImageModel(nheads=nheads, outs=tasks, classifacation=classifacation, dr=dropout,
                                      intermediate_rep=intermediate_rep, linear_layers=depth, pretrain=pretrain, forward=forward, freeze=freeze)

    if gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

        
    return train_loader, test_loader, model


if __name__ == '__main__':
    args = get_args()

    np.random.seed(args.r)
    torch.manual_seed(args.r)
    print("Model name", args.metric_plot_prefix)
    train_loader, test_loader, model = load_data_models(args.i, args.r, args.w, args.b, args.precomputed_values,args.precomputed_images,
                                                        pname='logp', nheads=args.nheads,
                                                        eval=args.eval,
                                                        tasks=args.t, gpus=args.g, rotate=args.rotate,
                                                        classifacation=args.classifacation, ensembl=args.ensemble_eval,
                                                        dropout=args.dropout_rate, intermediate_rep=args.width, depth=args.depth, pretrain=(not args.no_pretrain), forward = args.forward, freeze=args.freeze)
    print("Done.")

    print("Starting trainer.")
    if args.eval:
        model.load_state_dict(torch.load(args.o)['model_state'])
        model.to(device)
        run_eval(model, test_loader, ordinal=False, enseml=args.ensemble_eval)
        exit()
    model.to(device)
    if args.freeze:
        optimizer = args.optimizer(model.prop_model.parameters(), lr=args.lr)
    else:     
        optimizer = args.optimizer(model.parameters(), lr=args.lr)

    print("Number of parameters:",
          sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    model, history = trainer(model, optimizer, train_loader, test_loader, out=args.o, epochs=args.epochs, pb=args.pb,
                             gpus=args.g, classifacation=args.classifacation, tasks=args.t, mae=args.mae)
    history.plot_loss(save_file=args.metric_plot_prefix + "loss.png", title="Loss")
    history.plot_metric(save_file=args.metric_plot_prefix + "metric.png", title=history.metric_name)
    print("Finished training, now")
