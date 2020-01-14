import argparse
import multiprocessing

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from features.generateFeatures import MORDRED_SIZE
from features.datasets import ImageDataset, funcs, get_properety_function, ImageDatasetPreLoaded
from metrics import trackers
from models import imagemodel

if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_optimizer(c):
    if c == 'sgd':
        return torch.optim.SGD
    elif c == 'adam':
        return torch.optim.Adam
    elif c == 'adamw':
        return torch.optim.AdamW


def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return smiles if mol is not None else None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='smiles input file')
    parser.add_argument('--precomputed_values', type=str, required=False, default=None, help='precomputed decs for trainings')
    parser.add_argument('--imputer_pickle', type=str, required=False, default=None, help='imputer and scaler for transforming data')

    parser.add_argument('-p', choices=list(funcs.keys()), help='select property for model')
    parser.add_argument('-w', type=int, default=8, help='number of workers for data loaders to use.')
    parser.add_argument('-b', type=int, default=64, help='batch size to use')
    parser.add_argument('-o', type=str, default='saved_models/model.pt', help='name of file to save model to')
    parser.add_argument('-r', type=int, default=32, help='random seed for splitting.')
    parser.add_argument('-pb', action='store_true')
    parser.add_argument('--nheads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--metric_plot_prefix', default=None, type=str, help='prefix for graphs for performance')
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer to use',
                        choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', default=1e-4, type=float, help='learning to use')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs to use')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')

    args = parser.parse_args()
    if args.metric_plot_prefix is None:
        args.metric_plot_prefix = "".join(args.o.split(".")[:-1]) + "_"
    args.optimizer = get_optimizer(args.optimizer)
    return args


def trainer(model, optimizer, train_loader, test_loader, epochs=5):
    tracker = trackers.PytorchHistory()
    lr_red = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, cooldown=0, verbose=True, threshold=1e-4,
                               min_lr=1e-8)

    for epochnum in range(epochs):
        train_loss = 0
        test_loss = 0
        train_iters = 0
        test_iters = 0
        model.train()
        if args.pb:
            gen = tqdm(enumerate(train_loader))
        else:
            gen = enumerate(train_loader)
        for i, (drugfeats, value) in gen:
            optimizer.zero_grad()
            drugfeats, value = drugfeats.to(device), value.to(device)
            pred, attn = model(drugfeats)

            mse_loss = torch.nn.functional.mse_loss(pred, value).mean()
            mse_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += mse_loss.item()
            train_iters += 1
            tracker.track_metric(pred.detach().cpu().numpy(), value.detach().cpu().numpy())

        tracker.log_loss(train_loss / train_iters, train=True)
        tracker.log_metric(internal=True, train=True)

        model.eval()
        with torch.no_grad():
            for i, (drugfeats, value) in enumerate(test_loader):
                drugfeats, value = drugfeats.to(device), value.to(device)
                pred, attn = model(drugfeats)

                mse_loss = torch.nn.functional.mse_loss(pred, value).mean()
                test_loss += mse_loss.item()
                test_iters += 1
                tracker.track_metric(pred.detach().cpu().numpy(), value.detach().cpu().numpy())
        tracker.log_loss(train_loss / train_iters, train=False)
        tracker.log_metric(internal=True, train=False)

        lr_red.step(test_loss / test_iters)
        print("Epoch", epochnum, train_loss / train_iters, test_loss / test_iters, 'r2',
              tracker.get_last_metric(train=True), tracker.get_last_metric(train=False))

        torch.save({'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'history': tracker,
                    'nheads' : model.nheads}, args.o)
    return model, tracker


def load_data_models(fname, random_seed, workers, batch_size, pname='logp', return_datasets=False, nheads=1, precompute_frame=None, imputer_pickle=None):
    df = pd.read_csv(fname, header=None)
    smiles = []
    with multiprocessing.Pool() as p:
        gg = filter(lambda x: x is not None, p.imap(validate_smiles, list(df.iloc[:, 0])))
        for g in tqdm(gg, desc='validate smiles'):
            smiles.append(g)
    del df

    if precompute_frame is not None and imputer_pickle is not None:
        features = np.load(precompute_frame).astype(np.float32)
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        assert(features.shape[0] == len(smiles))
        assert(pname == 'all')
        train_idx, test_idx, train_smiles, test_smiles = train_test_split(list(range(len(smiles))), smiles, test_size=0.2, random_state=random_seed)
        train_features = features[train_idx]
        test_features = features[test_idx]

        train_dataset = ImageDatasetPreLoaded(train_smiles, train_features, imputer_pickle, property_func=get_properety_function(pname),
                                     values=MORDRED_SIZE if pname == 'all' else 1)
        train_loader = DataLoader(train_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size)

        test_dataset = ImageDatasetPreLoaded(test_smiles, test_features, imputer_pickle, property_func=get_properety_function(pname),
                                    values=MORDRED_SIZE if pname == 'all' else 1)
        test_loader = DataLoader(test_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size)

        model = imagemodel.ImageModel(nheads=nheads, outs=MORDRED_SIZE if pname == 'all' else 1)

    else:
        train_idx, test_idx = train_test_split(smiles, test_size=0.2, random_state=random_seed)

        train_dataset = ImageDataset(train_idx, property_func=get_properety_function(pname),
                                     values=MORDRED_SIZE if pname == 'all' else 1)
        train_loader = DataLoader(train_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size)

        test_dataset = ImageDataset(test_idx, property_func=get_properety_function(pname),
                                    values=MORDRED_SIZE if pname == 'all' else 1)
        test_loader = DataLoader(test_dataset, num_workers=workers, pin_memory=True, batch_size=batch_size)

        model = imagemodel.ImageModel(nheads=nheads, outs=MORDRED_SIZE if pname == 'all' else 1)

    if return_datasets:
        return train_dataset, test_dataset, model
    else:
        return train_loader, test_loader, model


if __name__ == '__main__':
    args = get_args()

    np.random.seed(args.r)
    torch.manual_seed(args.r)

    train_loader, test_loader, model = load_data_models(args.i, args.r, args.w, args.b, args.p, nheads=args.nheads, precompute_frame=args.precomputed_values, imputer_pickle=args.imputer_pickle)
    print("Done.")

    print("Starting trainer.")

    model.to(device)
    optimizer = args.optimizer(model.parameters(), lr=args.lr)

    print("Number of parameters:",
          sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    model, history = trainer(model, optimizer, train_loader, test_loader, epochs=args.epochs)
    history.plot_loss(save_file=args.metric_plot_prefix + "loss.png", title=args.mode + " Loss")
    history.plot_metric(save_file=args.metric_plot_prefix + "r2.png", title=args.mode + " " + history.metric_name)
    print("Finished training, now")
    #
    # print("Running evaluation for surface plots...")
    # res = produce_preds_timing(model, test_loader, cells[test_idx], drugs[test_idx], args.mode)
    # rds_model = rds.RegressionDetectionSurface(percent_min=-3)
    #
    # rds_model.compute(res[1], res[0], stratify=res[2], samples=30)
    # rds_model.plot(save_file=args.metric_plot_prefix + "rds_on_cell.png",
    #                title='Regression Enrichment Surface (Avg over Unique Cells)')
    # rds_model.compute(res[1], res[0], stratify=res[3], samples=30)
    # rds_model.plot(save_file=args.metric_plot_prefix + "rds_on_drug.png",
    #                title='Regression Enrichment Surface (Avg over Unique Drugs)')
    # print("Output all plots with prefix", args.metric_plot_prefix)
    #
    # print("r2", metrics.r2_score(res[1], res[0]))
    # print("mse", metrics.mean_squared_error(res[1], res[0]))
    #
    # print("AUC with cutoff", metrics.roc_auc_score((res[1] <= 0.5).astype(np.int32) , (res[0] <= 0.5).astype(np.int32) ))
    # print("Acc with cutoff", metrics.accuracy_score((res[1] <= 0.5).astype(np.int32) , (res[0] <= 0.5).astype(np.int32) ))
    # print("BalAcc with cutoff", metrics.balanced_accuracy_score((res[1] <= 0.5).astype(np.int32) , (res[0] <= 0.5).astype(np.int32) ))
    #
    # np.save(args.metric_plot_prefix + "preds.npy", res)
