import ray
from ray import tune
import argparse
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler
import numpy as np
import torch
from train import trainer, load_data_models, get_optimizer
from features.generateFeatures import MORDRED_SIZE
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='smiles input file')
    parser.add_argument('--precomputed_values', type=str, required=False, default=None,
                        help='precomputed decs for trainings')
    parser.add_argument('--imputer_pickle', type=str, required=False, default=None,
                        help='imputer and scaler for transforming data')
    parser.add_argument('-o', type=str, default='saved_models/model.pt', help='name of file to save model to')
    parser.add_argument('-r', type=int, default=32, help='random seed for splitting.')
    parser.add_argument('-g', type=int, default=1, help='use multiple GPUs')
    parser.add_argument('--metric_plot_prefix', default=None, type=str, help='prefix for graphs for performance')

    args = parser.parse_args()
    if args.metric_plot_prefix is None:
        args.metric_plot_prefix = "".join(args.o.split(".")[:-1]) + "_"
    args.t = 16
    print(args)
    return args

def train_qm8(config):
    device = torch.device("cuda")
    args = config['args']
    train_loader, test_loader, model = load_data_models(args.i, args.r, 8, config['batch_size'], 'custom', nheads=config['nheads'],
                                                        precompute_frame=args.precomputed_values,
                                                        imputer_pickle=args.imputer_pickle, eval=False,
                                                        tasks=args.t, gpus=1, rotate=True,
                                                        dropout=config['dropout_rate'])
    model.to(device)
    optimizer = get_optimizer(config['optimizer'])(model.parameters(), lr=config['optimizer'])
    model, history = trainer(model, optimizer, train_loader, test_loader, epochs=config['epochs'], gpus=1, tasks=args.t, mae=True)

    track.log(mae=history.get_last_metric(train=False)[0])

if __name__ == "__main__":

    args = get_args()

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")
    analysis = tune.run(
        train_qm8,
        name="qm8_exp",
        scheduler=sched,
        stop={
            "training_iteration":  100
        },
        resources_per_trial={
            "cpu": 8,
            "gpu": int(args.g)
        },
        num_samples=50,
        config={
            'args' : args,
            "optimizer" : tune.grid_search(['adam', 'adamw']),
            "dropout_rate" : tune.uniform(0, 0.3),
            "batch_size" : tune.sample_from(lambda spec: np.random.randint(32, 256)),
            "nheads" : tune.sample_from(lambda spec: 2 ** np.random.randint(2, 9)),
            "intermediate_rep" : tune.sample_from(lambda spec: np.random.randint(128, 512),
            "epochs" : 100,
            "lr": tune.sample_from(lambda spec: 10 * np.random.randint(-7, -2)),
            "use_gpu": int(args.g)
        })

    print("Best config is:", analysis.get_best_config(metric="mae"))