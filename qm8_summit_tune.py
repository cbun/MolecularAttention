import torch
from hyperspace import hyperdrive
from ray.tune import track

from rdkit_free_train import trainer, load_data_models, get_optimizer

config = {
    'i': 'i',
    'r': '32',
    'precomputed_values': "test",
    'precomputed_images': 'test'
}


def train_qm8(config):
    dropout_rate, batch_size, lr, use_cyclic, nheads, intermediate, linear_layers = config
    device = torch.device("cuda")
    args = config['args']
    train_loader, test_loader, model = load_data_models(config['i'], config['r'], 8, batch_size, 'custom',
                                                        nheads=batch_size,
                                                        precompute_frame=config['precomputed_values'],
                                                        imputer_pickle=None, eval=False,
                                                        tasks=16, gpus=1, rotate=True,
                                                        dropout=dropout_rate, intermediate_rep=intermediate)
    model.to(device)
    optimizer = get_optimizer('adamw')(model.parameters(), lr=lr)
    model, history = trainer(model, optimizer, train_loader, test_loader, epochs=200, gpus=1, tasks=16, mae=True,
                             pb=False, cyclic=use_cyclic)

    return track.log(mae=history.get_last_metric(train=False)[0])


if __name__ == '__main__':
    params = [(0.0, 0.5),  # dropout
              (32, 256),  # batch_size
              (1e-6, 1e-2),  # learning rate
              [True, False],  # use cyclic
              (0, 10),  # nheads
              (64, 1024),  # itnermedioate
              (1, 6)]  # linear layers

    hyperdrive(objective=train_qm8,
               hyperparameters=params,
               results_path='/path/to/save/results',
               checkpoints_path='/path/to/save/checkpoints',
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=0)
