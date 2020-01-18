import torch
from hyperspace import hyperdrive
import os
from rdkit_free_train import trainer, load_data_models, get_optimizer

config = {
    'i': '/gpfs/alpine/med106/proj-shared/aclyde/MolecularAttention/qm8/qm8.smi',
    'r': 42,
    'precomputed_values': "/gpfs/alpine/med106/proj-shared/aclyde/MolecularAttention/qm8/qm8_values.npy",
    'precomputed_images': "/gpfs/alpine/med106/proj-shared/aclyde/MolecularAttention/qm8/qm8_images.pkl",
    'cv' : 1,
    'resnet101' : '/gpfs/alpine/med106/proj-shared/aclyde/torch_cache/checkpoints/resnet101-5d3b4d8f.pth'
}


def train_qm8(ts):
    dropout_rate, batch_size, lr, use_cyclic, nheads, intermediate, linear_layers = ts
    device = torch.device("cuda")
    train_loader, test_loader, model = load_data_models(config['i'], config['r'], 8, batch_size, 'custom',
                                                        nheads=int(batch_size),
                                                        precompute_frame=config['precomputed_values'],
                                                        precomputed_images=config['precomputed_images'],
                                                        imputer_pickle=None, eval=False,
                                                        tasks=16, gpus=1, rotate=True,
                                                        dropout=dropout_rate, intermediate_rep=intermediate, cvs=config['cv'], linear_layers=linear_layers, model_checkpoint=config['resnet101'])
    model.to(device)
    optimizer = get_optimizer('adamw')(model.parameters(), lr=lr)
    model, history = trainer(model, optimizer, train_loader, test_loader, epochs=2, gpus=1, tasks=16, mae=True,
                             pb=False, cyclic=use_cyclic)

    return history.get_last_metric(train=False)[0]


if __name__ == '__main__':
    params = [(0.0, 0.5),  # dropout
              (32, 256),  # batch_size
              (1e-6, 1e-2),  # learning rate
              [True, False],  # use cyclic
              (0, 10),  # nheads
              (64, 1024),  # itnermedioate
              (1, 6)]  # linear layers
    try:
        os.makedirs('/gpfs/alpine/med106/proj-shared/aclyde/MolecularAttention/qm8/hyperopt/cv' + str(config['cv']) + '/results/')
    except:
        pass
    try:
        os.makedirs('/gpfs/alpine/med106/proj-shared/aclyde/MolecularAttention/qm8/hyperopt/' + str(config['cv']) + '/checkpoints/')
    except:
        pass
    hyperdrive(objective=train_qm8,
               hyperparameters=params,
               results_path='/gpfs/alpine/med106/proj-shared/aclyde/MolecularAttention/qm8/hyperopt/cv' + str(config['cv']) + '/results/',
               checkpoints_path='/gpfs/alpine/med106/proj-shared/aclyde/MolecularAttention/qm8/hyperopt/' + str(config['cv']) + '/checkpoints/',
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=0)
