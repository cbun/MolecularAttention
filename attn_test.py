import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

from train import load_data_models

if torch.cuda.is_available():
    import torch.backends.cudnn

    torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_attn_pred(drugfeats, value):
    drugfeats, value = drugfeats.to(device), value.to(device)
    model.return_attns = True
    pred, attn = model(drugfeats.unsqueeze(0))
    attn = attn.squeeze(0).detach()
    attn = torch.sum(attn, dim=0, keepdim=True)
    attn = attn.repeat([3, 1, 1]).unsqueeze(0)
    attn = torch.nn.functional.interpolate(attn, size=(128, 128), mode='bicubic')
    # drug_image = torch.cat([drugfeats.unsqueeze(0), 1.0 - attn[:, 1, :, :].unsqueeze(1)], dim=1)
    return pred, attn, drugfeats


if __name__ == '__main__':
    cmap = pl.cm.binary
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    _, dset, model = load_data_models("data/debugset.smi", 32, 1, 1, return_datasets=True)

    model = torch.load('saved_models/model.pt', map_location='cpu')['inference_model']
    model.eval()

    pred, attn, image = get_attn_pred(*dset[3])
    print(pred.shape, attn.shape, image.shape)

    plt.imshow(np.transpose(image.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    plt.contourf(list(range(128)), list(range(128)), 1.0 - attn.squeeze(0)[0], cmap=my_cmap, levels=5)
    plt.colorbar()
    # plt.imshow(np.transpose(image.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    # plt.imshow(np.transpose(attn.squeeze(0).numpy(), (1, 2, 0)), interpolation='nearest')
    plt.title("Predicition value " + str(pred.item()))
    plt.show()
