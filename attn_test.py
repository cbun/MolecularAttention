import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from models.imagemodel import ImageModel
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

    _, dset, model = load_data_models("data/debugset.smi", 32, 1, 1, 'hacceptor', return_datasets=True)

    model = ImageModel()
    model.load_state_dict(torch.load('saved_models/hacceptor_model.pt', map_location='cpu')['model_state'])
    model.eval()

    idx = 645
    imout, act = dset[idx]
    pred, attn, image = get_attn_pred(*dset[idx])
    print(pred.shape, attn.shape, image.shape)

    attn = attn.squeeze(0).numpy()
    atn_max = np.max(attn)
    atn_min = np.min(attn)
    attn = (attn - atn_min) / (atn_max - atn_min)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(np.transpose(imout.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    axs[1].imshow(np.transpose(image.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    # plt.contourf(list(range(128)), list(range(128)), 1.0 - attn, cmap=my_cmap, levels=10)
    im = axs[1].imshow(np.transpose(attn, (1,2,0))[:,:,0], cmap='jet', alpha=0.5)
    cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    # plt.imshow(np.transpose(image.detach().numpy(), (1, 2, 0)), interpolation='nearest')
    # plt.imshow(np.transpose(attn.squeeze(0).numpy(), (1, 2, 0)), interpolation='nearest')
    axs[1].set_title("ATTN, Predicition value " + str(pred.item()) + "actual " + str(act.item()))

    from captum.attr import IntegratedGradients
    from captum.attr import visualization as viz

    from matplotlib.colors import LinearSegmentedColormap

    model.return_attn = False
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(imout.unsqueeze(0), target=None, n_steps=100)
    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=256)

    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(imout.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 outlier_perc=1, plt_fig_axis=(fig,axs[2]))

    plt.show()
