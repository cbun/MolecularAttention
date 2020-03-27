import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn import metrics
from metrics import trackers

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    modelname = args.model_file.split(".")[0]
    history = torch.load(args.model_file)["history"]
    history.plot_loss(save_file=modelname+"loss.png")
    history.plot_metric(save_file=modelname+"metric.png")
