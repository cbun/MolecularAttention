import torch.nn as nn
import torchvision.models as models
import torch
class DescrModel(nn.Module):

    def __init__(self, flen, intermediate_rep=128, outs=1, dr=0, classifacation=False):
        super(DescrModel, self).__init__()
        print("Length is ", flen)
        self.model = nn.Sequential(
            nn.Linear(flen, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dr),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dr),

            nn.Linear(128, intermediate_rep),
            nn.BatchNorm1d(intermediate_rep),
            nn.ReLU(),
            nn.Dropout(dr),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dr),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dr),

            nn.Linear(32, outs)
        )    
       


    def forward(self, features):
        return self.model(features)
