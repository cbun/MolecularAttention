import ipdb
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch


class ImageModelPartial(nn.Module):
    def __init__(
        self,
        intermediate_rep=256,
        nheads=0,
        outs=1,
        dr=0,
        classification=False,
        linear_layers=2,
        model_path=None,
        pretrain=True,
        merge_dim=256,
    ):
        super(ImageModelPartial, self).__init__()
        self.return_attn = True
        self.outs = outs
        self.nheads = nheads

        if model_path is None:
            resnet18 = models.resnet101(pretrained=pretrain)
        else:
            resnet18 = models.resnet101(pretrained=False)
            if pretrain:
                resnet18.load_state_dict(torch.load(model_path))

        resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

        if self.nheads > 0:
            self.resnet181 = nn.Sequential(*list(resnet18.children())[:5])
            self.resnet182 = nn.Sequential(*list(resnet18.children())[5:])
            self.attention = nn.Sequential(
                nn.Conv2d(256, nheads, kernel_size=5, padding=2, stride=1),
            )
        else:
            self.resnet181 = nn.Sequential(*list(resnet18.children()))

        self.model = nn.Sequential(
            nn.Linear(2048, intermediate_rep),
            nn.BatchNorm1d(intermediate_rep),
            nn.Linear(intermediate_rep, merge_dim),
            nn.ReLU(),
            nn.Dropout(dr),
        )

        # self.linears = nn.ModuleList()
        # for i in range(linear_layers):
        #     self.linears.append(nn.Linear(intermediate_rep, intermediate_rep))
        #     self.linears.append(nn.ReLU(),)
        #     self.linears.append(nn.Dropout(dr))

        # self.linear = nn.Sequential(*self.linears)

        # self.prop_model = nn.Sequential(
        #     self.linear,
        #     nn.Linear(intermediate_rep, intermediate_rep),
        #     nn.ReLU(),
        #     nn.Dropout(dr),
        #     nn.Linear(intermediate_rep, self.outs),
        # )

    def forward(self, features):
        if self.nheads > 0:
            image = self.resnet181(features)
            attention = self.attention(image)
            attention = nn.functional.softmax(
                attention.view(attention.shape[0], self.nheads, -1), dim=-1
            ).view(attention.shape)
            attention = attention.repeat([1, int(256 / self.nheads), 1, 1])
            image = self.resnet182(image * attention)
            image = image.view(features.shape[0], -1)
        else:
            image = self.resnet181(features).view(features.shape[0], -1)
            attention = torch.zeros((features.shape[0], 1, 1, 1))

        # if self.return_attn:
        #     return self.prop_model(self.model(image)), attention
        # else:
        #     return self.prop_model(self.model(image))

        # if self.return_attn:
        #     return self.prop_model(self.model(image)), attention
        # else:
        # TODO attention
        return self.model(image)


class DescriptorModelPartial(nn.Module):
    def __init__(self, input_dim, merge_dim=256, dropout_rate=0.1):
        super(DescriptorModelPartial, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, merge_dim)
        # self.fc3 = nn.Linear(125, 60)
        # self.fc4 = nn.Linear(60, 30)
        # self.fc5 = nn.Linear(30, 1)

        # self.bn0 = nn.BatchNorm1d(num_features=input_dim)
        # self.bn1 = nn.BatchNorm1d(num_features=250)
        # self.bn2 = nn.BatchNorm1d(num_features=125)
        # self.bn3 = nn.BatchNorm1d(num_features=60)
        # self.bn4 = nn.BatchNorm1d(num_features=30)

        # torch.nn.init.normal_(self.fc5.weight.data)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch_seq_arr):
        # x = F.relu(self.bn1(self.fc1(self.dropout(self.bn0(batch_seq_arr)))))
        # x = F.relu(self.bn2(self.fc2(self.dropout(x))))
        # x = F.relu(self.bn3(self.fc3(self.dropout(x))))
        # x = F.relu(self.bn4(self.fc4(self.dropout(x))))
        # predictions = self.fc5(self.dropout((((x)))))
        predictions = self.fc2(self.dropout(self.fc1(batch_seq_arr)))
        return predictions


class FingerprintModelPartial(nn.Module):
    def __init__(self, input_dim, merge_dim=256, dropout_rate=0.1):
        super(DescriptorModelPartial, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, merge_dim)
        # self.fc3 = nn.Linear(125, 60)
        # self.fc4 = nn.Linear(60, 30)
        # self.fc5 = nn.Linear(30, 1)

        # self.bn0 = nn.BatchNorm1d(num_features=input_dim)
        # self.bn1 = nn.BatchNorm1d(num_features=250)
        # self.bn2 = nn.BatchNorm1d(num_features=125)
        # self.bn3 = nn.BatchNorm1d(num_features=60)
        # self.bn4 = nn.BatchNorm1d(num_features=30)

        # torch.nn.init.normal_(self.fc5.weight.data)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch_seq_arr):
        # x = F.relu(self.bn1(self.fc1(self.dropout(self.bn0(batch_seq_arr)))))
        # x = F.relu(self.bn2(self.fc2(self.dropout(x))))
        # x = F.relu(self.bn3(self.fc3(self.dropout(x))))
        # x = F.relu(self.bn4(self.fc4(self.dropout(x))))
        # predictions = self.fc5(self.dropout((((x)))))
        predictions = self.fc2(self.dropout(self.fc1(batch_seq_arr)))
        return predictions


class MultiModalModel(nn.Module):
    def __init__(
        self,
        input_dim_descriptors,
        input_dim_fingerprints,
        linear_layers=2,
        intermediate_rep=512,
        dr=0.15,
        merge_dim=256,
    ):
        super(MultiModalModel, self).__init__()

        self.image_model = ImageModelPartial()
        self.descriptor_model = DescriptorModelPartial(input_dim_descriptors)
        self.fingerprint_model = FingerprintModelPartial(input_dim_fingerprints)
        num_modalities = 1

        self.linears = nn.ModuleList()
        for i in range(linear_layers):
            self.linears.append(
                nn.Linear(merge_dim * num_modalities, merge_dim * num_modalities)
            )
            self.linears.append(nn.ReLU(),)
            self.linears.append(nn.Dropout(dr))

        self.linear = nn.Sequential(*self.linears)

        self.model = nn.Sequential(
            self.linear,
            nn.Linear(intermediate_rep, intermediate_rep),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(intermediate_rep, 1),
        )

    def forward(self, img_features, desc_features, fng_features):

        x_image = self.image_model(img_features)
        x_desc = self.descriptor_model(desc_features)
        x_fng = self.descriptor_model(fng_features)
        x = torch.cat((x_image, x_desc, x_fng), dim=1)
        return self.model(x), None  # TODO attn
