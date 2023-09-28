import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import KBinsDiscretizer


class AD_Layer(nn.Module):
    def __init__(self, bins=30, t=0.5, emb_size=100):
        super(AD_Layer, self).__init__()
        self.t = t
        self.emb_size = emb_size
        self.layer1 = nn.Sequential(
            nn.Linear(1, bins, bias=False),
            nn.LeakyReLU())
        self.lin = nn.Linear(bins, bins, bias=False)
        self.layer2 = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Linear(bins, self.emb_size, bias=False))

    def forward(self, x):
        y = self.layer1(x)
        y = self.lin(y)+0.1*y
        y = y*self.t
        y = self.layer2(y)

        return y

class AD_Embedding(nn.Module):
    def __init__(self, num_feature=30, bins=30, t=0.5, emb_size=100):
        super(AD_Embedding, self).__init__()
        self.num_feature = num_feature
        ad_layer = AD_Layer(bins, t, emb_size)
        self.ad_embed = nn.ModuleList()
        for i in range(num_feature):
            self.ad_embed.append(ad_layer)

    def forward(self, x):
        out_emb_list = []
        for i in range(self.num_feature):
            # non-time-series task is 2-dimension, otherwise 3-dimension
            # x_shape_len = len(x.shape)
            # if x_shape_len == 2:
            #     x_i = x[:, i]
            # if x_shape_len == 3:
            #     x_i = x[:, :, i]
            x_i = x[:, :, i]
            # # embedding
            # for layer_j in range(self.emb_hid_layers + 1):
            x_i = self.ad_embed[i](x_i)

            out_emb_list.append(x_i)

        # concatenate in feature dimension
        out_emb = torch.cat(out_emb_list, dim=-1)
        return out_emb