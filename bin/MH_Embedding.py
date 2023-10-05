# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# from sklearn.preprocessing import KBinsDiscretizer
#
#
# class MultiHot_Embedding(nn.Module):
#     def __init__(self, module, emb_size=20, total=100,
#                  inv=2, bins=50, num_feature=10, device='cuda', emb_hid_layers=0):
#         super(MultiHot_Embedding, self).__init__()
#         self.module = module
#         self.emb_size = emb_size
#         self.emb_hid_layers = emb_hid_layers
#
#         self.inv = inv
#         self.total = total
#         self.bins = bins
#         self.num_feature = num_feature
#
#
#         # build a multi-hot encoding matrix for bins (mapping index to multi-hot vector)
#         # step1: create a tensor in the ascending order (0, 1, 2,..., bins-1)
#         mh_bin = torch.tensor(range(bins), device=device)
#         # step2: one-hot encoding
#         mh_bin = F.one_hot(mh_bin, num_classes=total) # total = bins * 2
#         # step3: create padding zero tensor
#         padding = torch.zeros(mh_bin.shape, device=device)
#         # step4: initialize multi-hot matrix
#         mh_bin = torch.cat([padding, mh_bin, padding], dim=-1)
#         # step5: duplicate
#         emb_bin_temp = torch.zeros(mh_bin.shape, device=device)
#
#         for i in range(self.inv + 1):
#             temp_i = torch.roll(mh_bin, i, -1)
#             emb_bin_temp += temp_i
#         for i in range(self.inv + 1):
#             temp_i = torch.roll(mh_bin, -i, -1)
#             emb_bin_temp += temp_i
#
#         self.multiHot_bin = emb_bin_temp.float() - mh_bin
#         # step6: define fully-connected layers to embed 1 feature
#         mh_emb_layer_feat = nn.ModuleList()
#         for emb_layer_i in range(emb_hid_layers + 1):
#             if emb_layer_i == 0:
#                 mh_emb_layer_feat.append(nn.Linear(int(self.total * 3), self.emb_size, bias=False))
#             else:
#                 mh_emb_layer_feat.append(nn.Linear(self.emb_size, self.emb_size, bias=False))
#
#         # step7: multi-hot embedding layer
#         self.multiHot_embed = nn.ModuleList()
#         for i in range(num_feature):
#             self.multiHot_embed.append(mh_emb_layer_feat)
#
#     def forward(self, x):
#         out_emb_list = []
#         for i in range(self.num_feature):
#             # non-time-series task is 2-dimension, otherwise 3-dimension
#             x_shape_len = len(x.shape)
#             if x_shape_len == 2:
#                 x_i = x[:, i]
#             if x_shape_len == 3:
#                 x_i = x[:, :, i]
#
#             xf_i = F.one_hot(x_i, num_classes=self.bins).float()
#             # multi-hot encoding
#             x_i = torch.matmul(xf_i, self.multiHot_bin)
#             # embedding
#             for layer_j in range(self.emb_hid_layers + 1):
#                 x_i = self.multiHot_embed[i][layer_j](x_i)
#
#             out_emb_list.append(x_i)
#
#         # concatenate in feature dimension
#         out_emb = torch.cat(out_emb_list, dim=-1)
#         return out_emb
#
# def bins_discrete(bins_type, data_train, data_test,bins=100):
#     train_shape = data_train.shape
#     test_shape = data_test.shape
#
#     # only 3-dimension data will transform to 2-dimension
#     data_train = data_train.reshape(-1, train_shape[-1])
#     data_test = data_test.reshape(-1, test_shape[-1])
#
#     if bins_type == 'efde':
#         kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
#         kbd.fit(data_train[:, :-1])
#
#         data_train[:, :-1] = torch.from_numpy(kbd.transform(data_train[:, :-1]))
#         data_test[:, :-1] = torch.from_numpy(kbd.transform(data_test[:, :-1]))
#     if bins_type == 'ewde':
#         kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
#         kbd.fit(data_train[:, :-1])
#
#         data_train[:, :-1] = torch.from_numpy(kbd.transform(data_train[:, :-1]))
#         data_test[:, :-1] = torch.from_numpy(kbd.transform(data_test[:, :-1]))
#
#     data_train = data_train.reshape(train_shape)
#     data_test = data_test.reshape(test_shape)
#     return data_train, data_test
# if __name__ == '__main__':
#     # e.g.
#     # pseudo code in data processing:
#     # for num data:
#     data_train, data_test = bins_discrete(bins_type, data_train, data_test, bins)
#
#     # pseudo code in model:
#     def init():
#         self.me = MultiHot_Embedding(...)
#
#     def forward(x):
#         if num:
#             # for num feature
#             me = self.me(x)
#             x = me(x)
#         ...
##########################################################################################################
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter

class MultiHot_Embedding(nn.Module):
    def __init__(self, module, emb_size=20, total=100,
                 inv=3, bins=50, num_feature=10, device='cuda', emb_hid_layers=0):
        super(MultiHot_Embedding, self).__init__()
        self.module = module
        self.emb_size = emb_size
        self.emb_hid_layers = emb_hid_layers

        self.inv = inv
        self.total = total
        self.bins = bins
        self.num_feature = num_feature

        # self.tanh = nn.Tanh()


        # build a multi-hot encoding matrix for bins (mapping index to multi-hot vector)
        # step1: create a tensor in the ascending order (0, 1, 2,..., bins-1)
        mh_bin = torch.tensor(range(bins), device=device)
        # step2: one-hot encoding
        mh_bin = F.one_hot(mh_bin, num_classes=total) # total = bins * 2
        # step3: create padding zero tensor
        padding = torch.zeros(mh_bin.shape, device=device)
        # step4: initialize multi-hot matrix
        mh_bin = torch.cat([padding, mh_bin, padding], dim=-1)
        # step5: duplicate
        emb_bin_temp = torch.zeros(mh_bin.shape, device=device)

        for i in range(self.inv + 1):
            temp_i = torch.roll(mh_bin, i, -1)
            emb_bin_temp += temp_i
        for i in range(self.inv + 1):
            temp_i = torch.roll(mh_bin, -i, -1)
            emb_bin_temp += temp_i

        self.multiHot_bin = emb_bin_temp.float() - mh_bin
        # step6: define fully-connected layers to embed 1 feature
        mh_emb_layer_feat = nn.ModuleList()
        for emb_layer_i in range(emb_hid_layers + 1):
            if emb_layer_i == 0:
                mh_emb_layer_feat.append(nn.Linear(int(self.total * 3), self.emb_size, bias=False))
            else:
                mh_emb_layer_feat.append(nn.Linear(self.emb_size, self.emb_size, bias=False))

        # step7: multi-hot embedding layer
        self.multiHot_embed = nn.ModuleList()
        for i in range(num_feature):
            self.multiHot_embed.append(mh_emb_layer_feat)

    def forward(self, x):
        out_emb_list = []
        for i in range(self.num_feature):
        # for i in range(self.num_feature):
            # non-time-series task is 2-dimension, otherwise 3-dimension
            x_shape_len = len(x.shape)
            if x_shape_len == 2:
                x_i = x[:, i]
            if x_shape_len == 3:
                x_i = x[:, :, i]

            xf_i = F.one_hot(x_i.long(), num_classes=self.bins).float()
            # multi-hot encoding
            x_i = torch.matmul(xf_i, self.multiHot_bin)
            # embedding
            for layer_j in range(self.emb_hid_layers + 1):
                # x_i = self.tanh(self.multiHot_embed[i][layer_j](x_i))
                x_i = self.multiHot_embed[i][layer_j](x_i)

            out_emb_list.append(x_i)

        # concatenate in feature dimension
        out_emb = torch.cat(out_emb_list, dim=-1)
        return out_emb
def bins_discrete(bins_type, data_train, data_val, data_test, bins=100):

    train_shape = data_train.shape
    val_shape = data_val.shape
    test_shape = data_test.shape
    # if data_test is not None:
    #     test_shape = data_test.shape
    #     data_test = data_test.reshape(-1, test_shape[-1])
    # else:
    #     test_shape = None
    #
    # data_train = data_train.reshape(-1, train_shape[-1])

    # only 3-dimension data will transform to 2-dimension
    data_train = data_train.reshape(-1, train_shape[-1])
    data_val = data_val.reshape(-1, val_shape[-1])
    data_test = data_test.reshape(-1, test_shape[-1])

    if bins_type == 'efde':
        kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        kbd.fit(data_train.cpu().numpy())
        # define fun for fit
        data_train = torch.from_numpy(kbd.transform(data_train.cpu().numpy()))
        if data_test is not None:
            data_test = torch.from_numpy(kbd.transform(data_test.cpu().numpy()))
            data_val = torch.from_numpy(kbd.transform(data_val.cpu().numpy()))

    if bins_type == 'ewde':
        kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        kbd.fit(data_train.cpu().numpy())

        data_train = torch.from_numpy(kbd.transform(data_train.cpu().numpy()))
        if data_test is not None:
            data_test = torch.from_numpy(kbd.transform(data_test.cpu().numpy()))
            data_val = torch.from_numpy(kbd.transform(data_val.cpu().numpy()))


    data_train = data_train.reshape(train_shape)

    if data_test is not None:
        data_test = data_test.reshape(test_shape)

    return data_train, data_val, data_test


def test_MultiHot_Embedding_and_bins_discrete():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.manual_seed(42)  # for reproducibility
    data_train = torch.rand((100, 10)) * 10  # 100 samples, 10 features.
    data_test = torch.rand((50, 10)) * 10  # test data

    # Moving data to the desired device
    data_train = data_train.to(device)
    data_test = data_test.to(device)

    bins = 50
    bins_type = 'ewde'  # You can also test with 'ewde'

    #Test the bins_discrete function
    data_train_discrete, data_test_discrete = bins_discrete(bins_type, data_train, data_test, bins)

    # Print some values to check
    print("Original Train Data:")
    print(data_train[:5])
    print("\nDiscretized Train Data:")
    print(data_train_discrete[:5])
    print("\nDiscretized Test Data:")
    print(data_test_discrete[:5])
    #Test the MultiHot_Embedding class
    embedding_layer = MultiHot_Embedding('dasd', num_feature=10).to(device)
    embedded_data = embedding_layer(data_train_discrete)

    print("\nEmbedded Data Shape:", embedded_data.shape)
    print("Embedded Data Sample:")
    print(embedded_data[0])

    return True


# if __name__ == '__main__':
#     if test_MultiHot_Embedding_and_bins_discrete():
#         print("The functions yes.")
#     else:
#         print("Nay.")

