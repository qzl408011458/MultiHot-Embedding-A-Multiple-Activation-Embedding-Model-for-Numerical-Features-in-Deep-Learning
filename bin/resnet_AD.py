# %%
import os
import sys

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move one directory up to get the project root
project_root = os.path.dirname(script_dir)

# Append the project root to the system's Python path
sys.path.append(project_root)

import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
from torch import Tensor

import lib

from AD_Embedding import AD_Embedding


from collections import Counter
from sklearn.preprocessing import LabelEncoder
# %%


import os
import json
# Get the JSON file path from the environment variable
json_file_path = os.environ['JSON_FILE_PATH']

# Open the JSON file and load its contents into a dictionary
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Extract the inv and bins values from the dictionary
inv = data['inv']
bins = data['bins']



class ResNet(nn.Module):
    def __init__(self, d_numerical, d, d_hidden_factor, n_layers, activation,
                 normalization, hidden_dropout, residual_dropout, d_out, categories=None, d_embedding=160, ad_params=None,
                 bins_type=None, bins=50):

        super(ResNet, self).__init__()
        self.bins_type = bins_type
        self.bins = ad_params['bins']

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        # d_in = d_numerical


        d_hidden = int(d * d_hidden_factor)


        self.ad_embedding = AD_Embedding(num_feature=ad_params['num_feature'],
                                         bins=ad_params['bins'],
                                         t=ad_params['t'],
                                         emb_size=ad_params['emb_size'])
        # MultiHot_Embedding
        # if multi_hot_params:
        #     self.multi_hot_embedding = MH_Embedding.MultiHot_Embedding(**multi_hot_params)
        #     d_in = multi_hot_params["num_feature"] * multi_hot_params["emb_size"]
        #     # MultiHot_Embedding input feature shape = multi_hot_params["num_feature"] * multi_hot_params["emb_size"]

        d_in = ad_params['num_feature'] * ad_params['emb_size']

        # Category embeddings
        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')


        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)
        print('bins, t, emb_size:',
              ad_params["bins"], ad_params["t"], ad_params["emb_size"])


    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = []

        # numerical MHE
        if x_num is not None:
            x_num_ad = self.ad_embedding(x_num.unsqueeze(1))
            # x_num_MHE.shape()=emb_size*num_feature
            # torch.unsqueeze(input, dim)
            x.append(x_num_ad)


        # Category MHE
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat.long() + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)



        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


# %%
if __name__ == "__main__":
    args, output = lib.load_config()

    # %%
    zero.set_randomness(args['seed'])
    dataset_dir = lib.get_path(args['data']['path'])
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir.name,
        'algorithm': Path(__file__).stem,
        **lib.load_json(output / 'stats.json'),
    }
    timer = zero.Timer()
    timer.run()

    D = lib.Dataset.from_dir(dataset_dir)
    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args['data'].get('cat_policy', 'indices'),
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
        seed=args['seed'],
    )
    if not isinstance(X, tuple):
        X = (X, None)

    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))
    lib.dump_pickle(y_info, output / 'y_info.pickle')
    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y


    X_num, X_cat = X


    # ####################################################################################
    # Apply MH_Embedding.bins_discrete() to numerical data
    # print('number_index_list::', number_index_list)
    # print('category_index_list::', category_index_list)
    # bins = 100
    # # inv = 9
    emb_size = 20
    t = 0.5


    X_num_train, X_num_val, X_num_test = X_num['train'], X_num['val'], X_num['test']
    # X_num_train, X_num_val, X_num_test = MH_Embedding.bins_discrete('efde', X_num_train, X_num_val, X_num_test, bins)

    X_num_train = X_num_train.cuda()
    X_num_val = X_num_val.cuda()
    X_num_test = X_num_test.cuda()

    X_num['train'], X_num['val'], X_num['test'] = X_num_train, X_num_val, X_num_test



    ###################################################################################

    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )
    args["model"]["d_embedding"] = args["model"].get("d_embedding", None)
    #########################################################################################
    model = ResNet(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=lib.get_categories(X_cat),
        ad_params={
            "emb_size": emb_size,
            # MHE_output_dimension=emb_size*num_feature
            "bins": bins,
            "num_feature": X_num['train'].shape[1],
            "device": device,
            "t": t,
        },
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        **args['model'],
    ).to(device)
    #########################################################################################
    stats['n_parameters'] = lib.get_n_parameters(model)
    optimizer = lib.make_optimizer(
        args['training']['optimizer'],
        model.parameters(),
        args['training']['lr'],
        args['training']['weight_decay'],
    )


    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    timer = zero.Timer()
    checkpoint_path = output / 'checkpoint.pt'

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': lib.get_lr(optimizer),
                    'batch_size': batch_size,
                    'epoch_size': stats['epoch_size'],
                    'n_parameters': stats['n_parameters'],
                }.items()
            )
        )

    @torch.no_grad()
    def evaluate(parts):
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            predictions[part] = (
                torch.cat(
                    [
                        model(
                            None if X_num is None else X_num[part][idx].cuda(),
                            None if X_cat is None else X_cat[part][idx].cuda(),
                        )
                        for idx in lib.IndexLoader(
                            D.size(part),
                            args['training']['eval_batch_size'],
                            False,
                            device,
                        )
                    ]
                )
                .cpu()
                .numpy()
            )
            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part].numpy(),  # type: ignore[code]
                predictions[part],  # type: ignore[code]
                'logits',
                y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', lib.make_summary(part_metrics))
        return metrics, predictions

    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stream': stream.state_dict(),
                'random_state': zero.get_random_state(),
                **{
                    x: globals()[x]
                    for x in [
                        'progress',
                        'stats',
                        'timer',
                        'training_log',
                    ]
                },
            },
            checkpoint_path,
        )
        lib.dump_stats(stats, output, final)
        lib.backup_output(output)

    # %%
    timer.run()
    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()

        model.train()
        epoch_losses = []
        for batch_idx in epoch:
            optimizer.zero_grad()
            loss = loss_fn(
                model(
                    None if X_num is None else X_num[lib.TRAIN][batch_idx].cuda(),
                    None if X_cat is None else X_cat[lib.TRAIN][batch_idx].cuda(),
                ),
                Y_device[lib.TRAIN][batch_idx],
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach())
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics[lib.VAL]['score'])

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail:
            break

    # %%
    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(lib.PARTS)
    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)
    stats['time'] = lib.format_seconds(timer())
    save_checkpoint(True)
    print('Done!')
