# MultiHot-Embedding-in-Tabular-Learning

This repository is for the paper "MultiHot Embedding: A Multiple Activation Embedding Model for Numerical Features in Deep Learning".

## 1. Data Preparation
Before training models, the folder "data" should be created in the 
project root to save the datasets, which can be downloaded in the
repo https://github.com/yandex-research/tabular-dl-revisiting-models#33-data.

## 2. Configurations

The folder "configs" include the model configurations saved in
the subfolders named by datasets. The most essential hyperparameters
are inv, bins, emb_size, t in code as the same as $h, K, m, \tau$ 
in paper. All the model scripts are saved in the folder "bin"
.
## 3. Model Training
The models used in the experiments can be trained by following command:

```python
python run.py -m <input-module> -b <backbone> -d <dataset> 
```
There are three options provided for combining different moduls,
backbones and datasets. "-m" is to select input process module
like AD and MH in the paper. "-b" is to select backbone like MLP
and ResNet. "-d" is for the selected dataset. Please see run.py 
for more details about the available args.

The default path of saved training results is the subfolder also
named by dataset in "output".







