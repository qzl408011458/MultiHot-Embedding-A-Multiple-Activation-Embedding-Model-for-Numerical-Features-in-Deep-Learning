# MultiHot-Embedding-in-Tabular-Learning

This repository is for the paper "MultiHot Embedding: A Multiple Activation Embedding Model for Numerical Features in Deep Learning".

The models used in the experiments can be run by following command:

```python
python run.py -m <input-module> -b <backbone> -d <dataset> 
```
There are three options provided for combining different moduls,
backbones and datasets. "-m" is to select input process module
like AD and MH in the paper. "-b" is to select backbone like MLP
and ResNet. "-d" is for the selected dataset.

The folder "configs" include the model configurations saved in
the subfolders named by datasets. The most essential hyperparameters
are inv, bins, emb_size, t in code as the same as h, K, m, $\lambda$

