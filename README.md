UNCOVERING MAIN CAUSALITIES FOR LONG-TAILED INFORMATION EXTRACTION 
==========

This repo contains the *PyTorch* implementation for our proposed CFIE in EMNLP 2021 paper "Uncovering Main Causalities for Long-tailed Information Extraction".

## Overview of CFIE

### Abstract

Information Extraction (IE) aims to extract structural information from unstructured texts. In practice, long-tailed distributions caused by the selection bias of a dataset, may lead to spurious correlations between entities and labels in the conventional likelihood models. This motivates us to propose counterfactual IE (CFIE), a novel framework that aims to uncover the main causalities behind data in the view of causal inference. Specifically, 1) we first introduce a unified structural causal model (SCM) for various IE tasks, describing the relationships among variables; 2) with our SCM, we then generate counterfactuals based on an explicit language structure to better calculate the direct causal effect during the inference stage; 3) we further propose a novel debiasing approach to yield more robust predictions. Experiments on three IE tasks across five public datasets show the effectiveness of our CFIE model in mitigating the spurious correlation issues.

## Requirements

- Python 3 (tested on 3.6.5)
- PyTorch (> 1.0)
- tqdm
- unzip, wget (for downloading only)

## Preparation

The code requires that you have access to the NYT24 dataset. 

First, download and unzip GloVe vectors from the Stanford NLP group website, with:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/nyt24 dataset/nyt24 --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Training
You need to indicate the dataset firstly in the train.py, e.g.,
```
DATASET = NYT24
```

And change parameters (by commenting) in the utils.constant.py accordingly for different datasets as they have different NER/RE labels. It is for NYT24 by default. 

To train a baseline Contextualized GCN (C-GCN) model, run:
```
python3 train.py --id CGCN --seed 0 --effect_type None --lr 0.3 --num_epoch 40 --pooling max --mlp_layers 2 --pooling_l2 0.003
```
Model checkpoints and logs will be saved to `./saved_models/CGCN`.

To train TDE model with based on a predefined casual graph, run:
```
python3 train.py --id TDE --seed 0 --effect_type TDE --lr 0.03 --num_epoch 40 --optim adam --macro
```


Model checkpoints and logs will be saved to `./saved_models/<id>`.

For details on the use of other parameters, such as the pruning distance k, please refer to `train.py`.

## Evaluation

To run evaluation on the test set for CGCN, run:
```
python eval.py saved_models/CGCN --dataset test
```

or for the TDE model, run:
```
python eval.py saved_models/TDE --dataset test
```
The results are shown as follows. We can achieve a much better mean recall compared with C-GCN. I believe we can further improve the mean precision and mean F1 by finetuning.

This will use the `best_model.pt` file by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.

## Other baselines
### Retrain

Reload a pretrained model and finetune it, run:
```
python train.py --load --model_file saved_models/01/best_model.pt --optim sgd --lr 0.001
```

### Use BERT
Added use_bert flag for train and eval
```
python train.py ... --use_bert
python eval.py ... --use_bert
```

### Decoupled Training
#### Classifier Retraining
To do classifier retraining (cRT), add the following to the train command:
```
python train.py ... --crt --load --model_file saved_models/<model_dir>/best_model.pt --num_epoch 10
```
Then at evaluation, specified the new model:
```
python eval.py saved_models/<model_dir> --dataset test --model best_retrain_model.pt
```
#### Tau-normalize
After training, directly run:
```
python eval.py saved_models/<model_dir> --dataset test  --use_tau --tau 0.5
```

### LWS
After training, add the following to the train command:
```
python train.py ... --lws --load --model_file saved_models/<model_dir>/best_model.pt --num_epoch 10
```
Then at evaluation, specified the new model:
```
python eval.py saved_models/<model_dir> --dataset test --model best_lws_model.pt
```

### Balanced Losses
Use "--loss" to choose loss functions from FocalLoss, LiftedLoss and DiceLoss (default is cross entropy). E.g.:
```
python train.py ... --loss focal
```

## Citation

```
@inproceedings{nan2021uncovering,
  title={Uncovering Main Causalities for Long-tailed Information Extraction},
  author={Nan, Guoshun and Zeng, Jiaqi and Qiao, Rui and Guo, Zhijiang and Lu, Wei},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={9683--9695},
  year={2021}
}
```
