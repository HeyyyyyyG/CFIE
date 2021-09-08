UNCOVERING MAIN CAUSALITIES FOR LONG-TAILED INFORMATION EXTRACTION 
==========

This repo contains the *PyTorch* code for the paper "Uncovering Main Causalities for Long-tailed Information Extraction".

## Requirements

- Python 3 (tested on 3.6.5)
- PyTorch (> 1.0)
- tqdm
- unzip, wget (for downloading only)



## Training
To train a baseline Dep-guided LSTM model, run:
```python
python train.py --id 0 --seed 0 --effect_type None --lr 0.001 --num_epoch 1000 --data_dir dataset/atis --vocab_dir dataset/atis 
```

To train CAUSAL model with based on a predefined casual graph, run:
```python
python train.py --id 1 --seed 0 --effect_type CAUSAL --lr 0.001 --num_epoch 1000 --data_dir dataset/atis --vocab_dir dataset/atis 
```

Model checkpoints and logs will be saved to `./saved_models/00`.

Model checkpoints and logs will be saved to `./saved_models/01`.

For details on the use of other parameters, please refer to `train.py`.

## Evaluation

For Dep-guided LSTM model, run:

```python
python eval.py --model_dir saved_models/00 --data_dir dataset/atis --dataset test --effect_type None
```
For the CAUSAL model:

1. To find the optimal value of alpha on dev set, you can try different values, e.g. 

   ```python
   python eval.py --model_dir saved_models/01 --data_dir dataset/atis --dataset dev --effect_type CAUSAL --alpha 1.2
   ```

   We observe that the optimal value of alpha is around 1.2. 

2.  Run the model on test set by 

   ```python
   python eval.py --model_dir saved_models/01 --data_dir dataset/atis --dataset test --effect_type CAUSAL --alpha 1.2
   ```

This will use the `best_model.pt` file by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.



## Notice

Due to copyright issues, we only include ATIS dataset here. 

We also release a version of pretrained model for ATIS dataset. (saved_models/02) 

You can run the following command to reproduce similar results.

```python
python eval.py --model_dir saved_models/02 --data_dir dataset/atis --dataset test --effect_type CAUSAL --alpha 1.2
```

