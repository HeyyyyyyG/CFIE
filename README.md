UNCOVERING MAIN CAUSALITIES FOR LONG-TAILED INFORMATION EXTRACTION 
==========

This repo contains the *PyTorch* implementation for our proposed CFIE in EMNLP 2021 paper "Uncovering Main Causalities for Long-tailed Information Extraction".

## Overview of CFIE

### Abstract

Information Extraction (IE) aims to extract structural information from unstructured texts. In practice, long-tailed distributions caused by the selection bias of a dataset, may lead to spurious correlations between entities and labels in the conventional likelihood models. This motivates us to propose counterfactual IE (CFIE), a novel framework that aims to uncover the main causalities behind data in the view of causal inference. Specifically, 1) we first introduce a unified structural causal model (SCM) for various IE tasks, describing the relationships among variables; 2) with our SCM, we then generate counterfactuals based on an explicit language structure to better calculate the direct causal effect during the inference stage; 3) we further propose a novel debiasing approach to yield more robust predictions. Experiments on three IE tasks across five public datasets show the effectiveness of our CFIE model in mitigating the spurious correlation issues.

### Framework

<div align =center width = 80% height = 80%>

<img src="./fig/arch1.png" alt="arch1" style="zoom:30%;" />

<img src="./fig/arch2.png" alt="arch2" style="zoom:30%;" />
</div>

### Results for RE

TBA





## Requirements

- Python 3 (tested on 3.6.5)
- PyTorch (> 1.0)
- tqdm
- numpy


## Training

TBA

## Evaluation

TBA



## Notice

TBA

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

