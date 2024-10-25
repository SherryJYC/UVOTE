# UVOTE
This repository contains the implementation code for paper: <br/>
**[GCPR 2024] Uncertainty Voting Ensemble for Imbalanced Deep Regression** <br/>
Yuchang Jiang, Vivien Sainte Fare Garnot, Konrad Schindler, and Jan Dirk Wegner <br/>
[[Paper]](https://arxiv.org/abs/2305.15178)

## Introduction
Data imbalance is ubiquitous when applying machine learning to real-world problems, particularly regression problems. If training data are imbalanced, the learning is dominated by the densely covered regions of the target distribution and the learned regressor tends to exhibit poor performance in sparsely covered regions. Beyond standard measures like oversampling or reweighting, there are two main approaches to handling learning from imbalanced data. For regression, recent work leveraged the continuity of the distribution, while for classification, the trend has been to use ensemble methods, allowing some members to specialize in predictions for sparser regions. In our method, named UVOTE, we integrate recent advances in probabilistic deep learning with an ensemble approach for imbalanced regression. We replace traditional regression losses with negative log-likelihood, which also predicts sample-wise aleatoric uncertainty. Our experiments show that this loss function handles imbalance better. Additionally, we use the predicted aleatoric uncertainty values to fuse the predictions of different expert models in the ensemble, eliminating the need for a separate aggregation module. We compare our method with existing alternatives on multiple public benchmarks and show that UVOTE consistently outperforms the prior art, while at the same time producing better-calibrated uncertainty estimates.

## Usage
We separate the codebase for different datasets into different subfolders. Please go into the subfolders for more information (e.g., installation, dataset preparation, training, evaluation & models).

[AgeDB-DIR](https://github.com/SherryJYC/UVOTE/tree/main/agedb-dir)  |  [IMDB-WIKI-DIR](https://github.com/SherryJYC/UVOTE/tree/main/imdb-wiki-dir) | [WIND-DIR](https://github.com/SherryJYC/UVOTE/tree/main/wind-dir) |  [STS-B-DIR](https://github.com/SherryJYC/UVOTE/tree/main/sts-b-dir)

## Citations
```
@misc{jiang2024uncertaintyvotingensembleimbalanced,
      title={Uncertainty Voting Ensemble for Imbalanced Deep Regression}, 
      author={Yuchang Jiang and Vivien Sainte Fare Garnot and Konrad Schindler and Jan Dirk Wegner},
      year={2024},
      eprint={2305.15178},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.15178}, 
}
```


## Acknowledgment
The code is based on:
- [Deep Imbalanced Regression Benchmark, 2021](https://github.com/YyzHarry/imbalanced-regression/tree/main)
- [RankSim, 2022](https://github.com/BorealisAI/ranksim-imbalanced-regression)
