# STS-B-DIR
## Installation

#### Prerequisites

1. Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

2. __(Optional)__ We have provided both original STS-B dataset and our created balanced STS-B-DIR dataset in folder `./glue_data/STS-B`. To reproduce the results in the paper, please use our created STS-B-DIR dataset. If you want to try different balanced splits, you can delete the folder `./glue_data/STS-B` and run

```bash
python glue_data/create_sts.py
```

#### Dependencies

The required dependencies for this task are quite different to other three tasks, so it's better to create a new environment for this task. If you use conda, you can create the environment and install dependencies using the following commands:

```bash
conda create -n sts python=3.6
conda activate sts
# PyTorch 0.4 (required) + Cuda 9.2
conda install pytorch=0.4.1 cuda92 -c pytorch
# other dependencies
pip install -r requirements.txt
# The current latest "overrides" dependency installed along with allennlp 0.5.0 will now raise error. 
# We need to downgrade "overrides" version to 3.1.0
pip install overrides==3.1.0
```

## Code Overview

#### Main Files

- `train_morebranch.py`: main training and evaluation script
- `create_sts.py`: download original STS-B dataset and create STS-B-DIR dataset with balanced val/test set 

#### Main Arguments
- `--num_branch`: number of branch for model
- `--reweight`: cost-sensitive re-weighting scheme to use
- `--loss`: training loss type
- `--resume`: whether to resume training (only for training)
- `--evaluate`: evaluate only flag

#### Training
```bash
# for example, train with 3-expert model
python train_morebranch.py --loss l1nll --num_branch 3 --dynamic_loss
```

#### Evaluation
```bash
python train_morebranch.py [...evaluation model arguments...] --evaluate --eval_model <path_to_evaluation_ckpt>
```

## Pretrained model
- [model for sts-b](https://share.phys.ethz.ch/~pf/yujiangdata/mouv/sts-b-dir/model_state_best.th)

