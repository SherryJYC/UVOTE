# Wind-DIR
## Installation

#### Prerequisites

- set up radiant mlhub
```
https://radiant-mlhub.readthedocs.io/en/latest/getting_started.html#installation
```
- download data
```
python download_wind_data.py
```
- (optional) preprocess data (into csv)
```
python preprocess_wind.py
```

## Code Overview

#### Main Files

- `train_gradual.py`: main training and evaluation script
- `create_agedb.py`: create AgeDB raw meta data
- `preprocess_agedb.py`: create AgeDB-DIR meta file `agedb.csv` with balanced val/test set

## Main arguments
- `--data_dir`: data directory to place data and meta file
- `--num_branch`: number of branch for model
- `--loss`: training loss type
- `--resume`: path to resume checkpoint (for both training and evaluation)
- `--evaluate`: evaluate only flag

#### Training
```bash
# for example, train with 2-expert model
python train_gradual.py --loss l1nll --num_branch 2 --dynamic_loss
```

#### Evaluation
```bash
python train_gradual.py --evaluate --resume MODEL_CHECKPOINT  [other model settings: e.g.--loss l1nll --num_branch 2]
```

## Pretrained model
- [model for wind](https://share.phys.ethz.ch/~pf/yujiangdata/mouv/wind-dir/wind_resnet18_2_dyL_adam_l1nll.zip)


