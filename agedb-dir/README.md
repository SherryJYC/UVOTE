# AgeDB-DIR
## Installation

#### Prerequisites

1. Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data` 

2. __(Optional)__ We have provided required AgeDB-DIR meta file `agedb.csv` to set up balanced val/test set in folder `./data`. To reproduce the results in the paper, please directly use this file. If you want to try other different balanced splits, you can generate it using

```bash
python data/create_agedb.py
python data/preprocess_agedb.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL

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
- [model for agedb](https://share.phys.ethz.ch/~pf/yujiangdata/mouv/agedb-dir/agedb_resnet50_2_dyL_adam_l1nll.zip)

