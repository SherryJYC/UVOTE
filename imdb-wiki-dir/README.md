# IMDB-WIKI-DIR
## Installation

#### Prerequisites

1. Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```

2. __(Optional)__ We have provided required IMDB-WIKI-DIR meta file `imdb_wiki.csv` to set up balanced val/test set in folder `./data`. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Code Overview

#### Main Files

- `train_gradual.py`: main training and evaluation script
- `create_imdb_wiki.py`: create IMDB-WIKI raw meta data
- `preprocess_imdb_wiki.py`: create IMDB-WIKI-DIR meta file `imdb_wiki.csv` with balanced val/test set

## Main arguments
- `--data_dir`: data directory to place data and meta file
- `--num_branch`: number of branch for model
- `--loss`: training loss type
- `--resume`: path to resume checkpoint (for both training and evaluation)
- `--evaluate`: evaluate only flag

#### Training
```bash
# for example, train with 3-expert model
python train_gradual.py --loss l1nll --num_branch 3 --dynamic_loss
```

#### Evaluation
```bash
python train_gradual.py --evaluate --resume MODEL_CHECKPOINT  [other model settings: e.g.--loss l1nll --num_branch 3]
```

## Pretrained model
- [model for imdb-wiki](https://share.phys.ethz.ch/~pf/yujiangdata/mouv/imdb-wiki-dir/imdb_wiki_resnet50_3_dyL_adam_l1nll.zip)