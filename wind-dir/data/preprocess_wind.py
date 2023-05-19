import pandas as pd
import os
import numpy as np
import json
from random import shuffle

data_dir = 'data'
dataset_name = 'nasa_tropical_storm_competition'

def processCSV(split='train'):
    print(f'>>> process data for {split}')
    storm_ids = []
    image_ids = []
    image_seqs = []
    image_paths = []
    wind_speeds = []
    relative_times = []

    label_dir = f'{data_dir}/{dataset_name}/{dataset_name}_{split}_labels'
    for f in os.listdir(f'{label_dir}'):
        if not 'storm' in f:
            continue

        info = f.split('_')
        if int(info[-1]) < 2: # image0, image1 skipped for 3-image combination
            continue

        storm_ids.append(info[-2])
        image_ids.append('_'.join(info[-2:]))
        image_seqs.append(int(info[-1]))
        
        source_path = f'{dataset_name}_{split}_source/{dataset_name}_{split}_source_{info[-2]}_{info[-1]}'
        image_paths.append(f'{dataset_name}/{source_path}/image.jpg')

        with open (f'{label_dir}/{f}/vector_labels.json') as f:
            train_json = json.load(f)
        wind_speeds.append(int(train_json['wind_speed']))

        with open (f'{data_dir}/{dataset_name}/{source_path}/features.json') as f:
            source_json = json.load(f)
        relative_times.append(int(source_json['relative_time']))

    df = pd.DataFrame(storm_ids, columns=['storm_id'])
    df['image_id'] = image_ids
    df['image_path'] = image_paths
    df['image_seq'] = image_seqs
    df['wind_speed'] = wind_speeds
    df['relative_time'] = relative_times

    out_file = f'{data_dir}{split}.csv'

    df.to_csv(out_file, index=False)
    return out_file

# get train & test csv
train_csv = processCSV(split='train')
test_csv = processCSV(split='test')

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_storm = train_df['storm_id'].values
test_storm = test_df['storm_id'].values

train_storm_ids = np.unique(np.array(train_storm))
print(f'{len(train_storm_ids)} storms with total {len(train_df)} images')
test_storm_ids = np.unique(np.array(test_storm))
print(f'{len(test_storm_ids)} storms with total {len(test_df)} images')

# split train into train & val
train_percent = 0.8
len_train = int(train_percent * len(train_storm_ids))
shuffle(train_storm_ids)
val_storm_ids = train_storm_ids[len_train:]
train_storm_ids = train_storm_ids[:len_train]

val_df = train_df[train_df['storm_id'].isin(val_storm_ids)]
val_df['split'] = 'val'
print(f'len of val: {len(val_df)}')

train_df = train_df[train_df['storm_id'].isin(train_storm_ids)]
train_df['split'] = 'train'
print(f'len of train: {len(train_df)}')

test_df['split'] = 'test'

# combine train, val, test into one csv
final_df = pd.concat([train_df, test_df, val_df])

final_out_file = f'{data_dir}/wind.csv'
final_df.to_csv(final_out_file, index=False)
print(f'=== save to {final_out_file}')







