from radiant_mlhub import Dataset

ds = Dataset.fetch('nasa_tropical_storm_competition')
print(f'dataset size: {ds.estimated_dataset_size/10**9} GB') 

print(f'>>>> Downloading...')
ds.download(output_dir='./data')