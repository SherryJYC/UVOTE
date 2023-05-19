from PIL import Image
import os
import numpy as np
import logging
from scipy.ndimage import convolve1d
from sklearn.neighbors import KernelDensity

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import get_lds_kernel_window

print = logging.info

class Wind(Dataset):

    def __init__(self, df, data_dir, img_size, split='train', reweight=0,
                h=2, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.reweight = reweight

        assert reweight > 0 if  lds else True, "Set reweight > 1 when using LDS"

        self.weights = []

        # first branch is no-weight
        if self.reweight is not None:
            if self.reweight >= 1:
                rs = np.linspace(0, 1, num=int(self.reweight))
            else:
                rs = [self.reweight]
            for r in rs:
                self.weights.append(self._prepare_weights(r=r, h=h))
        else:
            self.weights = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = os.path.join(self.data_dir, row['image_path'])
        img = Image.open(image_path).convert('RGB')

        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['wind_speed']]).astype('float32')

        if self.weights is None:
            weight = np.asarray([np.float32(1.)])
        else:
            weight = []
            for w in self.weights:
                weight.append(
                    np.asarray([w[index]]).astype('float32') if w is not None else np.asarray([np.float32(1.)]))

        return img, label, weight

    def get_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ]
        )
        return transform

    def _prepare_weights(self, r=0, max_target=185, h=2):
        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['wind_speed'].values
        labels = labels.astype(int)

        # KDE fit
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(labels.reshape(-1, 1))
        score = kde.score_samples(labels.reshape(-1, 1)) # Compute the log-likelihood
        prob_label = np.exp(score)
        print(f'use KDE, h={h}, power = -[r = {r}]')

        if not len(prob_label):
            return None

        weights = np.power(np.array(prob_label), -r)
        scaling = len(weights) / np.sum(weights)
        weights *= scaling

        return weights
        
