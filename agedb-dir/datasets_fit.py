import os
import logging
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_lds_kernel_window

print = logging.info


class AgeDB(data.Dataset):
    def __init__(self, df, data_dir, img_size, split='train', reweight=0,
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.reweight = reweight

        assert reweight != 'none' if lds else True, "Set reweight > 1 when using LDS"

        self.weights = []

        # first branch is no-weight
        if self.reweight is not None:
            rs = np.linspace(0, 1, num=int(self.reweight))
            for r in rs:
                self.weights.append(self._prepare_weights(r=r, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks,
                                          lds_sigma=lds_sigma))
        else:
            self.weights = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['age']]).astype('float32')

        if self.weights is None:
            weight = np.asarray([np.float32(1.)])
        else:
            weight = []
            for w in self.weights:
                weight.append(
                    np.asarray([w[index]]).astype('float32') if w is not None else np.asarray([np.float32(1.)]))

        return img, label, weight

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

    def _prepare_weights(self, r=0, max_target=121, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['age'].values

        # Gaussian fit
        # mu, std = norm.fit(labels)
        # print(f'get fit {mu}, {std}, power = -[r = {r}]')

        # dist = norm(loc=mu, scale=std)
        # prob_label = [dist.pdf(label) for label in labels]

        # KDE fit
        h = 2
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
