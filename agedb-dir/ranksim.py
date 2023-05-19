# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import torch.nn.functional as F

from ranking import TrueRanker, rank_normalised

def batchwise_ranking_regularizer_sigma(outputs, targets, lambda_val):
    loss = 0

    # Reduce ties and boost relative representation of infrequent labels by computing the 
    # regularizer over a subset of the batch in which each label appears at most once
    batch_unique_targets = torch.unique(targets)
    aes = []
    stds = []

    if len(batch_unique_targets) < len(targets):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((targets == target).nonzero()[:,0]).item())
        y = targets[sampled_indices]
        for i in range(len(outputs)):
            aes.extend(torch.abs(outputs[i][0][sampled_indices].detach() - y))
            stds.extend(torch.exp(outputs[i][1][sampled_indices]) * torch.sqrt(torch.tensor(2)))
    else:
        y = targets
        for i in range(len(outputs)):
            aes.extend(torch.abs(outputs[i][0].detach() - y))
            stds.extend(torch.exp(outputs[i][1]) * torch.sqrt(torch.tensor(2)))

    # concatenate
    aes = torch.cat(aes, axis=0)
    stds = torch.cat(stds, axis=0)

    loss += F.l1_loss(aes, stds)

    # Compute ranking similarity loss
    # for i in range(len(aes)): # B_sampled x #branch
    #     aes_ranks = rank_normalised(torch.abs(aes[i] - aes).unsqueeze(dim=0))
    #     stds_ranks = TrueRanker.apply(torch.abs(stds[i] - stds).unsqueeze(dim=0), lambda_val)
    #     loss += F.mse_loss(aes_ranks, stds_ranks)
    
    return loss

def batchwise_ranking_regularizer(features, targets, lambda_val):
    loss = 0

    # Reduce ties and boost relative representation of infrequent labels by computing the 
    # regularizer over a subset of the batch in which each label appears at most once
    batch_unique_targets = torch.unique(targets)
    if len(batch_unique_targets) < len(targets):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((targets == target).nonzero()[:,0]).item())
        x = features[sampled_indices]
        y = targets[sampled_indices]
    else:
        x = features
        y = targets

    # Compute feature similarities
    xxt = torch.matmul(F.normalize(x.view(x.size(0),-1)), F.normalize(x.view(x.size(0),-1)).permute(1,0))

    # Compute ranking similarity loss
    for i in range(len(y)):
        label_ranks = rank_normalised(-torch.abs(y[i] - y).transpose(0,1))
        feature_ranks = TrueRanker.apply(xxt[i].unsqueeze(dim=0), lambda_val)
        loss += F.mse_loss(feature_ranks, label_ranks)
    
    return loss
