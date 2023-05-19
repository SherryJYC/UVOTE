import torch
import torch.nn.functional as F
import math

def weighted_l1nll_loss(mu, log_var, targets, weights=None):
    '''
    based on https://github.com/EPFL-VILAB/XDEnsembles/blob/master/task_configs.py#L126
    '''

    log_sigma = torch.clamp(log_var, min=-5, max=5)
    sigma = log_sigma.exp() + 1e-10

    lap_dist = torch.distributions.Laplace(loc=mu, scale=sigma)
    logprobs = -lap_dist.log_prob(targets)

    loss = logprobs #-torch.log(2 * sigma) - torch.abs(targets - mu) / sigma

    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)

    return loss

def weighted_mse_loss(inputs, targets, weights=None):
    loss = F.mse_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=0.5):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=20., gamma=1):
    loss = F.mse_loss(inputs, targets, reduce=False)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=20., gamma=1):
    loss = F.l1_loss(inputs, targets, reduce=False)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
