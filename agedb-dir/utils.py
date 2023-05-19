import os
import shutil
import torch
import logging
import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import norm

def measureUCE(pred, std, labels, num_bin=10, sample_threshold=1):
    # check std range
    large_idx = std > np.exp(10)
    if sum(large_idx) > 0:
        print(f'!!!!! WARNING: # {sum(large_idx)} stds > e^10 !!!!!')
        std[large_idx] = np.exp(10)
    if np.isnan(std).any():
        print(f'!!!!! WARNING: stds contains NaN. No UCE can be measured')
        return -1

    pred, std, labels = pred.squeeze(), std.squeeze(), labels.squeeze()

    hist, bin_edges = np.histogram(std, bins=num_bin)

    uce = 0

    for i, (h, b) in enumerate(zip(hist, bin_edges)):
        if h < sample_threshold:
            continue

        std_min = b
        if i == len(hist) - 1:
            std_max = np.inf
        else:
            std_max = bin_edges[i+1]

        idx = (std >= std_min) & (std < std_max)
        std_selected = std[idx]
        pred_selected = pred[idx]
        y_selected = labels[idx]

        mae = np.mean(np.abs(pred_selected - y_selected))
        mstd = np.mean(std_selected)
        one_uce = np.abs(mae - mstd)*sum(idx)
        uce += one_uce

    uce /= len(std)
    return uce

def set_seed(seed: int = 42) -> None:
    print('set random seed ', seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args):
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.

    lr_max = 0
    for param_group in optimizer.param_groups:
        if param_group['lr'] > lr_max:
            lr_max = param_group['lr']

    for param_group in optimizer.param_groups:
        if param_group['lr'] < lr_max:
            param_group['lr'] = lr *0.1
        else:
            param_group['lr'] = lr


def save_checkpoint(args, state, is_best, prefix=''):
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window
