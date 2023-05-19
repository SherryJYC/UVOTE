from typing import Dict
import numpy as np
import torch

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import pearsonr, spearmanr, gmean


def get_text_field_mask(text_field_tensors: Dict[str, torch.Tensor]) -> torch.LongTensor:
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask of shape
    ``(batch_size, num_tokens)``.  This mask will be 0 where the tokens are padding, and 1
    otherwise.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we assume that the tensor in
    the dictionary with the lowest number of dimensions has plain token ids.  This allows us to
    also handle cases where the input is actually a ``ListField[TextField]``.

    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.byteTensors inside Variables makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    >>> mask = torch.ones([260]).byte()
    >>> mask.sum() # equals 260.
    >>> var_mask = torch.autograd.Variable(mask)
    >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])
    token_tensor = tensor_dims[0][1]

    return (token_tensor != 0).long()

def device_mapping(cuda_device: int):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """
    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage
    return inner_device_mapping

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

def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.5, clip_max=2.):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 <= 0.).any() or (v2 < 0.).any():
        valid_pos = (((v1 > 0.) + (v2 >= 0.)) == 2)
        factor = torch.clamp(v2[valid_pos] / v1[valid_pos], clip_min, clip_max)
        matrix[:, valid_pos] = (matrix[:, valid_pos] - m1[valid_pos]) * torch.sqrt(factor) + m2[valid_pos]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2

def resume_checkpoint(model, model_state, backbone_only=False, morebranch=False):
    model.pair_encoder.load_state_dict(
        {k.split('.', 1)[1]: v for k, v in model_state.items() if 'pair_encoder' in k}
    )
    if not backbone_only and not morebranch:
        getattr(model, 'sts-b_pred_layer').load_state_dict(
            {k.split('.', 1)[1]: v for k, v in model_state.items() if 'sts-b_pred_layer' in k}
        )
    elif not backbone_only and morebranch:
        getattr(model, 'sts-b_mu_layer').load_state_dict(
            {k.split('.', 1)[1]: v for k, v in model_state.items() if 'sts-b_mu_layer' in k}
        )
        getattr(model, 'sts-b_logvar_layer').load_state_dict(
            {k.split('.', 1)[1]: v for k, v in model_state.items() if 'sts-b_logvar_layer' in k}
        )
    return model

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        # kernel = gaussian(ks)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

class STSShotAverage:
    def __init__(self, metric, uce=False):
        self._pred = []
        self._label = []
        self._std = []
        self._count = 0
        self._metric = metric
        self._num_bins = 50
        self._uce = uce

        # under np.float32 division
        self._shot_idx = {
            'many': [0, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 49],
            'medium': [2, 4, 6, 8, 27, 35, 37],
            'few': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 29, 31, 33, 39, 41, 43, 45, 47]
        }

        def get_bin_idx(label):
            _, bins_edges = np.histogram(a=np.array([], dtype=np.float32), bins=self._num_bins, range=(0., 5.))
            if label == 5.:
                return self._num_bins - 1
            else:
                return np.where(bins_edges > label)[0][0] - 1
        self._get_bin_idx = get_bin_idx

    def __call__(self, pred, label, std=None):
        self._pred += pred.tolist()
        self._label += label.tolist()
        if std is not None:
            self._std += std.tolist()
        self._count += len(pred)

    def get_metric(self, reset=False, type=None):
        label_bin_idx = list(map(self._get_bin_idx, self._label))
        def bin2shot(idx):
            if idx in self._shot_idx['many']:
                return 'many'
            elif idx in self._shot_idx['medium']:
                return 'medium'
            else:
                return 'few'
        label_category = np.array(list(map(bin2shot, label_bin_idx)))

        pred_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
        label_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
        std_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
        metric = {'many': {}, 'medium': {}, 'few': {}, 'overall': {}}
        for shot in ['overall', 'many', 'medium', 'few']:
            pred_shot[shot] = np.array(self._pred)[label_category == shot] * 5. if shot != 'overall' else np.array(self._pred) * 5.
            label_shot[shot] = np.array(self._label)[label_category == shot] if shot != 'overall' else np.array(self._label)
            if self._uce:
                std_shot[shot] = np.array(self._std)[label_category == shot]  if shot != 'overall' else np.array(self._std) 
            
            if 'mse' in self._metric:
                metric[shot]['mse'] = np.mean((pred_shot[shot] - label_shot[shot]) ** 2) if pred_shot[shot].size > 0 else 0.
            if 'l1' in self._metric:
                metric[shot]['l1'] = np.mean(np.abs(pred_shot[shot] - label_shot[shot])) if pred_shot[shot].size > 0 else 0.
            if 'gmean' in self._metric:
                if pred_shot[shot].size <= 0:
                    metric[shot]['gmean'] = 0.
                else:
                    diff = np.abs(pred_shot[shot] - label_shot[shot])
                    if diff[diff == 0.].size:
                        diff[diff == 0.] += 1e-10
                        metric[shot]['gmean'] = gmean(diff) if pred_shot[shot].size > 0 else 0.
                    else:
                        metric[shot]['gmean'] = gmean(np.abs(pred_shot[shot] - label_shot[shot])) if pred_shot[shot].size > 0 else 0.
            if 'pearsonr' in self._metric:
                metric[shot]['pearsonr'] = pearsonr(pred_shot[shot], label_shot[shot])[0] if pred_shot[shot].size > 1 else 0.
            if 'spearmanr' in self._metric:
                metric[shot]['spearmanr'] = spearmanr(pred_shot[shot], label_shot[shot])[0] if pred_shot[shot].size > 1 else 0.
            metric[shot]['num_samples'] = pred_shot[shot].size

            # UCE
            if self._uce:
                metric[shot]['uce'] = measureUCE(pred=pred_shot[shot], std=std_shot[shot], labels=label_shot[shot])
            else:
                metric[shot]['uce'] = -1

        if reset:
            self.reset()
        return metric['overall'] if type == 'overall' else metric


    def reset(self):
        self._pred = []
        self._label = []
        self._std = []
        self._count = 0

def measureUCE(pred, std, labels, num_bin=10, sample_threshold=1):
    # check std range
    large_idx = std > np.exp(10)
    if sum(large_idx) > 0:
        print(f'!!!!! WARNING: # {sum(large_idx)} stds > e^10 !!!!!')
        std[large_idx] = np.exp(10)
    if np.isnan(std).any():
        print(f'!!!!! WARNING: stds contains NaN. No UCE can be measured')
        return -1
    
    if len(pred.shape) > 1:
        pred, std, labels = pred.squeeze(), std.squeeze(), labels.squeeze()

    hist, bin_edges = np.histogram(std, bins=num_bin)
    
    if len(std) < 1:
        print(f'no valid std to compute UCE')
        return -1

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
