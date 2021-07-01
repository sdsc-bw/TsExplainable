import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
#from lime.wrappers.scikit_image import BaseWrapper
import numpy as np
import math

min = 1.0E-31
max = 1 - min

def log_odds(p):
    if p <= min:
        return math.log(min/max)
    elif p >= max:
        return math.log(max/min)
    else:
        return math.log(p/(1-p))


def change_in_log_odds(black_box, org_img, mask_img, start_idx, target_idx, keras=False):
    if keras:
        org_pred = black_box.predict(np.expand_dims(np.expand_dims(org_img, axis=2), axis=0))
        mask_pred = black_box.predict(np.expand_dims(np.expand_dims(mask_img, axis=2), axis=0))
    else:
        with torch.no_grad():
            org_pred = F.softmax(black_box(torch.tensor(org_img).unsqueeze(0).unsqueeze(0).float()), dim=1)
            mask_pred = F.softmax(black_box(torch.tensor(mask_img).unsqueeze(0).unsqueeze(0).float()), dim=1)
    change_start = log_odds(org_pred[0][start_idx].item()) - log_odds(mask_pred[0][start_idx].item())
    change_target = log_odds(mask_pred[0][target_idx].item()) - log_odds(org_pred[0][target_idx].item())
    return change_start + change_target

def change_in_log_odds2(black_box, org_img, mask_img, idx, swap=False, keras=False):
    if keras:
        org_pred = black_box.predict(np.expand_dims(np.expand_dims(org_img, axis=2), axis=0))
        mask_pred = black_box.predict(np.expand_dims(np.expand_dims(mask_img, axis=2), axis=0))
    else:
        with torch.no_grad():
            org_pred = F.softmax(black_box(torch.tensor(org_img).unsqueeze(0).unsqueeze(0).float()), dim=1)
            mask_pred = F.softmax(black_box(torch.tensor(mask_img).unsqueeze(0).unsqueeze(0).float()), dim=1)
    if swap:
        change = log_odds(mask_pred[0][idx].item()) - log_odds(org_pred[0][idx].item())
    else:
        change = log_odds(org_pred[0][idx].item()) - log_odds(mask_pred[0][idx].item())
    return change

def change_in_log_odds3(black_box, org_img, mask_img, idx, neg_change=False, keras=False):
    if keras:
        org_pred = black_box.predict(np.expand_dims(np.expand_dims(org_img, axis=2), axis=0))
        mask_pred = black_box.predict(np.expand_dims(np.expand_dims(mask_img, axis=2), axis=0))
    else:
        with torch.no_grad():
            org_pred = F.softmax(black_box(torch.tensor(org_img).unsqueeze(0).unsqueeze(0).float()), dim=1)
            mask_pred = F.softmax(black_box(torch.tensor(mask_img).unsqueeze(0).unsqueeze(0).float()), dim=1)
    if neg_change:
        change = log_odds(org_pred[0][idx].item()) - log_odds(mask_pred[0][idx].item())
    else:
        change = log_odds(mask_pred[0][idx].item()) - log_odds(org_pred[0][idx].item())
    return change

def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices

def load_mnist(idx, train=False):
    np.random.seed(0)
    torch.manual_seed(0)
    transform=transforms.Compose([
        transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=train, download=True, transform=transform)
    idx = get_indices(dataset, idx)
    loader = DataLoader(dataset,batch_size=1, sampler=SubsetRandomSampler(idx), shuffle=False)
    return loader
"""
class BlockSegmentationAlgorithm(BaseWrapper):
    def __init__(self, algo_type, **target_params):
        self.algo_type = algo_type
        if self.algo_type == 'block':
            BaseWrapper.__init__(self, block, **target_params)
            kwargs = self.filter_params(block)
            self.set_params(**kwargs)
            
    def __call__(self, *args):
        return self.target_fn(args[0], **self.target_params)
"""    
def kl_divergence(p, q):
    """ Epsilon is used here to avoid conditional code for checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    p = p+epsilon
    q = q+epsilon

    divergence = np.sum(p*np.log(p/q))
    return divergence
    
def block(image, kernel_size):
    assert type(image) is np.ndarray
    if len(image.shape) == 2:
        img = image
    elif image.shape[2] == 3:
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img = np.dot(image[...,:3], rgb_weights)
    elif image.shape[2] == 1:
        img = np.squeeze(image, axis=2)
    width, height, = img.shape
    segment_mask = np.zeros(img.shape, dtype=np.int64)
    n = 1
    for i in np.arange(0, height - kernel_size + 1, kernel_size):
        for j in range(0, width - kernel_size + 1, kernel_size):
            segment_mask[i:i+kernel_size, j:j+kernel_size] = n
            n += 1
    return segment_mask

def first_true(x, axis=0):
    ft = (x >= 1)
    _, idx = ((ft.cumsum(axis) == 1) & ft).max(axis)
    return idx.item()