import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    # ia.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_model_per_subject(output_root, model_dir_name, epoch, model, optimizer, subject):
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    save_path = os.path.join(output_root, subject, model_dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(
        save_path, 'model_' + str(epoch).zfill(3) + '.pth')
    if os.path.exists(save_file):
        os.remove(save_file)
    torch.save(state, save_file)


def save_model(output_root, model_dir_name, epoch, model, optimizer):
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    save_path = os.path.join(output_root, model_dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(
        save_path, 'model_' + str(epoch).zfill(3) + '.pth')
    if os.path.exists(save_file):
        os.remove(save_file)
    torch.save(state, save_file)


def load_pretrain_model():
    pretrained_weight_file_path = ("/home/whcold/workspace/micro-expression/"
                                   "bsn_spotting_regression/3_class.pth")
    checkpoint = torch.load(pretrained_weight_file_path,
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model'])


# refer to https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
def configure_optimizers(model, learning_rate, weight_decay):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
    blacklist_weight_modules = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(inter_params) == 0, (
        f"parameters {str(inter_params)} made it into both decay/no_decay sets!")
    assert len(param_dict.keys() - union_params) == 0, (
        f"parameters {str(param_dict.keys() - union_params)} "
        "were not separated into either decay/no_decay set!")

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer
