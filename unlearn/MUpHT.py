import sys
import time

import torch
import utils

import torch.nn as nn

from imagenet import get_x_y_from_data_dict

class BNFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input[0]
        self.output = output
    def close(self):
        self.hook.remove()

def MUpHT(
    data_loaders, model, criterion, args
):
    forget_loader = data_loaders["forget"]

    feature_layers = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            feature_layers.append(BNFeatureHook(module))
    start_time = time.time()
    model.eval()

    feature_list = []
    for imgs, label in forget_loader:
        with torch.no_grad():
            o = model(imgs.to(args.device))
            features = [mod.input for (idx, mod) in enumerate(feature_layers)]
            feature_list.append(features[-1])
    feature_all = torch.vstack(feature_list)

    unlearning_feature = feature_all.detach().squeeze()
    if len(unlearning_feature.shape) == 1:
        unlearning_feature = unlearning_feature[None, ...]
    u, s, v = torch.svd(unlearning_feature.T, some=False)
    if "resnet" in args.arch:
        fc_weight = model.fc.weight
    elif "vgg" in args.arch:
        fc_weight = model.classifier[4].weight
    elif "swin" in args.arch:
        fc_weight = model.mlp_head[1].weight

    u = u.to(args.device)
    u_unlearning = u[:, :args.mupht_num_u]
    u_conter = fc_weight @ u_unlearning
    u_conter = u_conter @ u_unlearning.T
    fc_weight_unlearning = fc_weight - args.mupht_weight*u_conter

    if "resnet" in args.arch:
        model.fc.weight = nn.parameter.Parameter(fc_weight_unlearning)
    elif "vgg" in args.arch:
        model.classifier[4].weight  = nn.parameter.Parameter(fc_weight_unlearning)
    elif "swin" in args.arch:
        model.mlp_head[1].weight  = nn.parameter.Parameter(fc_weight_unlearning)
        
    RTE = time.time() - start_time
    print('RTE:', RTE)


