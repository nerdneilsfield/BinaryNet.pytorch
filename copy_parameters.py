import argparse
import os
import re
from typing_extensions import OrderedDict
from copy import deepcopy


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import models
import shutil


parser = argparse.ArgumentParser(description='PyTorch')

parser.add_argument("-i", "--input", default="")
parser.add_argument("-g", "--gn", default="")
parser.add_argument("-r", "--cr", default="")
parser.add_argument("-f", "--fc", default="")


args = parser.parse_args()

save_path = os.path.join("./results", args.input)

checkpoint_file = os.path.join(save_path, "model_best.pth.tar")

vgg = models.vgg_cifar10_binary()

vgg_gbn = models.vgg_gbn()
vgg_crbn = models.vgg_crbn()
vgg_fcbn = models.vgg_fcbn()

checkpoint = torch.load(checkpoint_file)
vgg_dict = checkpoint['state_dict']

vgg_gbn_dict = vgg_gbn.state_dict()
vgg_crbn_dict = vgg_crbn.state_dict()
vgg_fcbn_dict = vgg_fcbn.state_dict()

gbn = {
    ".0.": ".0.",
    ".1.": ".2.",
    ".2.": ".3.",
    ".3.": ".4.",
    ".4.": ".6.",
    ".5.": ".7.",
    ".6.": ".8.",
    ".7.": ".9.",
    ".8.": ".11.",
    ".9.": ".12.",
    ".10.": ".13.",
    ".11.": ".15.",
    ".12.": ".16.",
    ".13.": ".17.",
    ".14.": ".18.",
    ".15.": ".20.",
    ".16.": ".21.",
    ".17.": ".22.",
    ".18.": ".24.",
    ".19.": ".25.",
    # ".20.": ".0.",
    # ".21.": ".0.",
    # ".22.": ".0.",
    # ".22.": ".0.",
    # ".23.": ".0.",
}


fcbn  = {
    ".0.": ".0.",
    ".1.": ".2.",
    ".2.": ".3.",
    ".3.": ".4.",
    ".4.": ".6.",
    ".5.": ".7.",
    ".6.": ".8.",
    ".7.": ".10.",
}


for k, v in vgg_dict.items():

    if 'features' in k:
        id =  re.findall(r"\.\d+\.", k)[0]

        print(id)

        new_k = k.replace(id, gbn[id])

        vgg_gbn_dict[new_k] = v
        vgg_crbn_dict[new_k] = v
        vgg_fcbn_dict[new_k] = v
    else:
        vgg_gbn_dict[k] = v
        vgg_crbn_dict[k] = v

        id =  re.findall(r"\.\d+\.", k)[0]
        print(f"classifier {id}")

        new_k = k.replace(id, fcbn[id])
        print(f"classifier new k: {new_k}")
        vgg_fcbn_dict[new_k] = v

print(vgg_gbn_dict.keys())


print("-------")

print(vgg_crbn_dict.keys())

print("-----")
print(vgg_fcbn_dict.keys())


vgg_gbn_cp = deepcopy(checkpoint)
vgg_cr_cp = deepcopy(checkpoint)
vgg_fc_cp = deepcopy(checkpoint)
print("-------")
print(f"save gn to {args.gn} and save crbn to {args.cr} and fcbn to {args.fc}")
vgg_gbn_cp['state_dict'] = vgg_gbn_dict
vgg_cr_cp['state_dict'] = vgg_crbn_dict
vgg_fc_cp['state_dict'] = vgg_fcbn_dict


torch.save(vgg_gbn_cp, args.gn)
torch.save(vgg_cr_cp, args.cr)
torch.save(vgg_fc_cp, args.fc)

