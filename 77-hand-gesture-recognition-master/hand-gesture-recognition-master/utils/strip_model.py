# remove optimizer data
import os
import sys
import torch

input_file = sys.argv[1]
output_file = sys.argv[2]

state_dict = torch.load(input_file)
del state_dict['optimizer']

torch.save(state_dict, output_file)
