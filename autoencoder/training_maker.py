import torch.nn as nn
import json

batch_sizes = [512, 256, 128, 64, 32, 16]
first_layers = [512, 256, 128, 64]
second_layers = [256, 128, 64, 32]
third_layers = [128, 64, 32, 16, 8]
activation_funcs = [0, 1] #0:nn.Sigmoid(),1:nn.ReLU()

ind = 1
training_dict = {}
for batch_s in batch_sizes:
    for first_layer in first_layers:
        for second_layer in second_layers:
            if second_layer >= first_layer:
                continue
            for third_layer in third_layers:
                if third_layer >= second_layer:
                    continue
                for activation_func in activation_funcs:
                    training_dict[ind] = [batch_s, first_layer, second_layer, third_layer, activation_func, 0]
                    ind+= 1
                    
with open('training.json', 'w') as f:
    json.dump(training_dict, f, indent=4)