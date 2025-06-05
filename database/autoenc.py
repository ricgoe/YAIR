import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
#from torchsummary import summary
import numpy as np
import pickle
from pathlib import Path
import json
import sys


class DataSetMaker:
    def __init__(self, ds_path: Path, debug = False):
        self.ds_path: Path = ds_path
        self.debug: bool = debug
        self.conc_ds: list = []
        self.make_ds_list()
        self.conc_ds_pyt = Data.ConcatDataset(self.conc_ds)
        if debug:
            self.train_set, self.test_set = torch.utils.data.random_split(self.conc_ds_pyt, [0.05, 0.95])
        else:
            self.train_set, self.test_set = torch.utils.data.random_split(self.conc_ds_pyt, [0.8, 0.2])
            #print(self.train_set.indices)
        
    def read_pickle(self, file: Path):
        try: 
            with open(file, 'rb') as file:
                arr = pickle.load(file)
        except Exception as e:
            print(f'Could not read {str(file.name)}')
            return np.array([])
        return arr
    
    def make_ds_list(self):
        for file in self.ds_path.iterdir():
            pic = self.read_pickle(file)
            if pic.size>0:
                self.conc_ds.append(pic)
            else:
                continue
    
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(32*32, params[1]),
            activation_funcs[params[4]](),
            nn.Linear(params[1],params[2]),
            activation_funcs[params[4]](),
            nn.Linear(params[2], params[3])
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(params[3], params[2]),
            activation_funcs[params[4]](),
            nn.Linear(params[2], params[1]),
            activation_funcs[params[4]](),
            nn.Linear(params[1], 32*32),
            activation_funcs[params[5]]()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_json(file: Path):
    with open(file, 'r') as f:
        return json.load(f)

   
def run_ae(index: int, logging_interval = 100):
    best_run = np.inf
    global params
    params = all_params[str(index)]
    if index == 300 or index == 299:
        maker = DataSetMaker(DS_PATH, debug=True)
    else:
        maker = DataSetMaker(DS_PATH, debug=False)
    train_loader = Data.DataLoader(maker.train_set,
                                   batch_size=params[0],
                                   shuffle=True)
    test_loader = Data.DataLoader(maker.test_set,
                                   batch_size=params[0],
                                   shuffle=False)

    ae = AutoEncoder()
    ae.to(device)
    #summary(model=ae, input_size=(1,32*32))
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCH):
        for step, x in enumerate(train_loader):
            
            b_x = x.view(-1, 32*32).requires_grad_(True)
            b_x = b_x.to(device)
            
            b_y = x.view(-1, 32*32).requires_grad_(True)
            b_y = b_y.to(device)
            ae.train()
            encoded, decoded = ae(b_x)
            loss = loss_function(decoded, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not step % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'% (epoch+1, EPOCH, step,len(train_loader), loss))
        
        
        with torch.no_grad():
            ae.eval()
            for step, input in enumerate(test_loader):
                input = input.to(device)
                input = input.view(-1, 32*32)
                encoded, decoded = ae(input)
                test_loss = loss_function(decoded, input)
                if test_loss < best_run:
                    best_run = test_loss
                    print(f'Found new best loss: {test_loss}')
                    torch.save(ae, f"autoencoder_full_{str(index)}_best.pth")
        
        
    with torch.no_grad():
        ae.eval()
        for step, input in enumerate(test_loader):
            input = input.to(device)
            input = input.view(-1, 32*32)
            encoded, decoded = ae(input)
            loss = loss_function(decoded, input)
    
    torch.save(ae, f"autoencoder_full_{str(index)}_last.pth")


torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DS_PATH = Path('ds_32_32')
SAVE_PATH = Path('results').mkdir(exist_ok=True)
LR = 0.005
EPOCH = 10
# Batchsize, Input, 1.Layer, 2.Layer, 3.Layer -> Decoder same, activation between layers, activation output
all_params = load_json(Path('training.json'))
activation_funcs = [nn.Sigmoid, nn.ReLU]#0:nn.Sigmoid(),1:nn.ReLU()

if __name__ == '__main__':
    args = sys.argv[1]
    run_ae(args)