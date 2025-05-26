import torch
import torchvision
from pathlib import Path

import torchvision.models.feature_extraction
from autoenc import AutoEncoder
from autoenc import DataSetMaker
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor

ds_path = Path('ds_32_32_INTER_AREA')
model_path = Path('autoencoder/res/autoencoder_full_1_best.pth')
torch.manual_seed(1)
maker = DataSetMaker(ds_path)
test_loader = Data.DataLoader(maker.train_set, shuffle=False)

device = 'mps' if torch.mps.is_available() else 'cpu'

model = torch.load(str(model_path), weights_only=False, map_location=torch.device('mps'))
train_nodes, _ = torchvision.models.feature_extraction.get_graph_node_names(model)

return_nodes: dict = {"encoder.4": "encoding_layer", "decoder.5": "decoding_layer"}
feature_extract = create_feature_extractor(model, return_nodes=return_nodes)


print(train_nodes)
print(type(model))
model.eval()
all_outputs = []
all_images = []


with torch.no_grad():  # No gradients needed for evaluation
    for i, inputs in enumerate(test_loader):
        all_images.append(inputs)
        inputs = inputs.to(device)
        inputs = inputs.view(-1, 32*32)
        print(inputs.shape)
        outputs =feature_extract(inputs)
        
        all_outputs.append(outputs['decoding_layer'].view(-1,32,32).cpu())
        
        if i > 100:
            break

for i in range(80,90):
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.set_title('Reconstructed image')
    ax2.set_title('Original image')
    ax1.imshow(all_outputs[i][0], cmap='grey')
    ax2.imshow(all_images[i][0][0], cmap='grey')

    plt.show()