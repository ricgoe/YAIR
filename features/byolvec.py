from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from pathlib import Path
from threading import Lock
from training import Resnet50SmallDim


class BYOLVecCalculator:
    
    def __init__(self, model_path: Path):
        # self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'Inference running on {self.dev}')
        self.model = Resnet50SmallDim(output_dim=256)
        #self.model = models.resnet50(weights = None)
        #self.model.fc = torch.nn.Identity()
        self.state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(self.state, strict = False)
        self.model.to(self.dev).eval()
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        self.lock = Lock()

        
    def gen_byol_vec(self, img: np.ndarray):
        cvt = self.tfm(img).expand(3, -1, -1)
        with torch.no_grad():
            vec = self.model(cvt.unsqueeze(0).to(self.dev))
        vec = vec.squeeze(0).cpu().numpy().reshape(1, -1)
        return vec
    
if __name__ == "__main__":
    cvc = BYOLVecCalculator("/raid/richard/checkpoints/byol_backbone_r50_e100.pth")
    a=cvc.gen_byol_vec("/raid/richard/image_data/DAISY_2025/20250328_101537.jpg")
    print(a[0].shape)