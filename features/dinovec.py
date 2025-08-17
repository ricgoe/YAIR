import numpy as np
from database import ImgConvertFailure
import torch
from pathlib import Path
from torchvision.transforms import transforms


class DINOVecCalculator:
    
    def __init__(self):
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Inference running on {self.dev}')
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD  = [0.229, 0.224, 0.225]
        self.model.to(self.dev).eval()
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        
    def gen_dino_vec(self, img: np.ndarray):
        cvt = self.tfm(img)
        with torch.no_grad():
            vec = self.model(cvt.unsqueeze(0).to(self.dev))
        vec = vec.squeeze(0).cpu().numpy().reshape(1, -1)
        return vec
    
if __name__ == "__main__":
    from PIL import Image
    import cv2 as cv
    cvc = DINOVecCalculator()
    patg = '/Volumes/Big Data/data/image_data/DAISY24/HSD_duesseldorf_IMG_9907.jpg'
    img = Image.open(patg)
    arr = np.array(img)
    arr_g = cv.cvtColor(arr, cv.COLOR_RGB2GRAY)
    a=cvc.gen_dino_vec(arr)
    print(a)