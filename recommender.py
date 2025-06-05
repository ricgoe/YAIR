import numpy as np
from pathlib import Path
import cv2 as cv
import torch
from autoencoder.autoenc import AutoEncoder
from torchvision.models.feature_extraction import create_feature_extractor
import os

class ImageRecomender:
    def __init__(self, ae_path: Path, struct_ds_path: Path = None, col_ds_path: Path = None, struct_ds_size: tuple[int, int] = (32,32)):
        self._struct_ds_path: Path = struct_ds_path
        self._struct_ds_size: tuple = struct_ds_size
        self._col_ds_path: Path = col_ds_path
        self._ae_path: Path = ae_path
        self._img_to_compare_path: Path = None
        self._img_to_compare_grey_struct: np.ndarray = None
        self._img_to_compare_color_vec = None
        self._img_to_compare_struct_vec = None
        self._img_similar = None
        if os.name == 'nt' and torch.cuda.is_available():
            self._device = 'cuda'
        elif torch.mps.is_available():
            self._device = 'mps'
        else: 
            self._device = 'cpu'
        self._model = torch.load(str(self._ae_path), weights_only=False, map_location=torch.device(self._device))
        self._return_nodes: dict = {"encoder.4": "encoding_layer", "decoder.5": "decoding_layer"}
        self._feature_extract = create_feature_extractor(self._model, return_nodes=self._return_nodes)
    
    @property
    def img_to_compare(self):
        return self._img_to_compare_path
    
    @img_to_compare.setter
    def img_to_compare(self, img: Path):
        self._img_to_compare_path = img
    
       
    def read_img_to_compare(self):
        try:
            img = cv.imread(str(self.img_to_compare),cv.IMREAD_GRAYSCALE)
            if img is None:
                return
            img = np.float32(img)/255
            img_resized = cv.resize(img, self._struct_ds_size, interpolation=cv.INTER_AREA)
            img_resized = np.clip(img_resized, 0, 1)
            self._img_to_compare_grey_struct = torch.from_numpy(np.reshape(img_resized, (1, self._struct_ds_size[0]*self._struct_ds_size[1])))
        except Exception as e:
            print(f"Error processing {self.img_to_compare}: {e}")
    
    def gen_embedding(self):
        with torch.no_grad():  # No gradients needed for evaluation
            self._img_to_compare_grey_struct = self._img_to_compare_grey_struct.to(self._device)
            outputs = self._feature_extract(self._img_to_compare_grey_struct)
        self._img_to_compare_struct_vec = outputs["encoding_layer"].cpu()#.view(self._struct_ds_size[0], self._struct_ds_size[1]).cpu()
        
    def gen_hist(self):
        pass
    
    def calc_color_sim(self):
        pass
    
    def calc_struct_sim(self):
        pass
    
    def get_closest(self, n: int=1):
        pass
    
            
    
if __name__ == '__main__':
    autoencoder_path = Path('autoencoder/res/autoencoder_full_1_best.pth')
    ir = ImageRecomender(autoencoder_path)
    ir.img_to_compare = Path('images/IMG_3627.jpeg')
    ir.read_img_to_compare()
    ir.gen_embedding()