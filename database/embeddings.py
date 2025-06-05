import numpy as np
from pathlib import Path
import cv2 as cv
from PIL import UnidentifiedImageError
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import os
from autoenc import AutoEncoder


class Embedder:
    def __init__(self, model_path: str):
        self._device = "mps"
        self._model = torch.load(model_path, weights_only=False, map_location=torch.device(self._device))
        self._return_nodes: dict = {"encoder.4": "encoding_layer", "decoder.5": "decoding_layer"}
        self._feature_extract = create_feature_extractor(self._model, return_nodes=self._return_nodes)
       
    def process(self, img_path) -> torch.Tensor:
        img = cv.imread(str(img),cv.IMREAD_GRAYSCALE)
        if img is None:
            raise UnidentifiedImageError
        img = np.float32(img)/255
        img_resized = cv.resize(img, (32,32), interpolation=cv.INTER_AREA)
        img_resized = np.clip(img_resized, 0, 1)
        return torch.from_numpy(np.reshape(img_resized, (1, 32*32)))
    
    def gen_embedding(self, img_path: str) -> torch.Tensor:
        with torch.no_grad():  # No gradients needed for evaluation
            processed_img = self.process(img_path).to(self._device)
            outputs: dict[str, torch.Tensor] = self._feature_extract(processed_img)
        return outputs["encoding_layer"].cpu()#.view(self._struct_ds_size[0], self._struct_ds_size[1]).cpu()
        
            