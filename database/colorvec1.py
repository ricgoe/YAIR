import cv2 as cv
import numpy as np
from ImgModel import ImgConvertFailure

class ColorVecCalculator:
    
    def __init__(self):
        self.hls_bins = [18, 4, 4]
        self.hls_max = [180, 256, 256]
    
    def filter_l_s(self, arr: np.ndarray):
        mask = (arr[...,1] > 20) & (arr[...,2] > 50)
        return arr[mask]
        
    
    def gen_color_vec(self, img_path: str):
        try:
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
            hls = self.quantize_channels(img)
            hist = []
            for i in range(hls.shape[-1]):
                idx, counts = np.unique(hls[:, i], return_counts=True)
                bins = np.zeros(self.hls_bins[i])
                bins[idx.astype(int)] = counts #are ints anyway
                hist.append(bins)
        except Exception as e:
            raise ImgConvertFailure(img_path, e)
        
        vec = np.concatenate(hist).reshape(1, -1).astype(np.float32)
        return vec, img.shape[0], img.shape[1]
        
    def quantize_channels(self, hls_arr: np.ndarray):
        hls = self.filter_l_s(hls_arr)
        hls = hls/np.array([180, 256, 256])
        for i in range(hls.shape[-1]):
            hls[...,i] = np.floor(hls[..., i] * self.hls_bins[i])
            hls[...,i][hls[..., i] == self.hls_bins[i]] = self.hls_bins[i]-1  #edge case hue=180
        return hls
    
if __name__ == "__main__":
    cvc = ColorVecCalculator()
    cvc.gen_color_vec(r'images\HSD_duesseldorf_IMG_9904_crop.jpg')