import cv2 as cv
import numpy as np

class ColorVecCalculator:
    
    def __init__(self):
        self.h_bins = 18
        self.l_bins = 4
        self.s_bins = 4
        self.epsilon = 10e-10
    
    def filter_l_s(self, arr: np.ndarray):
        mask = (arr[...,1] > 20) & (arr[...,2] > 50)
        return arr[mask]
        
    
    def gen_color_vec(self, img_path: str):
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
        hls = self.quantize_channels(img)
        hist = [np.unique(hls[:, i], return_counts=True)[-1] for i in range(hls.shape[-1])]
        
        vec = np.concatenate(hist).reshape(1, -1).astype(np.float32)
        return vec
        
    def quantize_channels(self, hls_arr: np.ndarray):
        hls = self.filter_l_s(hls_arr)
        hls = hls/np.array([180, 256, 256])
        hls[...,0] = np.floor(hls[..., 0] * self.h_bins)
        hls[...,2] = np.floor(hls[..., 2] * self.s_bins)
        hls[...,1] = np.floor(hls[..., 1] * self.l_bins)
        
        hls[...,0][hls[..., 0] == self.h_bins] = self.h_bins-1  #edge case hue=180
        hls[...,1][hls[..., 1] == self.l_bins] = self.l_bins-1  #edge case l=256
        hls[...,2][hls[..., 2] == self.s_bins] = self.s_bins-1  #edge case s=256
        return hls
    
if __name__ == "__main__":
    cvc = ColorVecCalculator()
    cvc.gen_color_vec('images/HSD_duesseldorf_IMG_9904.jpg')