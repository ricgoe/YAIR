import cv2 as cv
import numpy as np

class ColorVecCalculator:
    
    def __init__(self, length):
        h = int(np.round(0.7*length))
        l = int(np.round((length-h)/2))
        s = length-h-l
        if l != s:
            print("WARNING: unequal length of l and s for color vectors")
        self.hls_bins = [h, l, s]
        self.hls_max = [180, 256, 256]
    
    def filter_l_s(self, arr: np.ndarray):
        mask = (arr[...,1] > 20) & (arr[...,2] > 50)
        return arr[mask]
        
    
    def gen_color_vec(self, img: np.ndarray):
        img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
        hls = self.quantize_channels(img)
        hist =  []
        for i in range(hls.shape[-1]):
            hist.append(np.bincount(hls[:, i], minlength=self.hls_bins[i]))
        vec = np.concatenate(hist).reshape(1, -1).astype(np.float32)
        return vec
        
    def quantize_channels(self, hls_arr: np.ndarray):
        hls = self.filter_l_s(hls_arr)
        hls = hls/np.array([180, 256, 256])
        for i in range(hls.shape[-1]):
            hls[...,i] = np.floor(hls[..., i] * self.hls_bins[i])
            hls[...,i][hls[..., i] == self.hls_bins[i]] = self.hls_bins[i]-1  #edge case hue=180
        return hls.astype(np.uint8)
    
if __name__ == "__main__":
    cvc = ColorVecCalculator()
    a=cvc.gen_color_vec("/raid/richard/image_data/DAISY_2025/20250328_101537.jpg")
    print(a[0].shape)