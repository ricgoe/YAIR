import cv2 as cv
import numpy as np

class ColorVecCalculator:
    
    def __init__(self):
        self.h_bins = 18
        self.l_bins = 4
        self.s_bins = 4
        self.epsilon = 10e-10
    
    def filter_l_s(self, arr: np.ndarray):
        mask = (arr[:,:,1] > 20) & (arr[:,:,2] > 50)
        return arr[mask]
        
    
    def gen_color_vec(self, img_path: str):
        img = cv.imread(img_path)
        h1_quant, l1_quant, s1_quant = self.quantize_channels(img)
        his1_h = cv.calcHist([h1_quant], [0], None, [self.h_bins], [0, self.h_bins])
        his1_h /= his1_h.sum() + self.epsilon
        his1_l = cv.calcHist([l1_quant], [0], None, [self.l_bins], [0, self.l_bins])
        his1_l /= his1_l.sum() + self.epsilon
        his1_s = cv.calcHist([s1_quant], [0], None, [self.s_bins], [0, self.s_bins])
        his1_s /= his1_s.sum() + self.epsilon
        
        vec = np.concatenate((his1_h, his1_l, his1_s)).reshape(1, -1)
        return vec
        
    def quantize_channels(self, hls_arr: np.ndarray):
        h, l, s = self.filter_l_s(hls_arr)[:,0], hls_arr[:,:,1], hls_arr[:,:,2]
        return np.clip(h//(180//self.h_bins),a_min=0, a_max=self.h_bins-1), np.clip(l //(256//self.l_bins),a_min=0, a_max=self.l_bins-1), np.clip(s//(256//self.s_bins),a_min=0, a_max=self.s_bins)
