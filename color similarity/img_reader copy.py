import cv2 as cv
import numpy as np

class ImgColSimGet:
    def __init__(self, root_path: str, h_bins= 18, l_bins = 4, s_bins = 4, resize_fact: int = 1, mode: str = ('img_file', None), epsilon =10e-10 ,**kwargs):
        self.root_path: str = root_path
        self.resize_fact: int = resize_fact
        self.mode, self.image_key = mode
        self.current_images: list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        self.current_hist: list[np.ndarray, np.ndarray]
        self.test_matrix_builder = kwargs.get('test', False)
        self.h_bins = h_bins
        self.l_bins = l_bins
        self.s_bins = s_bins
        self.epsilon = epsilon
        

    def img_read(self, img_data: str | np.ndarray):
        if self.mode == 'img_file':
            bgr: np.ndarray = cv.imread(img_data)
            bgr = cv.resize(bgr, (bgr.shape[0]//self.resize_fact, bgr.shape[1]//self.resize_fact))
            hls = cv.cvtColor(bgr, cv.COLOR_BGR2HLS) 
            rgb = bgr[...,::-1]
            
        elif self.mode == 'rgb_array':
            rgb: np.ndarray = img_data
            rgb = cv.resize(rgb, (rgb.shape[0]//self.resize_fact, rgb.shape[1]//self.resize_fact))
            hls = cv.cvtColor(rgb, cv.COLOR_RGB2HLS)
            
        return rgb, hls        


    def filter_l_s(self, arr: np.ndarray):
        mask = (arr[:,:,1] > 20) & (arr[:,:,2] > 50)
        return arr[mask]


    def shift_arr(self, hls1_f: np.ndarray, hls2_f: np.ndarray) -> tuple[np.ndarray]:
        h1_quant, l1_quant, s1_quant = self.quantize_channels(hls1_f)
        h2_quant, l2_quant, s2_quant = self.quantize_channels(hls2_f)
        his1_h = cv.calcHist([h1_quant], [0], None, [self.h_bins], [0, self.h_bins])
        his1_h /= his1_h.sum() + self.epsilon
        his1_l = cv.calcHist([l1_quant], [0], None, [self.l_bins], [0, self.l_bins])
        his1_l /= his1_l.sum() + self.epsilon
        his1_s = cv.calcHist([s1_quant], [0], None, [self.s_bins], [0, self.s_bins])
        his1_s /= his1_s.sum() + self.epsilon
        his2_h = cv.calcHist([h2_quant], [0], None, [self.h_bins], [0, self.h_bins])
        his2_h /= his2_h.sum() + self.epsilon
        his2_l = cv.calcHist([l2_quant], [0], None, [self.l_bins], [0, self.l_bins])
        his2_l /= his2_l.sum() + self.epsilon
        his2_s = cv.calcHist([s2_quant], [0], None, [self.s_bins], [0, self.s_bins])
        his2_s /= his2_s.sum() + self.epsilon
        self.current_hist = [his1_h, his2_h]
        angle_1, angle_2 = np.argmax(his1_h), np.argmax(his2_h)
        his1_shifted_ang1 = np.roll(his1_h, (self.h_bins/2-angle_1))
        his2_shifted_ang1 =  np.roll(his2_h, (self.h_bins/2-angle_1))

        
        return his1_shifted_ang1, his2_shifted_ang1, his1_l, his2_l, his1_s, his2_s
     
        
    def quantize_channels(self, hls_arr: np.ndarray):
        h, l, s = self.filter_l_s(hls_arr)[:,0], hls_arr[:,:,1], hls_arr[:,:,2]
        
        return np.clip(h//(180//self.h_bins),a_min=0, a_max=self.h_bins-1), np.clip(l //(256//self.l_bins),a_min=0, a_max=self.l_bins-1), np.clip(s//(256//self.s_bins),a_min=0, a_max=self.s_bins)
        
    
    def calculate_similarity(self, h1: np.ndarray, h2: np.ndarray, l1: np.ndarray, l2: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[float, float]:

        sim_h = cv.compareHist(h1, h2, cv.HISTCMP_BHATTACHARYYA)
        sim_l = cv.compareHist(l1, l2, cv.HISTCMP_BHATTACHARYYA)
        sim_s = cv.compareHist(s1, s2, cv.HISTCMP_BHATTACHARYYA)
        
        return sim_h, sim_l, sim_s

        

if __name__ == "__main__":
    imganalyse = ImgColSimGet('cifar', mode = ('rgb_array', b'data'))
