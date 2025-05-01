import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from colorthief import ColorThief
import colorsys
import pickle
import glob

class ImgColSimGet:
    def __init__(self, root_path: str, h_bins= 18, l_bins = 4, s_bins = 4, resize_fact: int = 1, mode: str = ('img_file', None), epsilon =10e-10 ,**kwargs):
        self.root_path: str = root_path
        self.search_path: str = root_path + "/**/*.*"
        #self.n_bins: int = n_bins
        self.resize_fact: int = resize_fact
        self.mode, self.image_key = mode
        self.verbose: str = kwargs.get('verbose', False)
        self.should_show_hist: bool = kwargs.get('show_hist', False)
        self.simmatrix: np.ndarray = np.ndarray(0)
        self.num_images: int = None
        self.gen_filename = self.filename_gen()
        self.current_images: list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        self.current_hist: list[np.ndarray, np.ndarray]
        self.test_matrix_builder = kwargs.get('test', False)
        self.h_bins = h_bins
        self.l_bins = l_bins
        self.s_bins = s_bins
        self.epsilon = epsilon
        


    def img_read(self, img_data: str | np.ndarray, **kwargs):
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
    
    
    def pickle_gen(self, filename: str):
        with open(filename, 'rb') as f:
            img_dict = pickle.load(f, encoding='bytes')
            
        if self.image_key not in img_dict:
            raise KeyError(f"Key '{self.image_key}' not found in pickle file.")
        
        if self.simmatrix.size == 0:
            self.num_images = len(img_dict[self.image_key][:1000, :])
            self.simmatrix = np.zeros((self.num_images, self.num_images), dtype=np.float16)

        for img_index, img in enumerate(img_dict[self.image_key][:1000, :], start=1):
            if self.verbose:
                print(f'reading image {img_index}/{self.num_images}')
            yield img.reshape(3,32,32).transpose(1,2,0) # rgb array (32x32x3)
            
            
    # TODO gen for reading images
    def filename_gen(self):
        files = glob.glob(self.search_path, recursive=True)
        for file in files:
            yield file
      

    def filter_l_s(self, arr: np.ndarray):
        mask = (arr[:,:,1] > 20) & (arr[:,:,2] > 50)
        return arr[mask]


    def calculate_circ_mean(self, weights: np.ndarray):
        radians = np.arange(self.h_bins)*(360/self.h_bins)*(np.pi/180) # list of n_bins radians for each hue
        sin_sum = np.sum(weights*np.sin(radians))
        cos_sum = np.sum(weights*np.cos(radians))
        angle_mean = np.arctan2(sin_sum, cos_sum)
        angle_mean_deg = angle_mean*(180/np.pi)%360
        return angle_mean_deg


    def shift_arr(self, hls1_f: np.ndarray, hls2_f: np.ndarray) -> tuple[np.ndarray]:
        h1_quant, l1_quant, s1_quant = self.quantize_channels(hls1_f)
        h2_quant, l2_quant, s2_quant = self.quantize_channels(hls2_f)
        # his1 = self.create_hist(h1_quant, s1_quant, l1_quant)
        # his2 = self.create_hist(h2_quant, s2_quant, l2_quant)
        #h1,l1,s1 = cv.split(hls1_f)
        #h2,l2,s2 = cv.split(hls2_f)
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
        # angle_1, angle_2 = self.calculate_circ_mean(his1_h), self.calculate_circ_mean(his2_h)
        angle_1, angle_2 = np.argmax(his1_h), np.argmax(his2_h)
        # print(angle_1, angle_2)
        # print(angle_1+(self.h_bins/2-angle_1))
        his1_shifted_ang1 = np.roll(his1_h, (self.h_bins/2-angle_1))
        his2_shifted_ang1 =  np.roll(his2_h, (self.h_bins/2-angle_1))
        his1_shifted_ang2 =  np.roll(his1_h, (self.h_bins/2-angle_2))
        his2_shifted_ang2 =  np.roll(his2_h, (self.h_bins/2-angle_2))
        
        return his1_shifted_ang1, his2_shifted_ang1, his1_l, his2_l, his1_s, his2_s
        
    def quantize_channels(self, hls_arr: np.ndarray):
        h, l, s = self.filter_l_s(hls_arr)[:,0], hls_arr[:,:,1], hls_arr[:,:,2]
        
        return np.clip(h//(180//self.h_bins),a_min=0, a_max=self.h_bins-1), np.clip(l //(256//self.l_bins),a_min=0, a_max=self.l_bins-1), np.clip(s//(256//self.s_bins),a_min=0, a_max=self.s_bins)
    
    
    def create_hist(self, h_quant: np.ndarray, s_quant: np.ndarray, l_quant: np.ndarray):
        hist: np.ndarray = np.zeros((self.h_bins, self.s_bins, self.l_bins), dtype=np.float16)
        for i in range(h_quant.shape[0]):
            for j in range(h_quant.shape[1]):
                #print(h_quant[i,j], s_quant[i,j], l_quant[i,j])
                hist[h_quant[i,j], s_quant[i,j], l_quant[i,j]] += 1
        hist /= hist.sum()
        return hist
        
    
    def calculate_similarity(self, h1: np.ndarray, h2: np.ndarray, l1: np.ndarray, l2: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[float, float]:
        # sim_h = self.cos_sim(h1, h2)
        # sim_l = self.cos_sim(l1, l2)
        # sim_s = self.cos_sim(s1, s2)
        sim_h = cv.compareHist(h1, h2, cv.HISTCMP_BHATTACHARYYA)
        sim_l = cv.compareHist(l1, l2, cv.HISTCMP_BHATTACHARYYA)
        sim_s = cv.compareHist(s1, s2, cv.HISTCMP_BHATTACHARYYA)
        
        
        # sim_3 = cv.compareHist(his1_shifted_ang1, his2_shifted_ang1, cv.HISTCMP_BHATTACHARYYA)
        return sim_h, sim_l, sim_s
        #return np.max([sim_1, sim_2])


    def cos_sim(self, arr1: np.ndarray, arr2: np.ndarray):
        
        return np.dot(arr1.flatten(), arr2.flatten())/(np.linalg.norm(arr1)*np.linalg.norm(arr2))


    def show_histograms(self, img1, img2, h1, h2, l1, l2, s1, s2,sim_h, sim_l, sim_s, img_index) -> None:
        fig, axs = plt.subplots(4,2, figsize= (14,8))
        fig.suptitle(f"Sim_h: {sim_h}, Sim_l: {sim_l}, Sim_s: {sim_s}, Avg: {(6*sim_h+sim_l+2*sim_s)/9}, Index: {img_index+1}")
        axs[0, 0].imshow(img1)
        axs[0, 1].imshow(img2)
        
        ticks_h = [i for i in range(self.h_bins)]
        ticks_l = [i for i in range(self.l_bins)]
        ticks_s = [i for i in range(self.s_bins)]
        axs[0, 0].imshow(img1)
        axs[0, 1].imshow(img2)
        axs[1, 0].bar(ticks_h, h1.ravel())
        axs[1, 1].bar(ticks_h, h2.ravel())
        axs[2, 0].bar(ticks_l, l1.ravel())
        axs[2, 1].bar(ticks_l, l2.ravel())
        axs[3, 0].bar(ticks_s, s1.ravel())
        axs[3, 1].bar(ticks_s, s2.ravel())

        plt.show()
        
    def show_all_hists(self, img1, img2, h_hist1, h_hist2, l_hist1, l_hist2, s_hist1, s_hist2):
        fig, axs = plt.subplots(4,2, figsize= (14,8))
        ticks_h = [i for i in range(self.h_bins)]
        ticks_l = [i for i in range(self.l_bins)]
        ticks_s = [i for i in range(self.s_bins)]
        axs[0, 0].imshow(img1)
        axs[0, 1].imshow(img2)
        axs[1, 0].bar(ticks_h, h_hist1.ravel())
        axs[1, 1].bar(ticks_h, h_hist2.ravel())
        axs[2, 0].bar(ticks_l, l_hist1.ravel())
        axs[2, 1].bar(ticks_l, l_hist2.ravel())
        axs[3, 0].bar(ticks_s, s_hist1.ravel())
        axs[3, 1].bar(ticks_s, s_hist2.ravel())
        
        plt.show()
      
        
    def show_curr_imgs(self):
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(self.current_images[0])
        ax2.imshow(self.current_images[1])
        plt.show()
        
    def anlz_cycle(self):
        if self.mode == 'rgb_array':
            for filename in self.gen_filename:
                outer_gen = self.pickle_gen(filename)
                for out_index, out_loop in enumerate(outer_gen):
                    inner_gen = self.pickle_gen(filename)
                    for in_index, in_loop in enumerate(inner_gen):
                        self.current_images = [*self.img_read(out_loop), *self.img_read(in_loop)]
                        #testi = self.filter_l_s(self.current_images[1])
                        h1, h2, l1, l2, s1, s2 = self.shift_arr(self.current_images[1], self.current_images[3])
                        sim_h, sim_l, sim_s = self.calculate_similarity(h1, h2, l1, l2, s1, s2)
                        #self.show_all_hists(self.current_images[0], self.current_images[2], h1, h2, l1, l2, s1, s2)
                        self.show_histograms(self.current_images[0], self.current_images[2], h1, h2, l1, l2, s1, s2, sim_h, sim_l, sim_s, in_index)
                        self.simmatrix[out_index, in_index] = (6*sim_h+sim_l+2*sim_s)/9
                    print(f'Image {out_index+1} done')
                    
            with open(f'matrices/sim_matrix_n_bins_{self.h_bins}.pkl', 'wb') as f:
                pickle.dump(self.simmatrix, f)          
        if self.mode == 'img_file':
            pass
        
                    
        
        

if __name__ == "__main__":
    imganalyse = ImgColSimGet('cifar', mode = ('rgb_array', b'data'))
    imganalyse.anlz_cycle()

    
    
    # def kl_divergence(P, Q, epsilon=1e-10):
    #     P = np.array(P, dtype=np.float64) + epsilon
    #     Q = np.array(Q, dtype=np.float64) + epsilon
    #     P /= P.sum()
    #     Q /= Q.sum()
    #     return np.sum(P * np.log(P / Q))

    # #path = 'images/lilla.png'
    # path = 'images/HSD_duesseldorf_IMG_9904.jpg'
    # #path_2 = 'images/349.png'
    # path_2 = 'images/HSD_duesseldorf_IMG_9919.jpg'

    # # ct = ColorThief(path)
    # # ct2 = ColorThief(path_2)

    # rgb_img, hls_img = img_read(path)
    # rgb_img2, hls_img2 = img_read(path_2)

    # hls_f = hls_img#filter_l_s(hls_img)
    # hls_f2 = hls_img2#filter_l_s(hls_img2)
    
    
    # his1 = cv.calcHist([hls_f], [0], None, [90], [0, 181])
    # his2 = cv.calcHist([hls_f2], [0], None, [90], [0, 181])
    # shifted_arrs = shift_arr(hls_f, hls_f2)
    # simi, cos_simi = calculate_similarity(*shifted_arrs)
    # print(simi)
    # show_histograms(rgb_img, rgb_img2, his1, his2, *shifted_arrs, simi, cos_simi)

    # channels = cv.split(hlsss_filtered)
    # channels2 = cv.split(hlsss2_filtered)

    # histogram, bin_edges = np.histogram(hls_f[:,0], bins=180, range=(0, 181), density=True)
    # histogram2, bin_edges2 = np.histogram(hls_f2[:,0], bins=180, range=(0, 181), density=True)

    # hist = cv.calcHist([hls_f], [0], None, [180], [0, 181])
    # hist_shift = np.roll(hist, len(hist)//2)


    # hist2 = cv.calcHist([hls_f2], [0], None, [180], [0, 181])
    # hist2_shift = cv.calcHist([hls_f2], [0], None, [180], [0, 181])

    # sim = 'hist', cv.compareHist(hist, hist2, cv.HISTCMP_INTERSECT)
    # sim_shift = 'hist_shift', cv.compareHist(hist_shift, hist2_shift, cv.HISTCMP_INTERSECT)
    # print(channels)

    # hist = cv.calcHist(channels, [0], None, [100], [0, 361])
    # hist2 = cv.calcHist(channels2, [0], None, [100], [0, 361])


    # kl_divergence = np.sum(rel_entr(histogram, histogram2))
    # print(kl_divergence)
    # hlsss[:,:,0] += 20

    # plt.plot(hist)
    # plt.plot(hist2)
    # palette = ct.get_palette(color_count=5)
    # palette2 = ct2.get_palette(color_count=5)
    # TODO img = np.uint8([[color_rgb]])
    # colos_hls = [np.asarray(colorsys.rgb_to_hls(*i)) for i in palette]
    # colos_hls2 = [np.asarray(colorsys.rgb_to_hls(*i)) for i in palette2]

    # print(colos_hls)
    # print(colos_hls2)

    # for coloro1, coloro2 in zip(colos_hls, colos_hls2):
    #     print(np.linalg.norm(coloro1 - coloro2))


    # plt.imshow([[palette[i] for i in range(5)]])
    #plt.imshow([[palette2[i] for i in range(5)]])
    # plt.show()
    
    
    
    # glob
    # for i, 999
        # for j = i+1, 9999
    # TODO CIFAR-10 for all images with n-bins as hyper param
    # TODO tsne = TSNE(metric='precomputed')
    # TODO tsne with sim 1-sim_matrix 
