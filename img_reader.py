import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from colorthief import ColorThief
import colorsys

def img_read(filename):
    bgr = cv.imread(filename)
    hls = cv.cvtColor(bgr, cv.COLOR_BGR2HLS)
    
    return hls

def filtered(arr):
    mask = (arr[:,:,1] > 0.25) & (arr[:,:,2] > 0.1)
    return arr[mask]


# def kl_divergence(P, Q, epsilon=1e-10):
#     P = np.array(P, dtype=np.float64) + epsilon
#     Q = np.array(Q, dtype=np.float64) + epsilon
#     P /= P.sum()
#     Q /= Q.sum()
#     return np.sum(P * np.log(P / Q))



path = 'images/lilla.png'
path = 'images/HSD_duesseldorf_IMG_9904.jpg'
path_2 = 'images/349.png'
path_2 = 'images/HSD_duesseldorf_IMG_9904.jpg'

ct = ColorThief(path)
ct2 = ColorThief(path_2)

hls_img = img_read(path)
hls_img2 = img_read(path_2)

hls_f = filtered(hls_img)
hls_f2 = filtered(hls_img2)

# channels = cv.split(hlsss_filtered)
# channels2 = cv.split(hlsss2_filtered)

# histogram, bin_edges = np.histogram(hls_f[:,0], bins=180, range=(0, 181), density=True)
# histogram2, bin_edges2 = np.histogram(hls_f2[:,0], bins=180, range=(0, 181), density=True)

hist = cv.calcHist([hls_f], [0], None, [180], [0, 181])
hist2 = cv.calcHist([hls_f2], [0], None, [180], [0, 181])

print('hist', cv.compareHist(hist, hist2, cv.HISTCMP_INTERSECT))
# print(channels)

# hist = cv.calcHist(channels, [0], None, [100], [0, 361])
# hist2 = cv.calcHist(channels2, [0], None, [100], [0, 361])


# kl_divergence = np.sum(rel_entr(histogram, histogram2))
# print(kl_divergence)
# hlsss[:,:,0] += 20

# plt.plot(hist)
# plt.plot(hist2)
palette = ct.get_palette(color_count=5)
palette2 = ct2.get_palette(color_count=5)
# TODO img = np.uint8([[color_rgb]])
# colos_hls = [np.asarray(colorsys.rgb_to_hls(*i)) for i in palette]
# colos_hls2 = [np.asarray(colorsys.rgb_to_hls(*i)) for i in palette2]

# print(colos_hls)
# print(colos_hls2)

# for coloro1, coloro2 in zip(colos_hls, colos_hls2):
#     print(np.linalg.norm(coloro1 - coloro2))


plt.imshow([[palette[i] for i in range(5)]])
#plt.imshow([[palette2[i] for i in range(5)]])
plt.show()
