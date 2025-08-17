import cv2 as cv
import numpy as np


class SiftVecCalculator:
    
    def __init__(self, model, vector_length):
        self.vector_length = vector_length
        self.kmeans = model
        
    def gen_sift_vec(self, img: np.ndarray):
        embedding = SiftVecCalculator.gen_internal_embedding(img, self.vector_length)
        if embedding is None:
            embedding = np.zeros((self.vector_length,128))
        _, label = self.kmeans.search(embedding, 1)
        label = label.ravel()
        vec = np.bincount(label, minlength=self.vector_length)

        vec = vec.reshape(1, -1).astype(np.float32)
        return vec
    
    def gen_internal_embedding(img: np.ndarray, vector_length):
        sift = cv.SIFT_create(nfeatures=vector_length)
        _, embedding = sift.detectAndCompute(img, None)
        return embedding
    
if __name__ == "__main__":
    import faiss
    a = SiftVecCalculator.gen_internal_embedding(cv.imread("/Volumes/Big Data/data/image_data/DAISY_2025/20250328_101537.jpg", cv.IMREAD_GRAYSCALE), 500)
    print(a.shape)