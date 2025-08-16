import cv2 as cv
import numpy as np
from database import ImgConvertFailure


class OrbVecCalculator:
    
    def __init__(self, model, vector_length):
        self.vector_length = vector_length
        self.kmeans = model
        
    def gen_orb_vec(self, img: np.ndarray):
        embedding = OrbVecCalculator.gen_internal_embedding(img, self.vector_length)
        if embedding is None:
            embedding = np.zeros((500,256))
        _, label = self.kmeans.search(embedding, 1)
        label = label.ravel()
        vec = np.bincount(label, minlength=self.vector_length)

        vec = vec.reshape(1, -1).astype(np.float32)
        return vec
    
    def gen_internal_embedding(img: np.ndarray, vector_length):
        orb = cv.ORB_create(nfeatures=vector_length)
        _, embedding = orb.detectAndCompute(img, None)
        return np.unpackbits(embedding, axis=1)
    
if __name__ == "__main__":
    import faiss
    cvc = OrbVecCalculator(faiss.read_index('kmeans.faiss'), 500)
    a=cvc.gen_orb_vec("/raid/richard/image_data/DAISY_2025/20250328_101537.jpg")
    print(a)