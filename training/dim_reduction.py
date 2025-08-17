import cupy as cp
from cuml.manifold import UMAP
# from umap import UMAP
import numpy as np, faiss

index = faiss.read_index('ImageIDX.faiss')
base = index.index
n, d = base.ntotal, base.d
X = base.reconstruct_n(0, n).astype(np.float32)

Xg = cp.asarray(X)
um = UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=0, n_components=2)
Yg = um.fit_transform(Xg)
Y  = cp.asnumpy(Yg)
# Y = um.fit_transform(X)

labels = faiss.vector_to_array(index.id_map).astype(np.int64)
coords = np.column_stack([labels, Y])
np.save("coords.npy", coords)