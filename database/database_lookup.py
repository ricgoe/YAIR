import numpy as np
import faiss
from faiss import IndexFlatIP, IndexFlatL2, IndexIDMap
from database.image_model import ImgEntry
from sqlmodel import SQLModel, create_engine, Session

engine = create_engine(f"sqlite:///test.db")
SQLModel.metadata.create_all(engine)

# db = ImgDBMaker("test.db", DATA_LOCATION)
# db.populate_db()
fi: IndexIDMap = faiss.read_index('Index_db.faiss')
index : IndexFlatIP = fi.index
print(fi.ntotal)
# print(fi.is_trained)
ids = faiss.vector_to_array(fi.id_map)  # This is a numpy array (float32)
pos = int(np.where(ids == 98)[0][0])
print(pos)
vec: np.ndarray = index.reconstruct(pos, None).reshape(1, -1)
faiss.normalize_L2(vec)
D, I = fi.search(vec, 5)
print(D.min(), D.max())
print(I)
with Session(engine) as session:
    for i in I[0]:
        img = session.get(ImgEntry, int(i))
        print(img.path)
    print(session.get(ImgEntry, 98).path)
# print(embeddings)  # (num_vectors, vector_dim)
# print(i,"\n\n\n\n")
# print(vec)
# print(index.reconstruct(int(np.where(ids == i)[0][0]), None).reshape(1, -1))