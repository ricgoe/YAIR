from pathlib import Path
from sqlmodel import SQLModel, Field, create_engine, Session, select
from PIL import Image, UnidentifiedImageError
from embeddings import Embedder
from faiss import IndexFlatIP, IndexFlatL2, IndexIDMap
import faiss
from autoenc import AutoEncoder
from tqdm import tqdm



DATA_LOCATION = Path("/Volumes/Big Data/data/image_data")


class ImgEntry(SQLModel, table = True):
    id: int | None = Field(default=None, primary_key=True)
    path: str
    width: int
    height: int
    #faiss_id: int

class ImgDBMaker:
    def __init__(self, db_path: str, img_drive_path: Path):
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.idx = IndexIDMap(IndexFlatIP(64))
        self.embedder = Embedder(model_path="/Users/richardgox/Documents/4. Semester/GuglLens/database/autoencoder_full_131_best.pth")
        SQLModel.metadata.create_all(self.engine)

    def populate_db(self, n=None):
        i = 0
        for file in tqdm(DATA_LOCATION.rglob("*"), total=540_000):
            if n is not None and i > n: break
            try:
                try:
                    if file.is_dir(): continue
                    with Image.open(file) as img:
                        img_entry = ImgEntry(path=str(file), width=img.width, height=img.height)
                    self.write_to_db(img_entry)
                    i += 1
                except UnidentifiedImageError:
                    print(f'{file} is not a valid image')
            except KeyboardInterrupt:
                break
        faiss.write_index(self.idx, 'Index_db.faiss')
                
        
    def write_to_db(self, img: ImgEntry):
        try:
            with Session(self.engine) as session:
                session.add(img)
                session.commit()
                session.refresh(img)
            embedding = self.embedder.gen_embedding(img.path)
            self.idx.add_with_ids(embedding, img.id)
        except Exception:
            with Session(self.engine) as session:
                session.delete(img)
                session.commit()
            

if __name__ == "__main__":
    db = ImgDBMaker("test.db", DATA_LOCATION)
    db.populate_db()
    # fi = faiss.read_index('Index_db.faiss')
    # print(fi.is_trained)
    # embeddings = faiss.vector_to_array(fi.id_map)  # This is a numpy array (float32)
    # print(embeddings.shape)  # (num_vectors, vector_dim)
    # for i in range(1, 3742):
    #     if i != embeddings[i-1]:
    #         print(i, embeddings[i-1])
    #         break
    # print(embeddings[:-1])