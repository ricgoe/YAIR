from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session
from ImgModel import ImgEntry
from PIL import Image, UnidentifiedImageError
from embeddings import Embedder
from colorvec1 import ColorVecCalculator
from faiss import IndexFlatIP, IndexFlatL2, IndexIDMap
import faiss
from autoenc import AutoEncoder
from tqdm import tqdm
import numpy as np

DATA_LOCATION = Path("/Volumes/Big Data/data/image_data")

class ImgDBMaker:
    def __init__(self, db_path: str, img_drive_path: Path):
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)
        self.idx = IndexIDMap(IndexFlatIP(26))
        self.embedder = Embedder(model_path="/Users/richardgox/Documents/4. Semester/GuglLens/database/autoencoder_full_131_best.pth")
        self.colorvec = ColorVecCalculator()

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
                
        
    def write_to_db(self, result: tuple[Path, np.ndarray]):
        img_path, height, width, vec = result
        img = ImgEntry(path=str(img_path), width=width, height=height)
        with Session(self.engine) as session:
            try:
                session.add(img)
                session.flush()
                session.refresh(img)
    
                faiss.normalize_L2(vec)
                self.idx.add_with_ids(vec, img.id)
            except Exception as e:
                if isinstance(e, IntegrityError): print(f"{img_path} already exists in database")
                else: print("ERROR", e)
                session.rollback()
                if img.id:
                    self.idx.remove_ids(np.array([img.id]))
            finally:
                session.commit()
                self.idx.remove_ids(np.array([img.id]))
            

if __name__ == "__main__":
    db = ImgDBMaker("test.db", DATA_LOCATION)
    db.populate_db()