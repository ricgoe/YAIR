from pathlib import Path
from sqlmodel import SQLModel, Field, create_engine, Session, select
from PIL import Image
from embeddings import Embedder
from faiss import IndexFlatIP, IndexFlatL2



DATA_LOCATION = Path("/Volumes/Big Data/data/image_data")


class ImgEntry(SQLModel, table = True):
    id: int | None = Field(default=None, primary_key=True)
    path: str
    width: int
    height: int
    faiss_id: int

class ImgDBMaker:
    def __init__(self, db_path: str, img_drive_path: Path):
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.idx = IndexFlatIP(64)
        SQLModel.metadata.create_all(self.engine)

    def populate_db(self):
        for file in DATA_LOCATION.rglob("*"):
            with Image.open(file) as img:
                img_entry = ImgEntry(path=str(file), width=img.width, height=img.height)
                self.write_to_db(img_entry)
            break

    def write_to_db(self, img: ImgEntry):
        with Session(self.engine) as session:
            session.add(img)
            session.commit()
            session.refresh(img)
        embedder = Embedder(model_path="autoencoder/ae.pt", img=img.path)
        embedding = embedder.gen_embedding()
        self.idx.add_with_ids(1, embedding, img.id) 

if __name__ == "__main__":
    db = ImgDBMaker("test.db", DATA_LOCATION)
    db.populate_db()