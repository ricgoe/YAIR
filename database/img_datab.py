import sqlite3
from pathlib import Path
from sqlmodel import SQLModel, Field, create_engine, Session, select




class ImgEntry(SQLModel, table = True):
    id: int | None = Field(default=None, primary_key=True)
    path: str
    width: int
    height: int
    
    

class ImgDBMaker:
    def __init__(self, db_path: Path, img_drive_path: Path):
        self.engine = create_engine(db_path)
        SQLModel.metadata.create_all(self.engine)


    def write_to_db(self, img):
        with Session(self.engine) as session:
            session.add()
        
    
    
    def process_img(self, img_path: Path) -> None:
    try:
        img = cv.imread(str(img_path),cv.IMREAD_GRAYSCALE)
        if img is None:
            return
        img = np.float32(img)/255
        img_resized = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
        img_resized = np.clip(img_resized, 0, 1)
        rez = np.reshape(img_resized, (1, 32, 32))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
    return rez
