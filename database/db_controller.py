from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session
from database.image_model import ImgEntry, ImgConvertFailure
from database.colorvec import ColorVecCalculator
# from embeddings import Embedder
# from autoenc import AutoEncoder
from faiss import IndexFlatIP, IndexFlatL2, IndexIDMap
import faiss
from tqdm import tqdm
import numpy as np
from queue import Queue
import threading
from sqlalchemy.exc import IntegrityError
from sqlmodel import select, func
from threadables import Enqueuer, Worker, worker

class DBController:
    def __init__(self, db_path: str, index_path: Path, img_drive_path: Path, threads = 4,  estimated_load = None):
        # just_fix_windows_console() # colorama fix, does nothing on unix
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)
        self.idx = faiss.read_index(str(index_path)) if index_path.is_file() else IndexIDMap(IndexFlatIP(26))
        # self.embedder = Embedder(model_path="/Users/richardgox/Documents/4. Semester/GuglLens/database/autoencoder_full_131_best.pth")
        self.colorvec = ColorVecCalculator()
        self.files_to_process = Queue(maxsize=threads+1)
        self.vectors_to_index = Queue()
        self.images_done = 0
        self.worker_done = 0
        self.img_drive_path = img_drive_path
        self.index_path = index_path
        self.threads = threads
        self.kill_switch = threading.Event()
        self.tqdm = None
        if estimated_load:
            with Session(self.engine) as session:
                estimated_load -=session.exec(select(func.count(ImgEntry.id))).one()
            self.tqdm = tqdm(total=estimated_load)

    def populate_db(self, n=None):
        Enqueuer(self.files_to_process, self.img_drive_path.rglob("*"), self.filter_image_duplicates, self.threads, n, self.kill_switch).start()
        for _ in range(self.threads):
            Worker(self.files_to_process, self.process_file, self.on_worker_done, self.kill_switch).start() # calculate vector and write to output queue
        try:
            if self.tqdm and n: 
                self.tqdm.total = n
            worker(self.vectors_to_index, self.write_to_db, self.on_worker_done, self.kill_switch) # work non-wrapped to execute in main thread
        except KeyboardInterrupt:
            self.kill_switch.set()
        faiss.write_index(self.idx, str(self.index_path))
        
    def filter_image_duplicates(self, img_path: Path):
        if not img_path.is_file(): return False
        with Session(self.engine) as session:
            exists = session.exec(select(ImgEntry).where(ImgEntry.path == str(img_path))).first()
            if exists: return False
        return True
    
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
        self.images_done += 1
        if self.tqdm: self.tqdm.update(1)
        if self.images_done % 500 == 0: faiss.write_index(self.idx, str(self.index_path))
                
    def process_file(self, img_path: Path):
        try:
            vec, height, width = self.colorvec.gen_color_vec(img_path)
            # TODO concatinate with embeddings and weight properly
            self.vectors_to_index.put((img_path, height, width, vec))
        except ImgConvertFailure as e:
            print(e)
        
    def on_worker_done(self):
        self.worker_done += 1
        if self.worker_done >= self.threads:
            self.vectors_to_index.put(None)
 
if __name__ == "__main__":
    # import time
    # db = ImgDBMaker(Path("test.db"), Path('Index_db.faiss'), Path("/Volumes/Big Data/data/image_data"))
    db = DBController(Path("test.db"), Path('Index_db.faiss'), Path("images"), threads=4, estimated_load=540_000)
    # start = time.time()
    db.populate_db(n=100)
    # print(time.time()-start)