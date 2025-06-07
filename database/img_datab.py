from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session
from ImgModel import ImgEntry
# from embeddings import Embedder
# from autoenc import AutoEncoder
from colorvec1 import ColorVecCalculator
from faiss import IndexFlatIP, IndexFlatL2, IndexIDMap
import faiss
from tqdm import tqdm
import numpy as np
from queue import Queue
from collections.abc import Callable
import threading
from sqlalchemy.exc import IntegrityError

class ImgDBMaker:
    def __init__(self, db_path: str, index_path: Path, img_drive_path: Path, threads = 4, cap = None):
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
        self.cap = cap

    def populate_db(self, n=None):
        threading.Thread(target=self.enqueuer, args=(self.cap,)).start() # file_queue
        for _ in range(self.threads):
            threading.Thread(target=self.worker, args=(self.files_to_process, self.process_file, self.on_worker_done)).start() # calculate vector and write to output queue
        try:
            self.worker(self.vectors_to_index, self.write_to_db) # currently running on main thread. call line below for diffrent thread (?necessary?)
            #threading.Thread(target=self.worker, args=(self.vectors_to_index, self.write_to_db)).start()
        except KeyboardInterrupt:
            self.kill_switch.set()
        faiss.write_index(self.idx, str(self.index_path))
        
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
        if self.images_done % 500 == 0: faiss.write_index(self.idx, str(self.index_path))
                
    def process_file(self, img_path: Path):
        vec, height, width = self.colorvec.gen_color_vec(img_path)
        # TODO concatinate with embeddings and weight properly
        self.vectors_to_index.put((img_path, height, width, vec))
        
    def on_worker_done(self):
        self.worker_done += 1
        if self.worker_done >= self.threads:
            self.vectors_to_index.put(None)
        
    def enqueuer(self, n=None):
        i = 0
        for file in self.img_drive_path.rglob("*"):
            if file.is_dir():
                continue
            self.files_to_process.put(file)
            i += 1
            if n is not None and i >= n:
                break
            if self.kill_switch.is_set():
                print("Enqueuer killed")
                return
        for _ in range(self.threads):
            self.files_to_process.put(None)  # Sentinels to stop workers
        print("Enqueuer exited gracefully")
    
    def worker(self, queue: Queue, func: Callable, exit_func: Callable=None):
        while True:
            result = queue.get()
            if result is None:
                if exit_func is not None: exit_func()
                break
            if self.kill_switch.is_set():
                print("Worker killed")
                return
            func(result)
            queue.task_done()
        print("Worker exited gracefully")

 
if __name__ == "__main__":
    import time
    # db = ImgDBMaker(Path("test.db"), Path('Index_db.faiss'), Path("/Volumes/Big Data/data/image_data"))
    db = ImgDBMaker(Path("test.db"), Path('Index_db.faiss'), Path("images"), threads=4, cap=100)
    start = time.time()
    db.populate_db(n=100)
    print(time.time()-start)