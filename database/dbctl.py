from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session
from database.image_model import ImgEntry, ImgConvertFailure
from features import ColorVecCalculator, OrbVecCalculator, BYOLVecCalculator, SiftVecCalculator, DINOVecCalculator
# from embeddings import Embedder
# from autoenc import AutoEncoder
from faiss import IndexFlatIP, IndexIDMap
import faiss
from tqdm import tqdm
import numpy as np
from queue import Queue
import threading
from sqlalchemy.exc import IntegrityError
from sqlmodel import select, func
from threadables import Enqueuer, Worker, worker
import sys
import cv2 as cv
from database import ImgConvertFailure
from PIL import Image

class DBController:
    def __init__(self, db_path: str, index_path: Path, kmeans_path: Path, byol_path: Path, img_drive_path: Path, threads = 4,  estimated_load = None, orb_length=500, color_length=26, byol_length=256, cap: int = 1000):
        # just_fix_windows_console() # colorama fix, does nothing on unix
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)
        self.idx = faiss.read_index(str(index_path)) if index_path.is_file() else IndexIDMap(IndexFlatIP(orb_length+color_length+byol_length))
        # self.embedder = Embedder(model_path="/Users/richardgox/Documents/4. Semester/GuglLens/database/autoencoder_full_131_best.pth")
        self.vector_length = orb_length
        self.files_to_process = Queue(maxsize=threads+1)
        self.vectors_to_index = Queue()
        self.index_path = index_path
        self.img_drive_path = img_drive_path
        self.threads = threads
        self.worker_done = 0
        self.kill_switch = threading.Event()
        self.images_done = 0
        self.tqdm = None
        if estimated_load:
            with Session(self.engine) as session:
               initial=session.exec(select(func.count(ImgEntry.id))).one()
            self.tqdm = tqdm(total=estimated_load, ncols=80, dynamic_ncols=False, initial=initial, smoothing=1, mininterval=10, file=sys.stdout)
        self.colorvec = ColorVecCalculator(color_length)
        self.kmeans_path = kmeans_path
        self.kmeans = faiss.read_index(str(kmeans_path)) if kmeans_path.is_file() else print("No kmeans trained")
        self.orbvec = OrbVecCalculator(self.kmeans, self.vector_length)
        self.byol_length = byol_length
        self.byolvec = BYOLVecCalculator(byol_path)
        self.cap = cap

    def populate_db(self, n=None):
        Enqueuer(self.files_to_process, self.img_drive_path.rglob("*"), self.filter_image_duplicates, self.threads, n, self.kill_switch).start()
        for _ in range(self.threads):
            Worker(self.files_to_process, self.enqueue_vec, self.on_worker_done, self.kill_switch).start() # calculate vector and write to output queue
        try:
            if self.tqdm and n: 
                self.tqdm.total = n
            worker(self.vectors_to_index, self.write_to_db, self.on_worker_done, self.kill_switch) # work non-wrapped to execute in main thread
        except KeyboardInterrupt:
            self.kill_switch.set()
        finally:
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
                
    def enqueue_vec(self, img_path: Path) -> None:
        try:
            vec, height, width = self.get_vec(str(img_path))
            self.vectors_to_index.put((img_path, height, width, vec))
        except Exception as e:
            print(f"Failed to process {str(img_path)}: {e}", file=sys.stderr)
            
    def get_vec(self, img_path: str):
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        scale = self.cap / max(arr.shape)
        if scale < 1:
            arr = cv.resize(arr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        #cvec = self.colorvec.gen_color_vec(arr)
        arr_g = cv.cvtColor(arr, cv.COLOR_RGB2GRAY)
        ovec = self.orbvec.gen_orb_vec(arr_g)
        # bvec = self.byolvec.gen_byol_vec(arr_g)
        # if bvec.shape[1] != self.byol_length: 
        #     raise ValueError(f"Mismatch between expected length '{self.byol_length}' and actual length '{bvec.shape[1]}'")
        #return np.concatenate((cvec, bvec), axis=1), img.height, img.width 
        return ovec, img.height, img.width 

    def get_closest_from_db(self, img_path: Path, k: int) -> list[str]:
        vec, _, _ = self.get_vec(img_path)
        faiss.normalize_L2(vec)
        _, I = self.idx.search(vec, k)
        with Session(self.engine) as session:
            imgs = []
            for i in I[0]:
                img = session.get(ImgEntry, int(i))
                if img:
                    imgs.append(img.path)
        if len(imgs) < k: imgs.append(None)
        return imgs
                
    def build_feat_kmeans(self, n=None, mode="orb"):
        Enqueuer(self.files_to_process, self.img_drive_path.rglob("*"), self.filter_image_duplicates, self.threads, n, self.kill_switch).start()
        for _ in range(self.threads):
            Worker(self.files_to_process, lambda path: self.enqueue_feat_vec(path, mode), self.on_worker_done, self.kill_switch).start()
        orbis = []
        def append_feat_vec(vec):
            if (i:=len(orbis)) % 10 == 0:
                print(i, "done")
            orbis.append(vec)
        try:
            worker(self.vectors_to_index, append_feat_vec, lambda: print("main_done"), self.kill_switch)
            print(f"list done with {len(orbis)} elements")
            arr = np.vstack(orbis)
            print("vstack done", arr.shape)
            np.save("feats-128threads.npy", arr)
            km = faiss.Kmeans(arr.shape[1], self.vector_length, niter=25, verbose = True, gpu=True, max_points_per_centroid=600000)
            km.train(arr)
            _, d = km.centroids.shape
            index = faiss.IndexFlatL2(d)
            index.add(km.centroids)
            faiss.write_index(index, str(self.kmeans_path))
            self.worker_done = 0
        except KeyboardInterrupt:
            np.save("backup-kmeans.npy", arr)
            self.kill_switch.set()
        print("finished-building orb")
        return index
    
    def enqueue_feat_vec(self, path:Path, mode="orb"):
        try:
            if mode == "orb":
                calculator = OrbVecCalculator
            elif mode == "sift":
                calculator = SiftVecCalculator
            else: raise ValueError("mode must be 'orb' or 'sift'")
            img = Image.open(path).convert('RGB')
            arr = np.array(img)
            scale = self.cap / max(arr.shape)
            if scale < 1:
                arr = cv.resize(arr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            vec, _, _ = calculator.gen_internal_embedding(arr, self.vector_length)
            if vec is None: return
            self.vectors_to_index.put(vec)
        except ImgConvertFailure as e:
            print(str(path), file=sys.stderr)
        
     
    def on_worker_done(self):
        self.worker_done += 1
        if self.worker_done >= self.threads:
            self.vectors_to_index.put(None)
    
 
if __name__ == "__main__":
    # import time
    # db = ImgDBMaker(Path("test.db"), Path('Index_db.faiss'), Path("/Volumes/Big Data/data/image_data"))
    db = DBController(Path("test.db"), Path('Index_db.faiss'), Path("images"), threads=4, estimated_load=540_000)
    # start = time.time()
    # print(time.time()-start)