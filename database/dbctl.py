from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session
from database import ImgEntry
from features import ColorVecCalculator, OrbVecCalculator, SiftVecCalculator, DINOVecCalculator
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
from PIL import Image

class DBController:
    def __init__(self, db_path: str, index_path: Path, kmeans_path: Path, img_drive_path: Path, threads = 4,  estimated_load = None, feat_length=500, color_length=26, dino_length=384, cap: int = 1000):
        # just_fix_windows_console() # colorama fix, does nothing on unix
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)
        self.img_drive_path = img_drive_path
        self.vector_length = feat_length
        
        self.kmeans_path = kmeans_path
        self.index_path = index_path
        self.idx = faiss.read_index(index_path.resolve()) if index_path.is_file() else IndexIDMap(IndexFlatIP(feat_length+color_length+dino_length))
        self.kmeans = faiss.read_index(kmeans_path.resolve()) if kmeans_path.is_file() else print("No kmeans trained")
        
        self.threads = threads
        self.files_to_process = Queue(maxsize=threads+1)
        self.vectors_to_index = Queue()
        self.kill_switch = threading.Event()
        self.worker_done = 0
        self.images_done = 0
        
        self.dino_length = dino_length
        self.dinovec = DINOVecCalculator()
        self.siftvec = SiftVecCalculator(self.kmeans, self.vector_length)
        self.colorvec = ColorVecCalculator(color_length)
        self.cap = cap
        self.estimated_load = estimated_load

    def populate_db(self, n=None):
        if self.estimated_load:
            with Session(self.engine) as session:
               initial=session.exec(select(func.count(ImgEntry.id))).one()
            self.tqdm = tqdm(total=self.estimated_load, ncols=80, dynamic_ncols=False, initial=initial, smoothing=1, mininterval=10, file=sys.stdout)
        if n: self.tqdm.total = n
        Enqueuer(self.files_to_process, self.img_drive_path.rglob("*"), self.filter_image_duplicates, self.threads, n, self.kill_switch).start()
        for _ in range(self.threads):
            Worker(self.files_to_process, self.enqueue_vec, self.on_worker_done, self.kill_switch).start() # calculate vector and write to output queue
        try:
            worker(self.vectors_to_index, self.write_to_db, self.on_worker_done, self.kill_switch) # work non-wrapped to execute in main thread
        except KeyboardInterrupt:
            self.kill_switch.set()
        finally:
            faiss.write_index(self.idx, self.index_path.resolve())
        
    def filter_image_duplicates(self, img_path: Path):
        if not img_path.is_file(): return False
        with Session(self.engine) as session:
            exists = session.exec(select(ImgEntry).where(ImgEntry.path == img_path.relative_to(self.img_drive_path))).first()
            if exists: return False
        return True
    
    def write_to_db(self, result: tuple[Path, np.ndarray]):
        img_path, height, width, vec = result
        img = ImgEntry(path=img_path.relative_to(self.img_drive_path), width=width, height=height)
        with Session(self.engine) as session:
            try:
                session.add(img)
                session.flush()
                session.refresh(img)

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
        if self.images_done % 500 == 0: faiss.write_index(self.idx, self.index_path.resolve())
                
    def enqueue_vec(self, img_path: Path) -> None:
        try:
            path = img_path.resolve()
            vec, height, width = self.get_vec(path)
            self.vectors_to_index.put((img_path, height, width, vec))
        except Exception as e:
            print(f"Failed to process {path}: {e}", file=sys.stderr)
            
    def get_vec(self, img_path: Path):
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        scale = self.cap / max(arr.shape)
        if scale < 1:
            arr = cv.resize(arr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        arr_g = cv.cvtColor(arr, cv.COLOR_RGB2GRAY)
        # if dvec.shape[1] != self.dino_length: 
        #     raise ValueError(f"Mismatch between expected length '{self.dino_length}' and actual length '{dvec.shape[1]}'")
        ratios=[
                (self.dinovec.gen_dino_vec(arr), 16), 
                (self.colorvec.gen_color_vec(arr), 11),
                (self.siftvec.gen_sift_vec(arr_g), 4),
                ] 
        for v, _ in ratios: faiss.normalize_L2(v)
        total = sum([i for _, i in ratios])
        weighted = [np.sqrt(i / total) * vector for vector, i in ratios]
        vec = np.concatenate(weighted, axis=1)
        faiss.normalize_L2(vec)
        return vec, img.height, img.width 

    def get_closest_from_db(self, img_path: Path, k: int, img_path2: Path = None, mix: float = None) -> list[str]:
        vec, _, _ = self.get_vec(img_path)
        faiss.normalize_L2(vec)
        if img_path2 and mix is not None:
            vec2, _, _ = self.get_vec(img_path2)
            faiss.normalize_L2(vec2)
            vec = vec * mix + vec2 * (1-mix)
            faiss.normalize_L2(vec)
        _, I = self.idx.search(vec, k)
        with Session(self.engine) as session:
            ids = [int(id) for id in I[0]]
            results = session.exec(select(ImgEntry).where(ImgEntry.id.in_(ids)))
            results_map = {img.id: img.path for img in results}
            imgs = [results_map[key] for key in ids if key in results_map]
        if len(imgs) < k: imgs.append(None)
        return imgs
                
    def build_feat_kmeans(self, n=None, mode="orb"):
        if self.estimated_load:
            self.tqdm = tqdm(total=self.estimated_load, ncols=80, dynamic_ncols=False, smoothing=1, mininterval=10, file=sys.stdout)
        if n: self.tqdm.total = n
        Enqueuer(self.files_to_process, self.img_drive_path.rglob("*"), lambda p: p.is_file(), self.threads, n, self.kill_switch).start()
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
            faiss.write_index(index, self.kmeans_path.resolve())
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
        except Exception:
            print(path, file=sys.stderr)
        
     
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