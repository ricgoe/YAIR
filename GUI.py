import sys
import os
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QSlider
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtCore import Qt, QRunnable, QThreadPool, Signal, QObject
from math import ceil, floor
from database import DBController
from pathlib import Path

GRID_SIZE = 3
PRE_FETCH = 100

class Container(QObject):
    finished = Signal(object)
class DBWorker(QRunnable):
    def __init__(self, db: DBController, search_img, total):
        super().__init__()
        self.signals = Container()
        self.db = db
        self.search_img = search_img
        self.total = total

    def run(self):
        result = self.db.get_closest_from_db(Path(self.search_img), self.total)
        self.signals.finished.emit(result)
        
class ScalerWorker(QRunnable):
    def __init__(self, paths, size, location=""):
        super().__init__()
        self.signals = Container()
        self.paths = paths
        self.size = size
        self.location = location

    def run(self):
        results = [ # of form [(path, pixmap), ...]
            QPixmap(Path(self.location, *Path(path).parts).resolve()).scaled(
                self.size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ) if path else None
            for path in self.paths
        ]
        self.signals.finished.emit(results)

class ImageDropWidget(QWidget):
    def __init__(self, db: DBController, location: str):
        super().__init__()
        self.setWindowTitle("Drag and Drop Image Viewer")
        self.setAcceptDrops(True)
        self.showMaximized()

        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)

        self.control = QWidget()
        self.control_layout = QVBoxLayout(self.control)

        self.scroll_area = QScrollArea()
        self.scroll_bar = self.scroll_area.verticalScrollBar()
        self.scroll_bar.valueChanged.connect(self.scrolled)
        self.scroll_area.setWidgetResizable(True)
        self.images = QWidget()
        self.images_layout = QGridLayout(self.images)
        self.images.setLayout(self.images_layout)
        self.scroll_area.setWidget(self.images)

        self.main_layout.addWidget(self.control,4)
        self.main_layout.addWidget(self.scroll_area,6)
        
        self.label = QLabel("Drop an image here")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 2px dashed gray; padding: 20px;")
        self.control_panel = QWidget()
        self.control_panel_layout = QVBoxLayout(self.control_panel)
        self.control_layout.addWidget(self.label, 4)
        self.control_layout.addWidget(self.control_panel,1)
        
        self.slider1 = self.make_slider("Slider1", 0, 100)
        self.slider2 = self.make_slider("Slider2", 0, 100)
        self.slider3 = self.make_slider("Slider3", 0, 100)
        
        self.db = db
        self.search_img = None
        self.location = location
        # Multithreading
        self.caching = 1
        self.cached_closest:list[str] = []
        self.cached_pixmaps:list[QPixmap] = []
        self.thread_pool = QThreadPool.globalInstance()
        
    def cache_closest(self):
        worker = DBWorker(self.db, self.search_img, self.caching * PRE_FETCH)
        self.caching += 1
        worker.signals.finished.connect(self.paths_to_cache)
        self.thread_pool.start(worker)
    
    def paths_to_cache(self, result):
        self.cached_closest = result
    
    def cache_pixmap(self):
        start = GRID_SIZE**2 * (self.paginate+1) + GRID_SIZE
        end = start + GRID_SIZE**2
        worker = ScalerWorker(self.fetch_or_cached_paths(start, end), self.images.size() / (GRID_SIZE+0.5), self.location)
        worker.signals.finished.connect(self.pixmap_to_cache)
        self.thread_pool.start(worker)
    
    def pixmap_to_cache(self, result):
        self.cached_pixmaps = result
        
    def fetch_or_cached_paths(self, start, end):
        if len(self.cached_closest) > end: 
            return [self.cached_closest[i] for i in range(start, end)]
        else:
            paths = self.db.get_closest_from_db(Path(self.search_img), end)
        if end * GRID_SIZE**2 > len(self.cached_closest) and not None in self.cached_closest: self.cache_closest()
        return paths[start:end]
    
    def get_or_cached_pixmaps(self):
        start = GRID_SIZE**2 * self.paginate
        end = start + GRID_SIZE**2
        if self.paginate == 0: 
            end += GRID_SIZE
        else:
            start += GRID_SIZE
            end += GRID_SIZE
        n_maps = end - start
        if len(self.cached_pixmaps) >= n_maps:
            cache = [self.cached_pixmaps.pop(0) for _ in range(n_maps)]
            self.cache_pixmap()
            return cache
        else:
            pixmaps = [
                QPixmap(Path(self.location, *Path(path).parts).resolve()).scaled(
                    self.images.size() / (GRID_SIZE+0.5),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                if path else None
                for path in self.fetch_or_cached_paths(start, end)
            ]
        self.cache_pixmap()
        return pixmaps

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                pixmap = QPixmap(path)
                self.label.setPixmap(pixmap.scaled(
                    self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.search_img = path
                self.paginate = 0
                self.populate_closest()
                break
    
    def scrolled(self, value):
        if value >= self.scroll_bar.maximum()*0.95:
            self.paginate += 1
            self.populate_closest()
            
    def make_slider(self, name, minV, maxV):
        s = QWidget()
        l = QHBoxLayout(s)
        label = QLabel(name)
        v_label = QLabel("0")
        slider = QSlider(Qt.Horizontal)
        slider.setValue(0)
        slider.setMinimum(minV)
        slider.setMaximum(maxV)
        slider.valueChanged.connect(lambda value: v_label.setText(str(value)))
        l.addWidget(label)
        l.addWidget(slider)
        l.addWidget(v_label)
        self.control_panel_layout.addWidget(s)
        return s
        
    def populate_closest(self):
        idx = 0 
        pixmaps = self.get_or_cached_pixmaps()
        for row in range(ceil(len(pixmaps) / GRID_SIZE)):
            for col in range(GRID_SIZE):
                if idx >= len(pixmaps) or pixmaps[idx] is None: 
                    break
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 1px solid #ccc;")
                label.setPixmap(pixmaps[idx])
                self.images_layout.addWidget(label, row+GRID_SIZE*self.paginate, col)
                idx += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    db = DBController(Path("test.db"),
                  Path('Index_db.faiss'),
                  Path("models/binflat.faiss"),
                  Path("models/byol_256.pth"),
                  Path("/Volumes/Big Data/data"),
                  threads=4, estimated_load=200_000,
                  orb_length=1024, color_length=0, byol_length=0)
    widget = ImageDropWidget(db, "/Volumes/Big Data/data")
    widget.show()
    sys.exit(app.exec())
