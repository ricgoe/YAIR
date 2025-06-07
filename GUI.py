import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QSlider
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QUrl
from math import ceil, floor
from database import DBController
from pathlib import Path

GRID_SIZE = 3

class ImageDropWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drag and Drop Image Viewer")
        self.setAcceptDrops(True)
        self.showMaximized()

        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)

        self.control = QWidget()
        self.control_layout = QVBoxLayout(self.control)

        self.scroll_area = QScrollArea()
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
        
        self.db = DBController(Path("test.db"), Path('Index_db.faiss'), Path("images"), threads=6, estimated_load=540_000)
        

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                pixmap = QPixmap(path)
                self.label.setPixmap(pixmap.scaled(
                    self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.populate_closest(self.db.get_closes_from_db(Path(path), 9))
                break
            
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
        
    def populate_closest(self, paths: list[str]):
        idx = 0 
        size = self.images.size() / (GRID_SIZE+0.5)
        for row in range(ceil(len(paths) / GRID_SIZE)):
            for col in range(3):
                if idx >= len(paths): break
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 1px solid #ccc; background-color: white;")
                pixmap = QPixmap(paths[idx])
                scaled_pixmap = pixmap.scaled(
                    size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)
                self.images_layout.addWidget(label, row, col)
                idx += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = ImageDropWidget()
    widget.show()
    sys.exit(app.exec())
