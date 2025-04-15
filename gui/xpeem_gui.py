import sys
import h5py
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPEEM HDF5 Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Layouts and widgets
        self.label = QLabel("No file loaded")
        self.canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.button = QPushButton("Load HDF5 File")
        self.button.clicked.connect(self.load_hdf5)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_hdf5(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open HDF5 File", "", "HDF5 Files (*.hdf5 *.nxs)")
        if file_path:
            try:
                with h5py.File(file_path, "r") as f:
                    entry = list(f.keys())[0]
                    data = f[entry]['scan_data']['data_02'][()]
                    self.label.setText(f"Loaded {file_path}\nData shape: {data.shape}")
                    self.plot_image(data)
            except Exception as e:
                self.label.setText(f"Error loading file: {e}")

    def plot_image(self, data):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        if data.ndim == 3:
            ax.imshow(data[data.shape[0] // 2], cmap='gray')
            ax.set_title("Middle Slice of 3D Datacube")
        else:
            ax.text(0.5, 0.5, "Unsupported data shape", ha='center')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec())
