import sys
import numpy as np
import h5py
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QSlider, QGridLayout
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPEEM HDF5 Viewer")
        self.setGeometry(200, 200, 1000, 700)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Metadata display
        self.metadata_label = QLabel("No file loaded")
        self.layout.addWidget(self.metadata_label)

        # Load button
        self.load_button = QPushButton("Load HDF5 File")
        self.load_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_button)

        # Figure
        self.canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.ax = self.canvas.figure.subplots()
        self.layout.addWidget(self.canvas)

        # Sliders
        self.slider_layout = QGridLayout()
        self.energy_slider = QSlider(Qt.Horizontal)
        self.energy_slider.setMinimum(0)
        self.energy_slider.valueChanged.connect(self.update_image)
        self.slider_layout.addWidget(QLabel("Slice:"), 0, 0)
        self.slider_layout.addWidget(self.energy_slider, 0, 1)

        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.valueChanged.connect(self.update_x_slice)
        self.slider_layout.addWidget(QLabel("X (vertical) cross-section:"), 1, 0)
        self.slider_layout.addWidget(self.x_slider, 1, 1)

        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(0)
        self.y_slider.valueChanged.connect(self.update_y_slice)
        self.slider_layout.addWidget(QLabel("Y (horizontal) cross-section:"), 2, 0)
        self.slider_layout.addWidget(self.y_slider, 2, 1)

        self.layout.addLayout(self.slider_layout)

        # Data placeholder
        self.data = None
        self.energy = None
        self.slice_type = 'energy'

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open HDF5 File", "", "HDF5 Files (*.h5 *.hdf5 *.nxs)")
        if filename:
            with h5py.File(filename, 'r') as f:
                try:
                    entry = list(f.keys())[0]
                    self.data = f[entry]['scan_data']['data_02'][()]
                    self.energy = f[entry]['scan_data']['actuator_1_1'][()]
                    self.metadata_label.setText(f"File loaded: {filename}\nShape: {self.data.shape}, Energy range: {self.energy.min():.2f}-{self.energy.max():.2f} eV")
                except Exception as e:
                    self.metadata_label.setText(f"Failed to load: {e}")
                    return

            # Set slider limits
            self.energy_slider.setMaximum(self.data.shape[0] - 1)
            self.energy_slider.setValue(self.data.shape[0] // 2)

            self.x_slider.setMaximum(self.data.shape[1] - 1)
            self.x_slider.setValue(self.data.shape[1] // 2)

            self.y_slider.setMaximum(self.data.shape[2] - 1)
            self.y_slider.setValue(self.data.shape[2] // 2)

            self.update_image()

    def update_image(self):
        idx = self.energy_slider.value()
        if self.data is not None:
            self.slice_type = 'energy'
            self.ax.clear()
            self.ax.imshow(self.data[idx, :, :], cmap='inferno', origin='lower')
            self.ax.set_title(f"Energy slice #{idx} at {self.energy[idx]:.2f} eV")
            self.canvas.draw()

    def update_x_slice(self):
        idx = self.x_slider.value()
        if self.data is not None:
            self.slice_type = 'x'
            self.ax.clear()
            self.ax.imshow(self.data[:, idx, :], cmap='viridis', origin='lower')
            self.ax.set_title(f"X cross-section @ x={idx}")
            self.canvas.draw()

    def update_y_slice(self):
        idx = self.y_slider.value()
        if self.data is not None:
            self.slice_type = 'y'
            self.ax.clear()
            self.ax.imshow(self.data[:, :, idx], cmap='cividis', origin='lower')
            self.ax.set_title(f"Y cross-section @ y={idx}")
            self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec())
