from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QLabel,
)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.update_plot)
        layout.addWidget(self.slider)

        self.energies = []
        self.energy_label = QLabel("Energy: N/A")
        self.energy_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.energy_label)
        # Add vmin and vmax controls
        scale_layout = QHBoxLayout()
        self.vmin_input = QLineEdit()
        self.vmax_input = QLineEdit()
        self.vmin_input.setPlaceholderText("vmin")
        self.vmax_input.setPlaceholderText("vmax")
        self.vmin_input.editingFinished.connect(self.update_plot)
        self.vmax_input.editingFinished.connect(self.update_plot)
        scale_layout.addWidget(QLabel("Intensity Scale:"))
        scale_layout.addWidget(self.vmin_input)
        scale_layout.addWidget(self.vmax_input)
        layout.addLayout(scale_layout)

        self.setLayout(layout)
        self.stack = None

    def set_energies(self, energies):
        self.energies = energies

    def set_stack(self, image_stack):
        self.stack = image_stack
        self.slider.setMaximum(len(image_stack) - 1)
        self.slider.setValue(0)
        self.update_plot()

    def update_plot(self):
        if self.stack is not None:
            idx = self.slider.value()

            if idx < 0 or idx >= len(self.stack):
                return

            img = self.stack[idx]
            self.ax.clear()

            # Parse intensity limits
            try:
                vmin = float(self.vmin_input.text()) if self.vmin_input.text() else None
                vmax = float(self.vmax_input.text()) if self.vmax_input.text() else None
            except ValueError:
                vmin, vmax = None, None

            im = self.ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            # print(self.energies)
            # Update energy label
            if (
                hasattr(self, "energies")
                and self.energies
                and 0 <= idx < len(self.energies)
            ):
                # print(idx)
                self.energy_label.setText(f"Energy: {self.energies[idx]:.2f} eV")
            else:
                self.energy_label.setText("Energy: N/A")

            self.canvas.draw()
