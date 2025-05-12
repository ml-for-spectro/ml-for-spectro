import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QSpinBox,
    QGroupBox,
    QFormLayout,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FourierTab(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

        self.original_image = None
        self.filtered_image = None
        self.mask_params = [20, 20]  # [mask_width, mask_height]

        layout = QHBoxLayout()
        self.setLayout(layout)

        # Left side: controls
        control_panel = QVBoxLayout()
        layout.addLayout(control_panel, 1)

        self.init_controls(control_panel)

        # Right side: plots
        plot_panel = QVBoxLayout()
        layout.addLayout(plot_panel, 2)

        self.fig = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.fig)
        self.ax_orig = self.fig.add_subplot(1, 2, 1)
        self.ax_filt = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout()

        plot_panel.addWidget(self.canvas)

    def init_controls(self, layout):
        group = QGroupBox("Fourier Filter Controls")
        form = QFormLayout()

        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 200)
        self.slider_width.setValue(self.mask_params[0])
        self.slider_width.valueChanged.connect(self.update_mask_param)

        self.slider_height = QSlider(Qt.Horizontal)
        self.slider_height.setRange(1, 200)
        self.slider_height.setValue(self.mask_params[1])
        self.slider_height.valueChanged.connect(self.update_mask_param)

        form.addRow("Mask Width", self.slider_width)
        form.addRow("Mask Height", self.slider_height)

        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.clicked.connect(self.apply_filter)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_filter)

        form.addRow(self.apply_btn)
        form.addRow(self.undo_btn)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

    def update_mask_param(self):
        self.mask_params = [self.slider_width.value(), self.slider_height.value()]
        if self.original_image is not None:
            self.apply_filter()

    def load_image(self, image):
        """Call this when switching tabs or loading new data"""
        self.original_image = image
        self.filtered_image = None
        self.update_plot()

    def apply_filter(self):
        if self.original_image is None:
            return
        img = self.original_image
        f_img = np.fft.fftshift(np.fft.fft2(img))
        cx, cy = f_img.shape[1] // 2, f_img.shape[0] // 2
        mx, my = self.mask_params[0] // 2, self.mask_params[1] // 2

        mask = np.ones_like(f_img, dtype=bool)
        mask[cy - my : cy + my, cx - mx : cx + mx] = False
        f_img_filtered = f_img * mask

        self.filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(f_img_filtered)))
        self.viewer.image_current[0] = self.filtered_image.copy()
        self.viewer.set_stack(self.viewer.image_current)
        self.update_plot()

    def undo_filter(self):
        if self.original_image is not None:
            self.viewer.image_current[0] = self.original_image.copy()
            self.viewer.set_stack(self.viewer.image_current)
            self.filtered_image = None
            self.update_plot()

    def update_plot(self):
        self.ax_orig.clear()
        self.ax_filt.clear()

        if self.original_image is not None:
            self.ax_orig.imshow(self.original_image, cmap="gray")
            self.ax_orig.set_title("Original")

        if self.filtered_image is not None:
            self.ax_filt.imshow(self.filtered_image, cmap="gray")
            self.ax_filt.set_title("Filtered")
        else:
            self.ax_filt.set_title("Filtered (none)")

        self.ax_orig.axis("off")
        self.ax_filt.axis("off")
        self.canvas.draw()
