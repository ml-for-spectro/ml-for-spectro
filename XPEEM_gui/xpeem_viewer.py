import sys
import os
import numpy as np
import h5py
import time
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSlider,
    QGridLayout,
    QCheckBox,
    QRubberBand,
    QSpinBox,
    QMessageBox,
)
from PySide6.QtCore import Qt, QPoint, QRect, QSize, QSettings
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import numpy as np


class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPEEM HDF5 Viewer")
        self.setGeometry(200, 200, 1200, 700)

        self.settings = QSettings("Synchrotron SOLEIL", "XPEEM Viewer")
        self.last_directory = self.settings.value("last_directory", ".")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- Top metadata label ---
        self.metadata_label = QLabel("No file loaded")
        self.image_label = QLabel(self)
        self.layout.addWidget(self.metadata_label)

        # --- TWO rows of buttons ---
        self.button_row_top = QHBoxLayout()
        self.button_row_bottom = QHBoxLayout()

        # Top row buttons
        self.load_button = QPushButton("Load HDF5 File")
        self.load_button.clicked.connect(self.load_file)
        self.button_row_top.addWidget(self.load_button)

        self.roi_button = QPushButton("Enable ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.clicked.connect(self.toggle_roi_selector)
        self.button_row_top.addWidget(self.roi_button)

        self.mean_button = QPushButton("Show Mean Spectrum")
        self.mean_button.clicked.connect(self.show_mean_spectrum)
        self.button_row_top.addWidget(self.mean_button)

        self.crop_button = QPushButton("Crop ROI")
        self.crop_button.clicked.connect(self.enable_crop_mode)
        self.button_row_top.addWidget(self.crop_button)

        self.add_plot_checkbox = QCheckBox("Add Plot")
        self.add_plot_checkbox.setChecked(False)
        self.button_row_top.addWidget(self.add_plot_checkbox)

        # Bottom row buttons
        self.save_spectrum_button = QPushButton("Save ROI Spectrum")
        self.save_spectrum_button.clicked.connect(self.save_roi_spectrum)
        self.button_row_bottom.addWidget(self.save_spectrum_button)

        self.save_axis_button = QPushButton("Save as axis hdf5")
        self.save_axis_button.clicked.connect(self.save_for_axis)
        self.save_axis_button.setEnabled(False)
        self.button_row_bottom.addWidget(self.save_axis_button)

        self.cluster_button = QPushButton("Cluster Spectral Data")
        self.cluster_button.clicked.connect(self.run_clustering)
        self.cluster_button.setEnabled(True)
        self.button_row_bottom.addWidget(self.cluster_button)

        self.cluster_spinbox = QSpinBox()
        self.cluster_spinbox.setMinimum(2)
        self.cluster_spinbox.setMaximum(20)
        self.cluster_spinbox.setValue(4)
        self.cluster_spinbox.setSuffix(" clusters")
        self.button_row_bottom.addWidget(self.cluster_spinbox)

        self.pca_button = QPushButton("Run PCA")
        self.pca_button.clicked.connect(self.run_pca)
        self.button_row_bottom.addWidget(self.pca_button)

        self.save_cluster_button = QPushButton("Save Clustered Spectra")
        self.save_cluster_button.setEnabled(False)
        self.save_cluster_button.clicked.connect(self.save_clustered_spectra)
        self.button_row_bottom.addWidget(self.save_cluster_button)

        self.remove_norm_button = QPushButton("Remove Norm image")
        self.remove_norm_button.clicked.connect(self.remove_norm)
        self.button_row_bottom.addWidget(self.remove_norm_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        self.button_row_bottom.addWidget(self.exit_button)

        # Add button rows to the main layout
        self.layout.addLayout(self.button_row_top)
        self.layout.addLayout(self.button_row_bottom)

        # File label (show current loaded file)
        self.file_label = QLabel("File: None")
        self.layout.addWidget(self.file_label)

        # --- Main plotting canvas ---
        self.canvas = FigureCanvas(Figure(figsize=(10, 5)))
        self.ax_image = self.canvas.figure.add_subplot(121)
        self.ax_spectrum = self.canvas.figure.add_subplot(122)
        self.layout.addWidget(self.canvas)

        self.spectrum_canvas = self.canvas  # reuse the same canvas

        # --- Sliders for energy, x, y slices ---
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

        # --- Other initializations ---
        self.crop_mode = False
        self.crop_rect = None
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)
        self.origin = QPoint()

        self.data = None
        self.energy = None
        self.slice_type = "energy"
        self.roi_selector = None
        self.current_color_index = 0
        self.color_cycle = [
            "red",
            "blue",
            "green",
            "magenta",
            "orange",
            "cyan",
            "black",
        ]

    def load_file(self):
        self.current_spectrum = None

        start_dir = self.last_directory if hasattr(self, "last_directory") else "."

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open HDF5 File", start_dir, "HDF5 Files (*.h5 *.hdf5 *.nxs)"
        )

        if not file_path:
            return  # User cancelled

        self.filename = file_path
        self.file_label.setText(f"File: {os.path.basename(file_path)}")
        self.last_directory = os.path.dirname(self.filename)
        self.settings.setValue("last_directory", self.last_directory)

        self.save_axis_button.setEnabled(True)

        try:
            with h5py.File(self.filename, "r") as f:
                entry = list(f.keys())[0]
                self.data = f[entry]["scan_data"]["data_02"][()]
                try:
                    self.energy = f[entry]["scan_data"]["actuator_1_1"][()]
                    self.metadata_label.setText(
                        f"File loaded: {self.filename}\n"
                        f"Shape: {self.data.shape}, "
                        f"Energy range: {self.energy.min():.2f}-{self.energy.max():.2f} eV"
                    )
                except Exception:
                    self.energy = np.arange(self.data.shape[0])

            # crop around center
            height, width = self.data.shape[1], self.data.shape[2]
            center_x, center_y = width // 2, height // 2
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            radius = 972 / 2  # Diameter / 2
            circular_mask = dist_from_center <= radius

            # Mask outside circle
            mask3d = np.broadcast_to(circular_mask, self.data.shape)
            self.data = np.where(mask3d, self.data, np.nan)  # set outside to NaN
        except Exception as e:
            self.metadata_label.setText(f"Failed to load: {e}")
            return

        self.energy_slider.setMaximum(self.data.shape[0] - 1)
        self.energy_slider.setValue(self.data.shape[0] // 2)
        self.x_slider.setMaximum(self.data.shape[1] - 1)
        self.x_slider.setValue(self.data.shape[1] // 2)
        self.y_slider.setMaximum(self.data.shape[2] - 1)
        self.y_slider.setValue(self.data.shape[2] // 2)

        if not self.add_plot_checkbox.isChecked():
            self.ax_spectrum.clear()
            self.spectrum_canvas.draw()

        self.save_cluster_button.setEnabled(False)
        self.update_image()

    def save_roi_spectrum(self):
        if self.data is None or self.energy is None:
            self.metadata_label.setText("No data to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Spectrum", self.last_directory, "Text Files (*.txt)"
        )
        if file_path:
            self.last_directory = os.path.dirname(file_path)
            with open(file_path, "w") as f:
                f.write("# Energy (eV), Intensity\n")
                for e, i in zip(self.energy, self.current_spectrum):
                    f.write(f"{e:.2f}, {i:.6f}\n")
            self.metadata_label.setText(f"ROI spectrum saved to {file_path}")

    def update_image(self):
        idx = self.energy_slider.value()
        if self.data is not None:
            self.slice_type = "energy"
            self.ax_image.clear()
            self.ax_image.imshow(self.data[idx, :, :], cmap="inferno", origin="lower")
            self.ax_image.set_title(f"Energy slice #{idx} at {self.energy[idx]:.2f} eV")
            self.canvas.draw()

    def update_x_slice(self):
        idx = self.x_slider.value()
        if self.data is not None:
            self.slice_type = "x"
            self.ax_image.clear()
            self.ax_image.imshow(self.data[:, idx, :], cmap="viridis", origin="lower")
            self.ax_image.set_title(f"X cross-section @ x={idx}")
            self.canvas.draw()

    def update_y_slice(self):
        idx = self.y_slider.value()
        if self.data is not None:
            self.slice_type = "y"
            self.ax_image.clear()
            self.ax_image.imshow(self.data[:, :, idx], cmap="cividis", origin="lower")
            self.ax_image.set_title(f"Y cross-section @ y={idx}")
            self.canvas.draw()

    def toggle_roi_selector(self):
        if self.roi_selector:
            self.roi_selector.set_active(False)
            self.roi_selector = None
            return

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            self.plot_roi_spectrum(xmin, xmax, ymin, ymax)

        self.roi_selector = RectangleSelector(
            self.ax_image,
            onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.canvas.draw()

    def enable_crop_mode(self):
        self.crop_mode = True
        self.metadata_label.setText("Draw a rectangle to crop the datacube.")

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            self.crop_rect = (xmin, xmax, ymin, ymax)
            self.crop_mode = False
            self.apply_crop()

        if self.roi_selector:
            self.roi_selector.set_active(False)

        self.roi_selector = RectangleSelector(
            self.ax_image,
            onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.canvas.draw()

    def plot_roi_spectrum(self, xmin, xmax, ymin, ymax):
        if self.data is None:
            return

        idx = self.energy_slider.value()
        if self.slice_type == "energy":
            roi_data = self.data[:, ymin:ymax, xmin:xmax]
        elif self.slice_type == "x":
            roi_data = self.data[ymin:ymax, :, xmin:xmax]
        elif self.slice_type == "y":
            roi_data = self.data[ymin:ymax, xmin:xmax, :]
        else:
            return

        if not self.add_plot_checkbox.isChecked():
            self.ax_spectrum.clear()

        color = self.color_cycle[self.current_color_index % len(self.color_cycle)]
        self.current_color_index += 1

        spectrum = roi_data.mean(axis=(1, 2))

        self.current_spectrum = spectrum
        self.ax_spectrum.plot(
            self.energy, spectrum, label=f"ROI {self.current_color_index}", color=color
        )
        self.ax_spectrum.set_title("Spectrum from ROI")
        self.ax_spectrum.set_xlabel("Energy (eV)")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw()

    def show_mean_spectrum(self):
        if self.data is None or self.energy is None:
            self.metadata_label.setText("No data loaded.")
            return

        # mean_spectrum = self.data.mean(axis=(1, 2))
        mean_spectrum = self.compute_mean_spectrum(self.data)
        self.current_spectrum = mean_spectrum

        if not self.add_plot_checkbox.isChecked():
            self.ax_spectrum.clear()

        color = self.color_cycle[self.current_color_index % len(self.color_cycle)]
        self.current_color_index += 1

        self.ax_spectrum.plot(
            self.energy,
            mean_spectrum,
            label=f"Mean {self.current_color_index}",
            color=color,
        )
        self.ax_spectrum.set_title("Mean Spectrum (All Pixels)")
        self.ax_spectrum.set_xlabel("Energy (eV)")
        self.ax_spectrum.set_ylabel("Intensity")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw()
        self.metadata_label.setText("Displayed mean spectrum over full image.")

    def get_circular_mask(self, shape, diameter):
        """Returns a circular mask centered in the image with the given diameter."""
        h, w = shape
        center_y, center_x = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        radius = diameter / 2
        mask = dist_from_center <= radius
        return mask

    def compute_mean_spectrum(self, stack):
        """
        stack: 3D numpy array with shape (frames, height, width)
        Returns mean spectrum from circular region (800 px diameter).
        """
        mean_image = np.mean(stack, axis=0)
        mask = self.get_circular_mask(mean_image.shape, diameter=972)
        masked_pixels = stack[:, mask]
        mean_spectrum = np.mean(masked_pixels, axis=1)
        return mean_spectrum

    def mousePressEvent(self, event):
        if (
            self.crop_mode
            and event.button() == Qt.LeftButton
            and self.image_label.underMouse()
        ):
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if self.crop_mode and self.rubber_band.isVisible():
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event):
        if self.crop_mode and self.rubber_band.isVisible():
            self.crop_mode = False
            self.rubber_band.hide()
            rect = self.rubber_band.geometry()
            self.crop_rect = rect
            self.apply_crop()

    def save_for_axis(self):

        if self.filename:

            nxsFile = h5py.File(self.filename, "r")
            theEntry = list(nxsFile.keys())[0]
            if self.data is None:
                theData = nxsFile[theEntry]["scan_data"]["data_02"][()]
            else:
                theData = self.data

            try:
                energy = nxsFile[theEntry]["scan_data"]["actuator_1_1"][()]
            except:
                energy = np.zeros(len(theData))

            sampleX = np.linspace(0, np.shape(theData)[1] - 1, np.shape(theData)[1])
            sampleY = np.linspace(0, np.shape(theData)[2] - 1, np.shape(theData)[2])

            # hdfOutFileName = nxsFileName.split(".")[0] + "_axis.hdf5"
            savefilename, _ = QFileDialog.getSaveFileName(
                self, "Save axis readable HDF5 File", "", "HDF5 Files (*.hdf5)"
            )
            if savefilename:
                with h5py.File(savefilename, "w") as NXfout:
                    NXfout.attrs["HDF5_Version"] = np.array([b"1.8.4"])
                    NXfout.attrs["NeXus_version"] = np.array([b"4.3.0"])
                    NXfout.attrs["file_name"] = np.array([savefilename.encode("utf-8")])
                    NXtime = time.strftime("%Y-%m-%dT%H:%M:%S+01:00")
                    NXfout.attrs["file_time"] = np.array([NXtime.encode("utf-8")])
                    NXfout.create_group(b"entry1")
                    NXfout["entry1"].create_dataset(
                        "start_time", data=np.array([NXtime.encode("utf-8")])
                    )
                    NXfout["entry1"].create_dataset(
                        "end_time", data=np.array([NXtime.encode("utf-8")])
                    )
                    NXfout["entry1"].attrs["NX_class"] = b"NXentry"
                    NXfout["entry1"].create_dataset(
                        "definition", data=np.array([b"NXstxm"])
                    )
                    NXfout["entry1"]["definition"].attrs["version"] = np.array(b"1.1")
                    NXfout["entry1"].create_group(b"xpeem")
                    NXfout["entry1"]["xpeem"].attrs["NX_class"] = b"NXdata"
                    NXfout["entry1"]["xpeem"].create_dataset(
                        b"data", np.shape(theData), dtype="uint16", data=theData
                    )
                    NXfout["entry1"]["xpeem"].attrs["signal"] = "data"
                    NXfout["entry1"]["xpeem"].attrs["axes"] = [
                        "energy",
                        "sample_x",
                        "sample_y",
                    ]
                    NXfout["entry1"]["xpeem"].attrs["sample_y_indices"] = np.array(
                        [0], dtype="uint32"
                    )
                    NXfout["entry1"]["xpeem"].create_dataset(
                        b"sample_y",
                        (np.shape(theData)[2],),
                        dtype="float",
                        data=sampleY,
                    )
                    NXfout["entry1"]["xpeem"].attrs["sample_x_indices"] = np.array(
                        [1], dtype="uint32"
                    )
                    NXfout["entry1"]["xpeem"].create_dataset(
                        b"sample_x",
                        (np.shape(theData)[1],),
                        dtype="float",
                        data=sampleX,
                    )
                    NXfout["entry1"]["xpeem"].create_dataset(b"energy", data=energy)
                    NXfout["entry1"]["xpeem"].create_dataset(
                        "stxm_scan_type", data=np.array([b"sample image"])
                    )
                    NXfout["entry1"]["xpeem"]["sample_x"].attrs["axis"] = 2
                    NXfout["entry1"]["xpeem"]["sample_y"].attrs["axis"] = 1
                    NXfout["entry1"]["xpeem"]["energy"].attrs["units"] = "eV"
                    NXfout["entry1"]["xpeem"].create_dataset(
                        b"count_time", data=np.linspace(0.1, 0.1, np.shape(theData)[0])
                    )
                    NXfout["entry1"].create_dataset("title", data=[b"XPEEM data"])

    def apply_crop(self):
        if self.crop_rect and self.data is not None:
            xmin, xmax, ymin, ymax = self.crop_rect
            if self.slice_type == "energy":
                self.data = self.data[:, ymin:ymax, xmin:xmax]
            elif self.slice_type == "x":
                self.data = self.data[ymin:ymax, :, xmin:xmax]
            elif self.slice_type == "y":
                self.data = self.data[ymin:ymax, xmin:xmax, :]

            # Update sliders and image
            self.energy_slider.setMaximum(self.data.shape[0] - 1)
            self.x_slider.setMaximum(self.data.shape[1] - 1)
            self.y_slider.setMaximum(self.data.shape[2] - 1)
            self.update_image()
            self.metadata_label.setText("Data cropped to selected ROI.")

    def remove_norm(self):
        from PIL import Image  # Only import when needed

        tiff_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF Image", self.last_directory, "TIFF Files (*.tif *.tiff)"
        )
        if not tiff_path:
            return

        try:
            tiff_image = Image.open(tiff_path)
            tiff_array = np.array(tiff_image)

            # Ensure the TIFF image matches the spatial dimensions
            if self.data is None:
                self.metadata_label.setText("Load an HDF5 file first.")
                return

            if tiff_array.shape != self.data.shape[1:]:
                self.metadata_label.setText(
                    f"TIFF shape {tiff_array.shape} does not match data shape {self.data.shape[1:]}."
                )
                return

            # Divide the TIFF image from each energy slice
            self.data = self.data / tiff_array[np.newaxis, :, :]
            self.metadata_label.setText(f"Divided TIFF from all slices.")
            self.update_image()  # Refresh image view

        except Exception as e:
            self.metadata_label.setText(f"Error loading TIFF: {str(e)}")

    def cluster_cube(self, data_cube, n_clusters=4):
        E, Y, X = data_cube.shape
        flat_data = data_cube.reshape(E, -1).T  # shape: (Y*X, E)

        # Find valid (non-NaN) pixels
        valid_mask = ~np.isnan(flat_data).any(axis=1)
        flat_data_valid = flat_data[valid_mask]

        # Normalize (zero mean, unit variance) to focus on shape
        flat_data_valid = flat_data_valid.astype(np.float32)
        flat_data_norm = (
            flat_data_valid - flat_data_valid.mean(axis=1, keepdims=True)
        ) / (flat_data_valid.std(axis=1, keepdims=True) + 1e-8)

        # UMAP dimensionality reduction
        scaled = StandardScaler().fit_transform(flat_data_norm)
        # embedding = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(scaled)
        embedding = umap.UMAP(n_neighbors=30, min_dist=0.1).fit_transform(scaled)
        # KMeans clustering
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embedding)

        # Create a full label image, initialized with -1
        label_img = np.full((Y * X,), -1, dtype=np.int32)
        label_img[valid_mask] = labels
        label_img = label_img.reshape(Y, X)

        # Average spectra per cluster (using the original, non-normalized spectra)
        mean_spectra = []
        for i in range(n_clusters):
            cluster_indices = labels == i
            if np.any(cluster_indices):
                mean_spectra.append(flat_data_valid[cluster_indices].mean(axis=0))
            else:
                mean_spectra.append(np.zeros(E))
        mean_spectra = np.array(mean_spectra)

        return label_img, mean_spectra

    def run_clustering(self):
        self.data_cube = self.data
        if self.data_cube is None:
            return

        n_clusters = self.cluster_spinbox.value()  # <-- get user-selected number
        label_img, mean_spectra = self.cluster_cube(self.data, n_clusters=n_clusters)
        self.save_cluster_button.setEnabled(True)
        self.pca_button.setEnabled(False)

        self.label_img = label_img  # save for later use if needed
        self.cluster_results = (
            mean_spectra  # save results so save_clustered_spectra can access
        )

        # Show cluster map
        self.ax_image.clear()
        img = self.ax_image.imshow(
            np.ma.masked_where(label_img == -1, label_img),
            cmap="tab10",
            interpolation="none",
        )  # no smoothing
        self.ax_image.set_title(f"Cluster Map ({n_clusters} clusters)")
        self.canvas.draw_idle()  # <--- use draw_idle instead of draw
        self.canvas.flush_events()

        # Show mean spectra
        self.ax_spectrum.clear()
        for i, spectrum in enumerate(mean_spectra):
            self.ax_spectrum.plot(spectrum, label=f"Cluster {i}")
        self.ax_spectrum.set_title("Average Spectra")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw()

    def save_clustered_spectra(self):
        if (
            hasattr(self, "cluster_results")
            and self.cluster_results is not None
            and np.size(self.cluster_results) > 0
        ):
            spectra_to_save = self.cluster_results
            spectra_type = "Clustered Spectra"
        elif (
            hasattr(self, "pca_components")
            and self.pca_components is not None
            and np.size(self.pca_components) > 0
        ):
            spectra_to_save = self.pca_components
            spectra_type = "PCA Components"
        else:
            self.metadata_label.setText("No spectra (cluster or PCA) to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {spectra_type}", self.last_directory, "Text Files (*.txt)"
        )
        if file_path:
            self.last_directory = os.path.dirname(file_path)
            with open(file_path, "w") as f:
                f.write(
                    f"# Energy (eV), {spectra_type.replace(' ', '_')}_1, {spectra_type.replace(' ', '_')}_2, ...\n"
                )
                for i in range(len(self.energy)):
                    line = f"{self.energy[i]:.4f}"
                    for spectrum in spectra_to_save:
                        line += f", {spectrum[i]:.6f}"
                    f.write(line + "\n")
            self.metadata_label.setText(f"{spectra_type} saved to {file_path}")

        self.pca_button.setEnabled(True)
        self.save_cluster_button.setEnabled(False)
        self.cluster_button.setEnabled(True)

    def run_pca(self):
        if self.data is None:
            QMessageBox.warning(self, "No data loaded", "Please load a file first.")
            return

        from sklearn.decomposition import PCA

        stack = self.data
        n_images, height, width = stack.shape
        reshaped_O = stack.reshape(n_images, -1)  # shape (energy, pixels)

        # Select only valid pixels (no NaNs across energy)
        valid_mask = ~np.isnan(reshaped_O).any(axis=0)
        reshaped = reshaped_O[:, valid_mask]  # Now shape (energy, valid_pixels)

        # Normalize each pixel spectrum (divide by its norm)
        norms = np.linalg.norm(reshaped, axis=0)
        norms[norms == 0] = 1  # avoid division by zero
        reshaped_normalized = reshaped / norms

        # Number of components
        n_components = self.cluster_spinbox.value()

        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(
            reshaped_normalized.T
        )  # shape (pixels, components)

        # Rebuild PC images
        pc_images = np.full((n_components, height * width), np.nan)
        pc_images[:, valid_mask] = pca_result.T
        pc_images = pc_images.reshape(n_components, height, width)
        self.pc_images = pc_images
        self.save_cluster_button.setEnabled(True)
        self.cluster_button.setEnabled(False)
        # Show first principal component as image
        self.ax_image.clear()
        self.ax_image.imshow(self.pc_images[0], cmap="viridis")
        self.ax_image.set_title("PCA 1st Component")
        self.canvas.draw_idle()

        # Show component spectra ("PCA loadings") on ax_spectrum
        self.ax_spectrum.clear()
        for i in range(n_components):
            self.ax_spectrum.plot(pca.components_[i], label=f"PC {i+1}")
        self.ax_spectrum.set_title("PCA Component Spectra")
        self.ax_spectrum.legend()
        self.spectrum_canvas.draw_idle()
        self.pca_components = pca.components_


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = HDF5Viewer()
        window.show()
    except Exception as e:
        print(f"Error: {e}")
    sys.exit(app.exec())
