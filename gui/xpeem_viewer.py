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
)
from PySide6.QtCore import Qt, QPoint, QRect, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector


class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPEEM HDF5 Viewer")
        self.setGeometry(200, 200, 1200, 700)

        self.last_directory = ""  # Store last used directory

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.metadata_label = QLabel("No file loaded")
        self.image_label = QLabel(self)
        self.layout.addWidget(self.metadata_label)

        self.button_row = QHBoxLayout()
        self.load_button = QPushButton("Load HDF5 File")
        self.load_button.clicked.connect(self.load_file)
        self.button_row.addWidget(self.load_button)

        self.roi_button = QPushButton("Enable ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.clicked.connect(self.toggle_roi_selector)
        self.button_row.addWidget(self.roi_button)

        self.mean_button = QPushButton("Show Mean Spectrum")
        self.mean_button.clicked.connect(self.show_mean_spectrum)
        self.button_row.addWidget(self.mean_button)

        self.crop_button = QPushButton("Crop ROI")
        self.crop_button.clicked.connect(self.enable_crop_mode)
        self.button_row.addWidget(self.crop_button)

        self.add_plot_checkbox = QCheckBox("Add Plot")
        self.add_plot_checkbox.setChecked(False)
        self.button_row.addWidget(self.add_plot_checkbox)

        self.save_spectrum_button = QPushButton("Save ROI Spectrum")
        self.save_spectrum_button.clicked.connect(self.save_roi_spectrum)
        self.button_row.addWidget(self.save_spectrum_button)

        self.save_axis_button = QPushButton("Save as axis hdf5")
        self.save_axis_button.clicked.connect(self.save_for_axis)
        self.button_row.addWidget(self.save_axis_button)
        self.save_axis_button.setEnabled(False)

        self.remove_norm_button = QPushButton("Remove Norm image")
        self.remove_norm_button.clicked.connect(self.remove_norm)
        self.button_row.addWidget(self.remove_norm_button)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        self.button_row.addWidget(self.exit_button)

        self.layout.addLayout(self.button_row)

        self.canvas = FigureCanvas(Figure(figsize=(10, 5)))
        self.ax_image = self.canvas.figure.add_subplot(121)
        self.ax_spectrum = self.canvas.figure.add_subplot(122)
        self.layout.addWidget(self.canvas)

        self.spectrum_canvas = self.canvas  # reuse the same canvas

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
        self.filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 File",
            self.last_directory,
            "HDF5 Files (*.h5 *.hdf5 *.nxs)",
        )
        if self.filename:
            self.last_directory = os.path.dirname(self.filename)
            self.save_axis_button.setEnabled(True)
            with h5py.File(self.filename, "r") as f:
                try:
                    entry = list(f.keys())[0]
                    self.data = f[entry]["scan_data"]["data_02"][()]
                    self.energy = f[entry]["scan_data"]["actuator_1_1"][()]
                    self.metadata_label.setText(
                        f"File loaded: {self.filename}\nShape: {self.data.shape}, Energy range: {self.energy.min():.2f}-{self.energy.max():.2f} eV"
                    )
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

            self.update_image()

    def save_roi_spectrum(self):
        if self.data is None or self.energy is None:
            self.metadata_label.setText("No data to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Spectrum", self.last_directory, "Text Files (*.txt)"
        )
        if file_path:
            self.last_directory = os.path.dirname(self.filename)
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

        mean_spectrum = self.data.mean(axis=(1, 2))
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

            # Subtract the TIFF image from each energy slice
            self.data = self.data - tiff_array[np.newaxis, :, :]
            self.metadata_label.setText(f"Subtracted TIFF from all slices.")
            self.update_image()  # Refresh image view

        except Exception as e:
            self.metadata_label.setText(f"Error loading TIFF: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec())
