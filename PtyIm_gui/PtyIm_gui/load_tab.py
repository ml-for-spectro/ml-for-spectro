import os
import csv
import json
import numpy as np
import h5py
import time

from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
)
from PySide6.QtCore import Qt

from utils import (
    SETTINGS_FILE,
    load_text_image,
    normalize_image,
    rescale_image,
    crop_center_stack,
    register_images,
    load_settings,
    save_settings,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LoadTab(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        load_settings(self)

        self.setup_ui(main_layout)
        # self.init_preview_canvas(main_layout)

        self.images = []
        self.filenames = []
        self.pixel_inputs = []
        self.energy_inputs = []
        self.order_inputs = []

        # self.setLayout(main_layout)
        # self.init_preview_canvas(main_layout)

    def setup_ui(self, layout):
        # Create horizontal layout to split into left (controls) and right (preview)
        split_layout = QHBoxLayout()
        layout.addLayout(split_layout)

        # === LEFT: Control Panel ===
        control_layout = QVBoxLayout()
        split_layout.addLayout(control_layout, 1)  # 1 = stretch factor

        # === GROUP 1: Load / Save Batch ===
        batch_group = QGroupBox("Image File Batch")
        batch_layout = QGridLayout()
        self.load_btn = QPushButton("Load Images")
        self.load_btn.clicked.connect(self.load_images)
        self.load_batch_btn = QPushButton("Load Batch")
        self.load_batch_btn.clicked.connect(self.load_batch)
        self.save_btn = QPushButton("Save Batch")
        self.save_btn.clicked.connect(self.save_batch)
        batch_layout.addWidget(self.load_btn, 0, 0, 1, 2)
        batch_layout.addWidget(self.load_batch_btn, 1, 0)
        batch_layout.addWidget(self.save_btn, 1, 1)
        batch_group.setLayout(batch_layout)

        # === GROUP 2: Pixel Correction ===
        pixel_group = QGroupBox("Pixel Size Correction")
        pixel_layout = QHBoxLayout()
        self.correct_btn = QPushButton("Correct Pixel Sizes")
        self.correct_btn.clicked.connect(self.correct_pixel_sizes)
        pixel_layout.addWidget(self.correct_btn)
        pixel_group.setLayout(pixel_layout)

        # === GROUP 3: Cropping ===
        crop_group = QGroupBox("Crop to Center Region")
        crop_layout = QHBoxLayout()
        self.crop_btn = QPushButton("Crop Images")
        self.crop_btn.clicked.connect(self.crop_images)
        crop_layout.addWidget(self.crop_btn)
        crop_group.setLayout(crop_layout)

        # === GROUP 4: Registration ===
        reg_group = QGroupBox("Register Image Stack")
        reg_layout = QGridLayout()
        self.register_btn = QPushButton("Register Images")
        self.register_btn.clicked.connect(self.register_images)
        self.save_reg_btn = QPushButton("Save Registration")
        self.save_reg_btn.clicked.connect(self.save_registration)
        self.load_reg_btn = QPushButton("Apply Registration")
        self.load_reg_btn.clicked.connect(self.apply_registration)
        reg_layout.addWidget(self.register_btn, 0, 0)
        reg_layout.addWidget(self.save_reg_btn, 0, 1)
        reg_layout.addWidget(self.load_reg_btn, 1, 0, 1, 2)
        reg_group.setLayout(reg_layout)

        # === GROUP 5: Save to HDF5 ===
        hdf5_group = QGroupBox("Export")
        hdf5_layout = QHBoxLayout()
        self.save_hdf5_btn = QPushButton("Save HDF5 for Axis2000")
        self.save_hdf5_btn.clicked.connect(self.save_hdf5_for_axis2000)
        hdf5_layout.addWidget(self.save_hdf5_btn)
        hdf5_group.setLayout(hdf5_layout)

        # === GROUP 6: Exit Button ===
        exit_group = QGroupBox()
        exit_layout = QHBoxLayout()
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close_application)
        exit_layout.addStretch()
        exit_layout.addWidget(self.exit_btn)
        exit_layout.addStretch()
        exit_group.setLayout(exit_layout)

        # === Add all groups to control layout ===
        for group in [
            batch_group,
            pixel_group,
            crop_group,
            reg_group,
            hdf5_group,
            exit_group,
        ]:
            control_layout.addWidget(group)

        # === Input grid (for pixel/energy/order fields) ===
        self.input_grid = QGridLayout()
        control_layout.addLayout(self.input_grid)

        # === RIGHT: Preview plot ===
        preview_layout = QVBoxLayout()
        split_layout.addLayout(preview_layout, 2)  # 2 = wider than left

        self.preview_fig = Figure(figsize=(5, 5))
        self.preview_canvas = FigureCanvas(self.preview_fig)
        self.preview_ax = self.preview_fig.add_subplot(111)
        preview_layout.addWidget(QLabel("Preview (First Image):"))
        preview_layout.addWidget(self.preview_canvas)

    def init_preview_canvas(self, layout):
        self.preview_fig = Figure(figsize=(3, 3))
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_canvas = FigureCanvas(self.preview_fig)
        layout.addWidget(QLabel("Preview (First Image):"))
        layout.addWidget(self.preview_canvas)

    def update_preview(self):
        if hasattr(self, "image_current") and len(self.image_current) > 0:
            self.preview_ax.clear()
            self.preview_ax.imshow(self.image_current[0], cmap="gray")
            self.preview_ax.set_title("First Image Preview")
            self.preview_ax.axis("off")
            self.preview_canvas.draw()

    def load_images(self):
        count, ok = QInputDialog.getInt(
            self, "Load Images", "Number of images:", self.last_file_count, 1, 100
        )
        if not ok:
            return

        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", self.previous_dir, "Text Files (*.txt)"
        )
        if len(filenames) != count:
            QMessageBox.warning(
                self,
                "Error",
                f"You selected {len(filenames)} files instead of {count}.",
            )
            return

        self.previous_dir = os.path.dirname(filenames[0])
        self.last_file_count = count
        save_settings(self)

        self.filenames = filenames
        # self.images = [normalize_image(load_text_image(f)) for f in filenames]
        self.images = [load_text_image(f) for f in self.filenames]
        # Clear previous widgets
        for i in reversed(range(self.input_grid.count())):
            widget = self.input_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.pixel_inputs = []
        self.energy_inputs = []
        self.order_inputs = []

        for i, f in enumerate(filenames):
            label = QLabel(os.path.basename(f))
            pixel_box = QLineEdit()
            energy_box = QLineEdit()
            order_box = QSpinBox()
            order_box.setRange(1, count)
            order_box.setValue(i + 1)

            self.input_grid.addWidget(label, i, 0)
            self.input_grid.addWidget(pixel_box, i, 1)
            self.input_grid.addWidget(energy_box, i, 2)
            self.input_grid.addWidget(order_box, i, 3)

            self.pixel_inputs.append(pixel_box)
            self.energy_inputs.append(energy_box)
            self.order_inputs.append(order_box)

        self.image_stack = np.stack(cropped)
        # Sort images and metadata by order
        orders = [sb.value() for sb in self.order_inputs]
        sorted_indices = np.argsort(orders)

        self.images = [self.images[i] for i in sorted_indices]
        self.filenames = [self.filenames[i] for i in sorted_indices]
        self.pixel_inputs = [self.pixel_inputs[i] for i in sorted_indices]
        self.energy_inputs = [self.energy_inputs[i] for i in sorted_indices]
        self.order_inputs = [self.order_inputs[i] for i in sorted_indices]

        energies = [float(e.text()) for e in self.energy_inputs]
        self.viewer.set_energies(energies)
        # Update image stack
        self.image_stack = self.images
        self.image_current = self.image_stack.copy()
        self.viewer.set_stack(self.image_current)
        self.update_preview()
        # print("Here in Load_image")
        # print(self.image_current)

    def correct_pixel_sizes(self):
        try:
            pixel_sizes = [float(inp.text()) for inp in self.pixel_inputs]
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid pixel sizes.")
            return

        ref_idx, ok = QInputDialog.getInt(
            self,
            "Reference Image",
            "Select reference image index (1-based):",
            1,
            1,
            len(self.images),
        )
        if not ok:
            return

        ref_idx -= 1
        ref_pixel = pixel_sizes[ref_idx]
        rescaled = [
            rescale_image(img, ps, ref_pixel)
            for img, ps in zip(self.images, pixel_sizes)
        ]
        min_shape = np.min([img.shape for img in rescaled], axis=0)
        cropped = [img[: min_shape[0], : min_shape[1]] for img in rescaled]
        self.image_stack = np.stack(cropped)
        self.image_current = self.image_stack.copy()
        self.viewer.set_stack(self.image_current)
        # print("Here in crop_image")
        # print(self.image_current)
        self.update_preview()

    def crop_images(self):
        if not hasattr(self, "image_stack"):
            QMessageBox.warning(self, "Error", "Correct pixel sizes before cropping.")
            return
        crop_size, ok = QInputDialog.getInt(
            self, "Crop Size", "Enter square crop size (pixels):", 200, 50, 1000
        )
        if not ok:
            return
        self.image_stack = crop_center_stack(self.image_stack, crop_size)
        self.image_current = self.image_stack.copy()
        self.viewer.set_stack(self.image_current)
        self.update_preview()

    def register_images(self):
        if not hasattr(self, "image_stack"):
            QMessageBox.warning(
                self, "Error", "Correct and crop images before registration."
            )
            return
        self.image_stack = register_images(self.image_stack)
        self.image_current = self.image_stack.copy()
        self.viewer.set_stack(self.image_current)
        # print("Here in register_image")
        # print(self.image_current)

    def save_batch(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch File", self.previous_dir, "CSV Files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "PixelSize", "Energy", "Order"])
            for i in range(len(self.filenames)):
                writer.writerow(
                    [
                        self.filenames[i],
                        self.pixel_inputs[i].text(),
                        self.energy_inputs[i].text(),
                        self.order_inputs[i].value(),
                    ]
                )

    def load_batch(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Batch File", self.previous_dir, "CSV Files (*.csv)"
        )
        if not path:
            return

        with open(path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        self.filenames = [row["Filename"] for row in rows]
        # self.images = [normalize_image(load_text_image(f)) for f in self.filenames]
        self.images = [load_text_image(f) for f in self.filenames]
        self.previous_dir = os.path.dirname(self.filenames[0])
        self.last_file_count = len(self.filenames)
        save_settings(self)

        for i in reversed(range(self.input_grid.count())):
            widget = self.input_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.pixel_inputs = []
        self.energy_inputs = []
        self.order_inputs = []

        for i, row in enumerate(rows):
            label = QLabel(os.path.basename(row["Filename"]))
            pixel_box = QLineEdit()
            pixel_box.setText(row["PixelSize"])
            energy_box = QLineEdit()
            energy_box.setText(row["Energy"])
            order_box = QSpinBox()
            order_box.setRange(1, len(rows))
            order_box.setValue(int(row.get("Order", i + 1)))

            self.input_grid.addWidget(label, i, 0)
            self.input_grid.addWidget(pixel_box, i, 1)
            self.input_grid.addWidget(energy_box, i, 2)
            self.input_grid.addWidget(order_box, i, 3)

            self.pixel_inputs.append(pixel_box)
            self.energy_inputs.append(energy_box)
            self.order_inputs.append(order_box)

        # Sort images and metadata by order
        orders = [sb.value() for sb in self.order_inputs]
        sorted_indices = np.argsort(orders)

        self.images = [self.images[i] for i in sorted_indices]
        self.filenames = [self.filenames[i] for i in sorted_indices]
        self.pixel_inputs = [self.pixel_inputs[i] for i in sorted_indices]
        self.energy_inputs = [self.energy_inputs[i] for i in sorted_indices]
        self.order_inputs = [self.order_inputs[i] for i in sorted_indices]

        energies = [float(e.text()) for e in self.energy_inputs]
        self.viewer.set_energies(energies)
        # Update image stack
        self.image_stack = self.images
        self.image_current = self.image_stack.copy()
        self.viewer.set_stack(self.image_current)
        self.update_preview()

    def save_registration(self):
        if not hasattr(self, "image_stack"):
            QMessageBox.warning(self, "Error", "No registered image stack to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Registration", self.previous_dir, "NumPy Files (*.npy)"
        )
        if not path:
            return
        np.save(path, self.image_current)

    def apply_registration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Registration", self.previous_dir, "NumPy Files (*.npy)"
        )
        if not path:
            return
        reg_stack = np.load(path)
        if hasattr(self, "image_stack") and self.image_stack.shape == reg_stack.shape:
            self.image_stack = reg_stack
            self.image_current = self.image_stack.copy()
            self.viewer.set_stack(self.image_current)
        else:
            QMessageBox.warning(self, "Error", "Registration stack shape mismatch.")

        self.update_preview()

    def save_hdf5_for_axis2000(self):
        # print("entered save hdf5")
        # print(self.image_current)
        if not hasattr(self, "image_stack") or not hasattr(self, "energy_inputs"):
            QMessageBox.warning(self, "Error", "No stack or energy values to save.")
            return

        # Ask user for file path
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5 File", "", "HDF5 Files (*.hdf5)"
        )
        if not filepath:
            return

        # Convert energy inputs to float array
        try:
            energies = np.array([float(e.text()) for e in self.energy_inputs])
        except Exception:
            QMessageBox.critical(self, "Error", "Failed to parse energy values.")
            return
        sorted_indices = np.argsort(energies)
        energy = np.array(energies)[sorted_indices]
        theData = self.image_current[sorted_indices]

        # theData = sorted_stack # shape: (N, Y, X)
        sampleX = np.arange(theData.shape[2])
        sampleY = np.arange(theData.shape[1])
        # print("here in save hdf5")
        # print(theData)
        # Save to HDF5
        try:
            # === WRITE HDF5 ===
            NXtime = time.strftime("%Y-%m-%dT%H:%M:%S+01:00")

            with h5py.File(filepath, "w") as NXfout:
                NXfout.attrs["HDF5_Version"] = [b"1.8.4"]
                NXfout.attrs["NeXus_version"] = [b"4.3.0"]
                NXfout.attrs["file_name"] = [filepath.encode("utf-8")]
                NXfout.attrs["file_time"] = [NXtime.encode("utf-8")]

                entry = NXfout.create_group("entry1")
                entry.attrs["NX_class"] = b"NXentry"
                entry.create_dataset("start_time", data=[NXtime.encode("utf-8")])
                entry.create_dataset("end_time", data=[NXtime.encode("utf-8")])
                entry.create_dataset("definition", data=[b"NXstxm"])
                entry["definition"].attrs["version"] = b"1.1"

                xpeem = entry.create_group("Ptychorecon")
                xpeem.attrs["NX_class"] = b"NXdata"
                xpeem.create_dataset("data", data=theData)
                xpeem.attrs["signal"] = "data"
                xpeem.attrs["axes"] = ["energy", "sample_x", "sample_y"]
                xpeem.attrs["sample_y_indices"] = np.array([0], dtype="uint32")
                xpeem.attrs["sample_x_indices"] = np.array([1], dtype="uint32")

                xpeem.create_dataset("sample_y", data=sampleY)
                xpeem["sample_y"].attrs["axis"] = 1
                xpeem.create_dataset("sample_x", data=sampleX)
                xpeem["sample_x"].attrs["axis"] = 2
                xpeem.create_dataset("energy", data=energy)
                xpeem["energy"].attrs["units"] = "eV"
                xpeem.create_dataset("count_time", data=np.full(theData.shape[0], 0.1))
                xpeem.create_dataset("stxm_scan_type", data=[b"sample image"])

                entry.create_dataset("title", data=[b"Ptychorecon data"])

            print(f"\n HDF5 file written to: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save HDF5: {e}")

    def close_application(self):
        self.window().close()
