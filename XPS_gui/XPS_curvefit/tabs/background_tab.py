from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QFileDialog,
    QButtonGroup,
    QMessageBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QSettings
import numpy as np
import logging
from XPS_curvefit.utils.plotting import PlotCanvas, be_to_ke, ke_to_be
from XPS_curvefit.utils.background import shirley_bg


class BackgroundTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self._full_x = None  # Store original x for undoing crop
        self._full_y = None  # Store original y for undoing crop
        self.pt_indices = []  # stores two indices chosen by user
        self.bg_subtracted = False

        # -------- UI ----------
        layout = QVBoxLayout()
        top_bar = QHBoxLayout()

        self.r_lin = QRadioButton("Linear")
        self.r_lin.setChecked(True)
        self.r_shi = QRadioButton("Shirley")
        btn_grp = QButtonGroup(self)
        btn_grp.addButton(self.r_lin)
        btn_grp.addButton(self.r_shi)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._start_selection)
        self.min_be = QDoubleSpinBox()
        self.max_be = QDoubleSpinBox()
        self.min_be.setDecimals(2)
        self.max_be.setDecimals(2)
        self.min_be.setMaximum(1000.0)
        self.max_be.setMaximum(1000.0)
        self.min_be.setValue(self.parent.x.min() if self.parent.x is not None else 0)
        self.max_be.setValue(self.parent.x.max() if self.parent.x is not None else 0)
        self.crop_btn = QPushButton("Crop")
        self.crop_btn.clicked.connect(self._apply_crop)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self._undo_bg)
        self.undo_crop_btn = QPushButton("Undo Crop")
        self.undo_crop_btn.clicked.connect(self._undo_crop)
        self.undo_crop_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)
        self.save_btn = QPushButton("Save Spectrum")
        self.save_btn.clicked.connect(self._save_bgsub)
        self.save_btn.setEnabled(False)  # only active after subtraction

        for w in (
            self.r_lin,
            self.r_shi,
            self.apply_btn,
            self.undo_btn,
            self.undo_crop_btn,
            self.save_btn,
            self.min_be,
            self.max_be,
            self.crop_btn,
        ):
            top_bar.addWidget(w)
        top_bar.addStretch()

        self.coord_label = QLabel("X: -, Y: -")
        self.canvas = PlotCanvas(self, self.coord_label)
        self._prev_curve = None

        layout.addLayout(top_bar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.coord_label)
        self.setLayout(layout)

    # ---------- Public API called by LoadTab -----------
    def refresh(self):
        self._full_x = None
        self._full_y = None

        # Reset all internal state
        self.pt_indices = []
        self.pt_coords = []
        self.pt_ys = []
        self.bg_subtracted = False
        self._prev_curve = None

        if hasattr(self, "_click_cid"):
            try:
                self.canvas.mpl_disconnect(self._click_cid)
            except Exception:
                pass
            del self._click_cid

        self.undo_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        if self.parent.x is not None and self.parent.y_current is not None:
            self.load_crop_settings()
            self._plot_raw()

    # ---------- Internal helpers -----------------------
    def _plot_raw(self):
        self.canvas.ax1.clear()
        # self.canvas.ax2.clear()
        # self.canvas.ax2 = self.canvas.ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_current, color="black", label="Raw"
        )
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.legend()
        self.canvas.ax1.invert_xaxis()
        # self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
        self.canvas.draw()
        self.bg_subtracted = False
        self.undo_btn.setEnabled(False)

    # ---------------- Background selection -------------
    def _start_selection(self):
        if self.parent.x is None:
            QMessageBox.warning(self, "No data", "Load a spectrum first.")
            return
        self.pt_indices.clear()
        QMessageBox.information(
            self,
            "Select points",
            "Click TWO points on the plot that define the background region.",
        )
        cid = self.canvas.mpl_connect("button_press_event", self._on_click)
        # store connection id so we can disconnect after two clicks
        self._click_cid = cid

    def _on_click(self, event):
        if event.inaxes != self.canvas.ax1:
            return

        x_clicked = event.xdata
        y_clicked = event.ydata
        logging.info("SClicking for bgnd was done")
        logging.info(f"X ={event.xdata})")
        logging.info(f"Y ={event.ydata})")
        idx = (np.abs(self.parent.x - x_clicked)).argmin()

        self.pt_indices.append(idx)

        if not hasattr(self, "pt_coords"):
            self.pt_coords = []
        if not hasattr(self, "pt_ys"):
            self.pt_ys = []

        self.pt_coords.append(x_clicked)
        self.pt_ys.append(y_clicked)

        # print("Clicked X:", self.pt_coords)
        # print("Clicked Y:", self.pt_ys)

        if len(self.pt_coords) == 1:
            if hasattr(self, "y1_input"):
                self.y1_input.setValue(y_clicked)
        elif len(self.pt_coords) == 2:
            if hasattr(self, "y2_input"):
                self.y2_input.setValue(y_clicked)
            self.canvas.mpl_disconnect(self._click_cid)
            self._apply_background()

    # ---------------- Apply / Undo ----------------------
    def _apply_background(self):
        # --- Safety checks ---
        if not hasattr(self, "pt_coords") or not hasattr(self, "pt_indices"):
            QMessageBox.warning(self, "Error", "You must select two points first.")
            logging.info("Error in selecting the points.")
            return
        if len(self.pt_coords) != 2 or len(self.pt_indices) != 2:
            QMessageBox.warning(self, "Error", "You must select exactly two points.")
            return

        # Extract coordinates and indices
        x1, x2 = self.pt_coords
        i1, i2 = self.pt_indices
        x, y = self.parent.x, self.parent.y_current

        # Use manually entered Y values if available, else fallback to clicked Ys
        if hasattr(self, "pt_ys") and len(self.pt_ys) == 2:
            y1_clicked, y2_clicked = self.pt_ys
        else:
            y1_clicked, y2_clicked = y[i1], y[i2]

        y1 = self.y1_input.value() if hasattr(self, "y1_input") else y1_clicked
        y2 = self.y2_input.value() if hasattr(self, "y2_input") else y2_clicked

        # --- Background calculation ---
        if self.r_lin.isChecked():
            try:
                slope = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                QMessageBox.warning(self, "Error", "X values must be distinct.")
                return
            bg = y1 + slope * (x - x1)
        else:
            # Shirley still relies on index range
            bg = shirley_bg(x, y, i1, i2)

        self._preview_background(bg)
        self._confirm_background(bg)

    def _plot_bg_subtracted(self, bg):
        self.canvas.ax1.clear()
        # self.canvas.ax2.clear()
        # self.canvas.ax2 = self.canvas.ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_bgsub, label="BG‑subtracted", color="red"
        )
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.invert_xaxis()
        self.canvas.ax1.legend()
        # self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
        self.canvas.draw()

    def _undo_bg(self):
        if self.bg_subtracted and self._prev_curve is not None:
            self.parent.y_current = self._prev_curve.copy()
            self._plot_raw()
            logging.info("Background removal undone.")
            self.bg_subtracted = False
            self.save_btn.setEnabled(False)
            self.undo_btn.setEnabled(False)

            # Re-enable crop undo if needed
            if self._full_x is not None and self._full_y is not None:
                self.undo_crop_btn.setEnabled(True)

            # Reset interaction state
            self.pt_indices = []
            self.pt_coords = []
            self.pt_ys = []

            self._click_cid = self.canvas.mpl_connect(
                "button_press_event", self._on_click
            )

            self.parent.tabs.widget(3).refresh()

    def _preview_background(self, bg):
        self.canvas.ax1.clear()
        # self.canvas.ax2.clear()
        # self.canvas.ax2 = self.canvas.ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_current, color="black", label="Raw"
        )
        self.canvas.ax1.plot(
            self.parent.x, bg, color="gray", linestyle="--", label="Proposed BG"
        )
        self.canvas.ax1.legend()
        self.canvas.ax1.invert_xaxis()
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        # self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
        self.canvas.draw()

    def _confirm_background(self, bg):
        res = QMessageBox.question(
            self,
            "Confirm",
            "Subtract this background?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if res == QMessageBox.Yes:
            self._prev_curve = self.parent.y_current.copy()  # ← save
            self.parent.y_bgsub = self.parent.y_current - bg
            self.parent.y_current = self.parent.y_bgsub
            self._plot_bg_subtracted(bg)
            logging.info("Background was removed.")
            self.bg_subtracted = True
            self.undo_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.parent.tabs.widget(3).refresh()
        else:
            # Reset interaction state so user can click again
            self.pt_indices = []
            self.pt_coords = []
            self.pt_ys = []
            # Reconnect click handler
            if hasattr(self, "_click_cid"):
                self.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = self.canvas.mpl_connect(
                "button_press_event", self._on_click
            )
            self._plot_raw()
            self.parent.tabs.widget(3).refresh()

    def _save_bgsub(self):
        if not hasattr(self.parent, "y_bgsub"):
            return
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Save Background‑Subtracted Spectrum",
            "",
            "CSV files (*.csv);;Text files (*.txt)",
        )
        if fn:
            import numpy as np, os

            data = np.column_stack((self.parent.x, self.parent.y_bgsub))
            np.savetxt(fn, data, delimiter=",", header="X,BG_subtracted_Y", comments="")
            QMessageBox.information(self, "Saved", os.path.basename(fn) + " written.")
            logging.info(f"Saved {os.path.basename(fn)} written.")

    def _apply_crop(self):
        """Crop spectrum to user-defined BE window and propagate downstream."""
        vmin = self.min_be.value()
        vmax = self.max_be.value()
        self.save_crop_values(vmin, vmax)

        if vmin >= vmax:
            QMessageBox.warning(self, "Range error", "Min BE must be < Max BE")
            return

        # Save full spectrum only if not already saved
        if self._full_x is None or self._full_y is None:
            self._full_x = self.parent.x.copy()
            self._full_y = self.parent.y_current.copy()

        # Build mask based on full x
        mask = (self._full_x >= vmin) & (self._full_x <= vmax)
        if mask.sum() < 2:
            QMessageBox.warning(
                self, "Range error", "Window must contain at least two data points."
            )
            return

        # Apply crop using _full_x and _full_y
        self.parent.x = self._full_x[mask]
        self.parent.y_current = self._full_y[mask]

        # Crop optional arrays
        if hasattr(self.parent, "y_raw"):
            self.parent.y_raw = self.parent.y_raw[mask]
        if hasattr(self.parent, "y_smoothed"):
            self.parent.y_smoothed = self.parent.y_smoothed[mask]
        if hasattr(self.parent, "y_bgsub"):
            self.parent.y_bgsub = self.parent.y_bgsub[mask]
        logging.info("Cropping was done.")
        # Update spinboxes, redraw, notify fit tab
        self.undo_crop_btn.setEnabled(True)
        self.reset_crop_spinboxes()
        self._plot_raw()
        self.parent.tabs.widget(3).refresh()

    def get_crop_values(self):
        return (self.min_be.value(), self.max_be.value())

    def set_crop_values(self, values):
        vmin, vmax = values
        self.min_be.setValue(vmin)
        self.max_be.setValue(vmax)

    def reset_crop_spinboxes(self):
        """Reset crop spinboxes intelligently: preserve values if still valid, otherwise reset."""
        if self.parent.x is None:
            return
        x_min = self.parent.x.min()
        x_max = self.parent.x.max()

        # Update min/max range of spinboxes
        self.min_be.setMinimum(x_min)
        self.min_be.setMaximum(x_max)
        self.max_be.setMinimum(x_min)
        self.max_be.setMaximum(x_max)

        # Only reset values if old ones are invalid
        if not (x_min <= self.min_be.value() <= x_max):
            self.min_be.setValue(x_min)
        if not (x_min <= self.max_be.value() <= x_max):
            self.max_be.setValue(x_max)

    def _undo_crop(self):
        if self._full_x is not None and self._full_y is not None:
            self.parent.x = self._full_x.copy()
            self.parent.y_current = self._full_y.copy()

            # Do NOT delete _prev_curve; it's needed for undo_bg
            for attr in ["y_raw", "y_smoothed", "y_bgsub"]:
                if hasattr(self.parent, attr):
                    delattr(self.parent, attr)

            self._full_x = None
            self._full_y = None
            self.reset_crop_spinboxes()
            self._plot_raw()
            self.undo_crop_btn.setEnabled(False)

            # Reconnect click handler
            if hasattr(self, "_click_cid"):
                self.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = self.canvas.mpl_connect(
                "button_press_event", self._on_click
            )

            # Enable background undo button if background was subtracted earlier
            if self.bg_subtracted and self._prev_curve is not None:
                self.undo_btn.setEnabled(True)

            self.parent.tabs.widget(3).refresh()
            logging.info("Cropping was undone.")

    def save_crop_values(self, x1, x2):
        settings = QSettings()
        settings.setValue("crop_x1", x1)
        settings.setValue("crop_x2", x2)

    def load_crop_settings(self):
        settings = QSettings()
        x1 = settings.value("crop_x1", None, type=float)
        x2 = settings.value("crop_x2", None, type=float)

        if x1 is not None and x2 is not None:
            self.min_be.setValue(x1)
            self.max_be.setValue(x2)
        else:
            self.reset_crop_spinboxes()
