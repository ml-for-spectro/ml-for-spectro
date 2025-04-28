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
from PySide6.QtCore import Qt
import numpy as np

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
        if self.parent.x is not None and self.parent.y_current is not None:
            self.reset_crop_spinboxes()
            self._plot_raw()

    # ---------- Internal helpers -----------------------
    def _plot_raw(self):
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax2 = self.canvas.ax1.secondary_xaxis(
            "top", functions=(be_to_ke, ke_to_be)
        )
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_current, color="black", label="Raw"
        )
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.legend()
        self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
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
        # find nearest data index
        idx = (np.abs(self.parent.x - event.xdata)).argmin()
        self.pt_indices.append(idx)
        if len(self.pt_indices) == 2:
            self.canvas.mpl_disconnect(self._click_cid)
            self.pt_indices.sort()
            self._apply_background()

    # ---------------- Apply / Undo ----------------------
    def _apply_background(self):
        i1, i2 = self.pt_indices
        x, y = self.parent.x, self.parent.y_current

        if self.r_lin.isChecked():
            slope = (y[i2] - y[i1]) / (x[i2] - x[i1])
            bg = y[i1] + slope * (x - x[i1])
        else:
            bg = shirley_bg(x, y, i1, i2)

        self._preview_background(bg)
        self._confirm_background(bg)

    def _plot_bg_subtracted(self, bg):
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax2 = self.canvas.ax1.secondary_xaxis(
            "top", functions=(be_to_ke, ke_to_be)
        )
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_bgsub, label="BG‑subtracted", color="red"
        )
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.legend()
        self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
        self.canvas.draw()

    def _undo_bg(self):
        if self.bg_subtracted and self._prev_curve is not None:
            self.parent.y_current = self._prev_curve
            self._plot_raw()
            self.bg_subtracted = False
            self.save_btn.setEnabled(False)
            self.undo_btn.setEnabled(False)
            self.parent.tabs.widget(3).refresh()

    def _preview_background(self, bg):
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax2 = self.canvas.ax1.secondary_xaxis(
            "top", functions=(be_to_ke, ke_to_be)
        )
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_current, color="black", label="Raw"
        )
        self.canvas.ax1.plot(
            self.parent.x, bg, color="gray", linestyle="--", label="Proposed BG"
        )
        self.canvas.ax1.legend()
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
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
            self.bg_subtracted = True
            self.undo_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.parent.tabs.widget(3).refresh()
        else:
            self._plot_raw()

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

    def _apply_crop(self):
        """Crop spectrum to user‑defined BE window and propagate downstream."""
        vmin = self.min_be.value()
        vmax = self.max_be.value()

        if vmin >= vmax:
            QMessageBox.warning(self, "Range error", "Min BE must be < Max BE")
            return

        # Build mask and ensure at least two points remain
        mask = (self.parent.x >= vmin) & (self.parent.x <= vmax)
        if mask.sum() < 2:
            QMessageBox.warning(
                self, "Range error", "Window must contain at least two data points."
            )
            return

        # Save full spectrum only if not already saved
        if self._full_x is None or self._full_y is None:
            self._full_x = self.parent.x.copy()
            self._full_y = self.parent.y_current.copy()

        # Apply crop
        self.parent.x = self.parent.x[mask]
        self.parent.y_current = self.parent.y_current[mask]

        # Crop optional arrays
        if hasattr(self.parent, "y_raw"):
            self.parent.y_raw = self.parent.y_raw[mask]
        if hasattr(self.parent, "y_smoothed"):
            self.parent.y_smoothed = self.parent.y_smoothed[mask]
        if hasattr(self.parent, "y_bgsub"):
            self.parent.y_bgsub = self.parent.y_bgsub[mask]

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

            # Clear any smoothed or bgsub arrays because they are no longer valid
            if hasattr(self.parent, "y_raw"):
                del self.parent.y_raw
            if hasattr(self.parent, "y_smoothed"):
                del self.parent.y_smoothed
            if hasattr(self.parent, "y_bgsub"):
                del self.parent.y_bgsub

            self._full_x = None
            self._full_y = None
            self.reset_crop_spinboxes()
            self._plot_raw()
            self.undo_crop_btn.setEnabled(False)
            self.parent.tabs.widget(3).refresh()
