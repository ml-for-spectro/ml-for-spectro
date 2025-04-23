from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
import numpy as np

from utils.plotting import PlotCanvas, be_to_ke, ke_to_be
from utils.fitting_helpers import build_voigt_model


class FitTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.peak_ids = []  # indices user clicked
        self.fit_result = None
        self._prev_curve = None

        # ----- UI -----
        layout = QVBoxLayout()
        top = QHBoxLayout()
        self.fit_btn = QPushButton("Fit")
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        for b in (self.fit_btn, self.undo_btn, self.save_btn):
            top.addWidget(b)
        top.addStretch()

        self.coord_label = QLabel("X: -, Y: -")
        self.canvas = PlotCanvas(self, self.coord_label)

        layout.addLayout(top)
        layout.addWidget(self.canvas)
        layout.addWidget(self.coord_label)
        self.setLayout(layout)

        # signals
        self.fit_btn.clicked.connect(self._begin_pick)
        self.undo_btn.clicked.connect(self._undo_fit)
        self.save_btn.clicked.connect(self._save_fit)

    # ---------- called by other tabs ----------
    def refresh(self):
        if self.parent.x is not None and self.parent.y_current is not None:
            self._plot_curve()

    # ---------- internal plotting -------------
    def _plot_curve(self):
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax2 = self.canvas.ax1.secondary_xaxis(
            "top", functions=(be_to_ke, ke_to_be)
        )
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_current, color="black", label="Spectrum"
        )
        self.canvas.ax1.legend()
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
        self.canvas.draw()

    # ------------- pick peaks then fit --------
    def _begin_pick(self):
        if self.parent.y_current is None:
            QMessageBox.warning(self, "No data", "Load / process a spectrum first")
            return
        self.peak_ids.clear()
        QMessageBox.information(
            self, "Pick Peaks", "Click initial peak positions (double‑click last one)."
        )
        self._cid = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, ev):
        if ev.dblclick:  # finish on double‑click
            self.canvas.mpl_disconnect(self._cid)
            if len(self.peak_ids) == 0:
                QMessageBox.warning(self, "None", "No peaks picked.")
                return
            centers = self.parent.x[self.peak_ids]
            self._do_fit(centers)
        else:
            idx = (np.abs(self.parent.x - ev.xdata)).argmin()
            self.peak_ids.append(idx)
            # simple visual cue
            self.canvas.ax1.axvline(self.parent.x[idx], color="gray", ls=":")
            self.canvas.draw()

    def _do_fit(self, centers):
        self._prev_curve = self.parent.y_current.copy()
        model, params = build_voigt_model(self.parent.x, centers)
        self.fit_result = model.fit(self.parent.y_current, params, x=self.parent.x)

        y_fit = self.fit_result.best_fit
        self.parent.y_current = y_fit  # pipeline forward

        # plot result
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax2 = self.canvas.ax1.secondary_xaxis(
            "top", functions=(be_to_ke, ke_to_be)
        )
        self.canvas.ax1.plot(
            self.parent.x, self._prev_curve, color="black", label="Input"
        )
        self.canvas.ax1.plot(self.parent.x, y_fit, color="red", label="Fit")

        # individual components
        comps = self.fit_result.eval_components(x=self.parent.x)
        for name, comp in comps.items():
            self.canvas.ax1.plot(self.parent.x, comp, ls="--")

        self.canvas.ax1.legend()
        self.canvas.draw()
        self.undo_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    # ------------- undo and save --------------
    def _undo_fit(self):
        if self.fit_result is None:
            return
        self.parent.y_current = self._prev_curve
        self.fit_result = None
        self.undo_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self._plot_curve()

    def _save_fit(self):
        if self.fit_result is None:
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Fit Report", "", "Text files (*.txt)"
        )
        if fn:
            with open(fn, "w") as f:
                f.write(self.fit_result.fit_report())
            QMessageBox.information(self, "Saved", fn)
