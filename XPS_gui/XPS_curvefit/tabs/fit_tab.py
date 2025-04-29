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
import logging
from XPS_curvefit.utils.plotting import PlotCanvas, be_to_ke, ke_to_be
from XPS_curvefit.utils.fitting_helpers import build_voigt_model
from XPS_curvefit.tabs.fit_param_editor import PeakEditor


class FitTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.peak_ids = []  # indices user clicked
        self.editor = None  # <-- Add this!!
        self.amp_guesses = []
        self.fit_result = None
        self._prev_curve = None

        # ----- UI -----
        layout = QVBoxLayout()
        top = QHBoxLayout()
        self.fit_btn = QPushButton("Fit")
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.save_btn = QPushButton("Save Fit report")
        self.save_btn.setEnabled(False)
        self.save_curve_btn = QPushButton("Save Curve")
        self.save_curve_btn.setEnabled(False)

        for b in (self.fit_btn, self.undo_btn, self.save_btn, self.save_curve_btn):
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
        self.save_curve_btn.clicked.connect(self._save_curves)
        # self.editor.fit_done.connect(self._handle_fit_done)

    # ---------- called by other tabs ----------
    def refresh(self):
        if self.parent.x is not None and self.parent.y_current is not None:
            self._plot_curve()
        self.fit_result = None
        self.undo_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.save_curve_btn.setEnabled(False)

    # ---------- internal plotting -------------
    def _plot_curve(self):
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        # print(self.parent.x)
        # print(self.parent.y_current)

        if self.parent.x is None or self.parent.y_current is None:
            self.canvas.draw()
            return
        if len(self.parent.x) != len(self.parent.y_current):
            self.canvas.draw()
            return

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
        if self.parent.x is None or self.parent.y_current is None:
            QMessageBox.warning(self, "No data", "Load / process a spectrum first")
            return
        if len(self.parent.x) != len(self.parent.y_current):
            QMessageBox.warning(
                self, "Invalid data", "X and Y have mismatched lengths."
            )
            return

        self.peak_ids.clear()
        self.amp_guesses.clear()
        QMessageBox.information(
            self,
            "Pick Peaks",
            "Click initial peak positions and height (double‑click last one).",
        )
        self._cid = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, ev):
        if ev.dblclick:
            self.canvas.mpl_disconnect(self._cid)

            if not self.peak_ids:
                QMessageBox.warning(self, "None", "No peaks picked.")
                self._plot_curve()
                return

            centers = self.parent.x[self.peak_ids]
            amps = np.array(self.amp_guesses)
            names = [chr(ord("A") + k) for k in range(len(self.peak_ids))]
            self._prev_curve = self.parent.y_current
            # open parameter editor
            self.editor = PeakEditor(
                self, self.parent.x, self.parent.y_current, centers, amps, names
            )
            self.editor.fit_done.connect(self._handle_fit_done)
            self.editor.exec()

            # reset
            self.peak_ids.clear()
            self.amp_guesses.clear()

        else:
            idx = (np.abs(self.parent.x - ev.xdata)).argmin()
            self.peak_ids.append(idx)
            self.amp_guesses.append(self.parent.y_current[idx])

            label = chr(ord("A") + len(self.peak_ids) - 1)
            self.canvas.ax1.axvline(self.parent.x[idx], color="gray", ls=":")
            self.canvas.ax1.text(
                self.parent.x[idx],
                self.parent.y_current[idx],
                f" {label}",
                rotation=90,
                va="bottom",
            )
            self.canvas.draw()

    # def _do_fit(self, centers):
    #     self._prev_curve = self.parent.y_current.copy()
    #     model, params = build_voigt_model(self.parent.x, centers)
    #     self.fit_result = model.fit(self.parent.y_current, params, x=self.parent.x)

    #     y_fit = self.fit_result.best_fit
    #     self.parent.y_current = y_fit  # pipeline forward

    #     # plot result
    #     self.canvas.ax1.clear()
    #     self.canvas.ax2.clear()
    #     self.canvas.ax2 = self.canvas.ax1.secondary_xaxis(
    #         "top", functions=(be_to_ke, ke_to_be)
    #     )
    #     self.canvas.ax1.plot(
    #         self.parent.x, self._prev_curve, color="black", label="Input"
    #     )
    #     self.canvas.ax1.plot(self.parent.x, y_fit, color="red", label="Fit")

    #     # individual components
    #     comps = self.fit_result.eval_components(x=self.parent.x)
    #     for name, comp in comps.items():
    #         self.canvas.ax1.plot(self.parent.x, comp, ls="--")

    #     self.parent()._display_fit(result)
    #     self.accept()

    #     self.canvas.ax1.legend()
    #     self.canvas.draw()
    #     self.undo_btn.setEnabled(True)
    #     self.save_btn.setEnabled(True)

    # ------------- undo and save --------------
    def _undo_fit(self):
        if self.fit_result is None:
            return
        self.parent.y_current = self._prev_curve
        self.fit_result = None
        self._prev_curve = None  # <--- ADD THIS LINE
        self.undo_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.save_curve_btn.setEnabled(False)
        print("here")
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

    def _save_curves(self):
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save All Curves", "", "CSV files (*.csv)"
        )
        if not fn:
            return
        raw = self.parent.y_raw
        smooth = getattr(self.parent, "y_smoothed", np.full_like(raw, np.nan))
        bg = getattr(self.parent, "bg_used", np.full_like(raw, np.nan))
        bgsub = getattr(self.parent, "y_bgsub", np.full_like(raw, np.nan))
        comps = self.fit_result.eval_components(x=self.parent.x)
        total = self.fit_result.best_fit
        cols = (
            [self.parent.x, be_to_ke(self.parent.x), raw, smooth, bg, bgsub]
            + [comps[k] for k in sorted(comps)]
            + [total]
        )
        array = np.column_stack(cols)
        header = (
            ["BE", "KE", "raw", "smooth", "bg", "bgsub"] + sorted(comps) + ["total"]
        )
        np.savetxt(fn, array, delimiter=",", header=",".join(header), comments="")

    def _display_fit(self, result):
        self.fit_result = result
        self._y_fit = result.best_fit  # <-- keep best fit separately
        # DO NOT modify self.parent.y_current!

        ax1 = self.canvas.ax1
        ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax2 = ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))

        # raw + total fit
        ax1.plot(
            self.parent.x, self.parent.y_current, color="black", label="Input"
        )  # <-- original data
        ax1.plot(
            self.parent.x, self._y_fit, color="red", label="Total fit"
        )  # <-- fit curve

        # plot individual components
        comps = result.eval_components(x=self.parent.x)
        for name, comp in comps.items():
            ax1.plot(self.parent.x, comp, ls="--")
            idx_peak = np.argmax(comp)
            x_peak = self.parent.x[idx_peak]
            y_peak = comp[idx_peak]
            ax1.text(
                x_peak,
                y_peak,
                f" {name.rstrip('_')}",
                va="bottom",
                ha="left",
                fontsize=8,
            )

        chi2 = result.redchi
        ax1.text(
            0.02,
            0.95,
            f"χ²_red = {chi2:.3g}",
            transform=ax1.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.7),
        )

        ax1.set_xlabel("Binding Energy (eV)")
        ax1.set_ylabel("Intensity (a.u.)")
        ax1.legend()
        self.canvas.draw()

        hits = []
        for par in self.fit_result.params.values():
            if par.vary and par.stderr is not None:
                if np.isclose(par.value, par.min, atol=1e-8):
                    hits.append(f"{par.name} at lower bound")
                if np.isclose(par.value, par.max, atol=1e-8):
                    hits.append(f"{par.name} at upper bound")

        if hits:
            QMessageBox.warning(
                self,
                "Boundary warning",
                "Some parameters hit bounds:\n" + "\n".join(hits),
            )

    def _handle_fit_done(self, result):
        self._display_fit(result)
        self.save_btn.setEnabled(True)
        self.save_curve_btn.setEnabled(True)
        logging.info(f"Fitting was done.")
