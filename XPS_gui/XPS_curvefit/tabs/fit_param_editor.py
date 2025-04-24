from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QFileDialog,
)
import numpy as np
from utils.fitting_helpers import build_voigt_model
from utils.plotting import be_to_ke, ke_to_be


class PeakEditor(QDialog):
    def __init__(self, parent, xdata, ydata, centers, amps):
        super().__init__(parent)
        self.setWindowTitle("Peak parameters")
        self.x, self.y = xdata, ydata

        # -------- 1. 10 columns -------------------------------------
        self.table = QTableWidget(len(centers), 8, self)

        # -------- 2. headers ----------------------------------------
        self.table.setHorizontalHeaderLabels(
            [
                "Center",
                "Sigma",
                "Gamma",
                "Amplitude",
                "ConstrCenter",
                "ConstrSigma",
                "ConstrGamma",
                "ConstrAmp",
            ]
        )

        # -------- 3. fill defaults ----------------------------------
        for r, (c, a) in enumerate(zip(centers, amps)):
            self.table.setItem(r, 0, QTableWidgetItem(f"{c:.3f}"))  # center
            self.table.setItem(r, 1, QTableWidgetItem("0.3"))  # sigma
            self.table.setItem(r, 2, QTableWidgetItem("0.3"))  # gamma
            self.table.setItem(r, 3, QTableWidgetItem(f"{a:.3f}"))  # amplitude
            # empty expression cells
            for col in range(4, 8):
                self.table.setItem(r, col, QTableWidgetItem(""))

        # ------------------------------------------------------------
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Params")
        self.save_btn = QPushButton("Save Params")
        self.refresh_btn = QPushButton("Refresh")
        self.fit_btn = QPushButton("Fit")
        self.cancel_btn = QPushButton("Close")

        for b in (
            self.load_btn,
            self.save_btn,
            self.refresh_btn,
            self.fit_btn,
            self.cancel_btn,
        ):
            btn_layout.addWidget(b)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(btn_layout)

        # connections
        self.refresh_btn.clicked.connect(self._preview)
        self.fit_btn.clicked.connect(self._do_fit)
        self.cancel_btn.clicked.connect(self.reject)
        self.load_btn.clicked.connect(self._load_params)
        self.save_btn.clicked.connect(self._save_params)

    # ----- helpers -----
    def _get_table_params(self):
        """
        Returns 8 arrays:
        centers, sigmas, gammas, amps          (float)
        cen_con, sig_con, gam_con, amp_con     (str)
        Raises ValueError if any numeric cell is non‑numeric.
        """
        cols = list(
            zip(
                *[
                    [self.table.item(r, c).text() for c in range(8)]
                    for r in range(self.table.rowCount())
                ]
            )
        )

        # first 4 columns are numeric
        try:
            numeric = [np.array(list(map(float, col))) for col in cols[:4]]
        except ValueError:
            raise ValueError("Center, Sigma, Gamma and Amplitude must be numeric.")

        # last 4 columns are strings (constraints/expr)
        constraints = [np.array(cols[i]) for i in range(4, 8)]
        return numeric + constraints  # total 8 arrays

    def _parse_constraint(self, text, par, guess):
        """
        Interpret the constraint cell:
        [min,max] → bounded;  numeric → fixed;  expr → tied;  blank → free.
        """
        t = text.strip()

        # expression (tie)
        if t and not t.startswith("[") and not t.replace(".", "").isdigit():
            par.expr = t
            par.vary = False
            return

        # bounds [min,max]
        if t.startswith("[") and t.endswith("]"):
            try:
                lo, hi = map(float, t[1:-1].split(","))
                par.set(value=guess, min=lo, max=hi, vary=True)
            except ValueError:
                raise ValueError(f"Bad bounds: {t}")
            return

        # fixed numeric
        if t.replace(".", "").isdigit():
            par.set(value=float(t), vary=False)
            return

        # empty → free, no bounds
        par.set(value=guess, vary=True)

    def _preview(self):
        try:
            (centers, sigmas, gammas, amps, cen_con, sig_con, gam_con, amp_con) = (
                self._get_table_params()
            )
        except ValueError as err:
            QMessageBox.warning(self, "Bad input", str(err))
            return

        model, pars = build_voigt_model(self.x, centers)

        for i, c in enumerate(centers, 1):
            pref = f"v{i}_"
            self._parse_constraint(cen_con[i - 1], pars[pref + "center"], c)
            self._parse_constraint(sig_con[i - 1], pars[pref + "sigma"], sigmas[i - 1])
            self._parse_constraint(gam_con[i - 1], pars[pref + "gamma"], gammas[i - 1])
            self._parse_constraint(
                amp_con[i - 1], pars[pref + "amplitude"], amps[i - 1]
            )

        preview = model.eval(pars, x=self.x)

        ax = self.parent().canvas.ax1
        # remove previous preview lines
        [l.remove() for l in ax.lines if l.get_label() == "preview"]
        ax.plot(self.x, preview, ls="--", color="gray", label="preview")
        ax.legend()
        self.parent().canvas.draw()

    def _do_fit(self):
        try:
            (centers, sigmas, gammas, amps, cen_con, sig_con, gam_con, amp_con) = (
                self._get_table_params()
            )
        except ValueError as err:
            QMessageBox.warning(self, "Bad input", str(err))
            return

        model, pars = build_voigt_model(self.x, centers)

        for i, c in enumerate(centers, 1):
            pref = f"v{i}_"
            self._parse_constraint(cen_con[i - 1], pars[pref + "center"], c)
            self._parse_constraint(sig_con[i - 1], pars[pref + "sigma"], sigmas[i - 1])
            self._parse_constraint(gam_con[i - 1], pars[pref + "gamma"], gammas[i - 1])
            self._parse_constraint(
                amp_con[i - 1], pars[pref + "amplitude"], amps[i - 1]
            )

        result = model.fit(self.y, pars, x=self.x)
        self.parent()._display_fit(result)  # update FitTab
        self.accept()  # close dialog

    def _save_params(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Params", "", "CSV (*.csv)")
        if not fn:
            return
        arr = [
            [self.table.item(r, c).text() for c in range(self.table.columnCount())]
            for r in range(self.table.rowCount())
        ]
        np.savetxt(fn, arr, fmt="%s", delimiter=",")

    def _load_params(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load Params", "", "CSV (*.csv)")
        if not fn:
            return
        data = np.genfromtxt(fn, dtype=str, delimiter=",")
        self.table.setRowCount(len(data))
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(val))
        self._preview()  # draw initial curves
