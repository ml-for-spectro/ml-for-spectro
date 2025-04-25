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
from XPS_curvefit.utils.fitting_helpers import build_voigt_model
from XPS_curvefit.utils.plotting import be_to_ke, ke_to_be
from PySide6.QtCore import Qt
import csv


class PeakEditor(QDialog):
    def __init__(self, parent, xdata, ydata, centers, amps, names):
        super().__init__(parent)
        self.setWindowTitle("Peak parameters")
        self.x, self.y = xdata, ydata

        # -------- 9 columns  ----------------------------------------
        self.table = QTableWidget(len(centers), 9, self)

        self.table.setHorizontalHeaderLabels(
            [
                "Name",
                "Center",
                "Sigma",
                "Gamma",
                "Amplitude",
                "Constr Center",
                "Constr Sigma",
                "Constr Gamma",
                "Constr Amp",
            ]
        )

        # -------- fill defaults -------------------------------------
        for r, (c, a) in enumerate(zip(centers, amps)):
            # Name column (read‑only)
            item = QTableWidgetItem(names[r])
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, item)

            self.table.setItem(r, 1, QTableWidgetItem(f"{c:.3f}"))  # Center
            self.table.setItem(r, 2, QTableWidgetItem("0.30"))  # Sigma
            self.table.setItem(r, 3, QTableWidgetItem("0.30"))  # Gamma
            self.table.setItem(r, 4, QTableWidgetItem(f"{a:.3f}"))  # Amplitude

            # empty constraint cells (cols 5‑8)
            for col in range(5, 9):
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
            numeric = [np.array(list(map(float, col))) for col in cols[1:5]]
        except ValueError:
            raise ValueError("Center, Sigma, Gamma and Amplitude must be numeric.")

        # last 4 columns are strings (constraints/expr)
        constraints = [np.array(cols[i]) for i in range(6, 9)]
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

    def _parse_table(self):
        """Return names, numeric arrays and constraint arrays from table."""
        cols = list(
            zip(
                *[
                    [self.table.item(r, c).text() for c in range(9)]
                    for r in range(self.table.rowCount())
                ]
            )
        )

        names = np.array(cols[0])
        try:
            centers = np.array(list(map(float, cols[1])))
            sigmas = np.array(list(map(float, cols[2])))
            gammas = np.array(list(map(float, cols[3])))
            amps = np.array(list(map(float, cols[4])))
        except ValueError:
            raise ValueError("Center, Sigma, Gamma, Amplitude must be numeric.")

        cen_con = np.array(cols[5])
        sig_con = np.array(cols[6])
        gam_con = np.array(cols[7])
        amp_con = np.array(cols[8])

        return names, centers, sigmas, gammas, amps, cen_con, sig_con, gam_con, amp_con

    # ------------------------------------------------------------------
    def _preview(self):
        try:
            (
                names,
                centers,
                sigmas,
                gammas,
                amps,
                cen_con,
                sig_con,
                gam_con,
                amp_con,
            ) = self._parse_table()
        except ValueError as err:
            QMessageBox.warning(self, "Bad input", str(err))
            return

        # --- build model with custom prefixes --------------------------
        prefixes = [f"{n}_" for n in names]  # ['A_','B_',...]
        model, pars = build_voigt_model(self.x, centers, pref_list=prefixes)

        # --- apply constraints / guesses -------------------------------
        for i, pref in enumerate(prefixes):
            self._parse_constraint(cen_con[i], pars[pref + "center"], centers[i])
            self._parse_constraint(sig_con[i], pars[pref + "sigma"], sigmas[i])
            self._parse_constraint(gam_con[i], pars[pref + "gamma"], gammas[i])
            self._parse_constraint(amp_con[i], pars[pref + "amplitude"], amps[i])

        preview = model.eval(pars, x=self.x)

        ax = self.parent().canvas.ax1
        [l.remove() for l in ax.lines if l.get_label() == "preview"]
        ax.plot(self.x, preview, ls="--", color="gray", label="preview")
        ax.legend()
        self.parent().canvas.draw()

    # ------------------------------------------------------------------
    def _do_fit(self):
        try:
            # names is column 0; numeric 1‑4; constraints 5‑8
            cols = list(
                zip(
                    *[
                        [self.table.item(r, c).text() for c in range(9)]
                        for r in range(self.table.rowCount())
                    ]
                )
            )
            names = np.array(cols[0])
            centers = np.array(list(map(float, cols[1])))
            sigmas = np.array(list(map(float, cols[2])))
            gammas = np.array(list(map(float, cols[3])))
            amps = np.array(list(map(float, cols[4])))
            cen_con = np.array(cols[5])
            sig_con = np.array(cols[6])
            gam_con = np.array(cols[7])
            amp_con = np.array(cols[8])
        except ValueError:
            QMessageBox.warning(
                self, "Bad input", "Center, Sigma, Gamma, Amplitude must be numeric."
            )
            return

        # ---- Build model using name prefixes (e.g. 'A_', 'B_') ----------
        model = None
        pars = None
        for pref, cen in zip(names, centers):
            v, p = build_voigt_model(self.x, [cen], pref_list=[pref + "_"])
            model = v if model is None else model + v
            pars = p if pars is None else (pars.update(p), pars)[1]

        # ---- apply constraints / initial values -------------------------
        for i, pref in enumerate(names):
            pref += "_"  # 'A_' etc.
            self._parse_constraint(cen_con[i], pars[pref + "center"], centers[i])
            self._parse_constraint(sig_con[i], pars[pref + "sigma"], sigmas[i])
            self._parse_constraint(gam_con[i], pars[pref + "gamma"], gammas[i])
            self._parse_constraint(amp_con[i], pars[pref + "amplitude"], amps[i])

        result = model.fit(self.y, pars, x=self.x)
        self.parent()._display_fit(result)  # update FitTab

        # ---- Update table with best-fit values -------------------------
        for i, pref in enumerate(names):
            pref += "_"
            self.table.setItem(
                i, 1, QTableWidgetItem(f"{result.params[pref + 'center'].value:.3f}")
            )
            self.table.setItem(
                i, 2, QTableWidgetItem(f"{result.params[pref + 'sigma'].value:.3f}")
            )
            self.table.setItem(
                i, 3, QTableWidgetItem(f"{result.params[pref + 'gamma'].value:.3f}")
            )
            self.table.setItem(
                i, 4, QTableWidgetItem(f"{result.params[pref + 'amplitude'].value:.3f}")
            )

        # Optional: notify user
        QMessageBox.information(
            self, "Fit Complete", "Fit complete. Table updated with best-fit values."
        )

        # Don't close the dialog!
        # self.accept()  ← REMOVE THIS

    def _save_params(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Params", "", "CSV (*.csv)")
        if not fn:
            return

        with open(fn, mode="w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            for r in range(self.table.rowCount()):
                row = [
                    self.table.item(r, c).text()
                    for c in range(self.table.columnCount())
                ]
                writer.writerow(row)

    def _load_params(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load Params", "", "CSV (*.csv)")
        if not fn:
            return

        try:
            with open(fn, newline="") as f:
                reader = csv.reader(f)
                data = list(reader)
        except Exception as e:
            QMessageBox.warning(self, "Load Failed", f"Could not read file:\n{e}")
            return

        expected_cols = self.table.columnCount()
        for i, row in enumerate(data):
            if len(row) != expected_cols:
                QMessageBox.warning(
                    self,
                    "Wrong format",
                    f"Row {i+1} has {len(row)} columns, expected {expected_cols}.",
                )
                return

        self.table.setRowCount(len(data))
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                clean_val = (
                    val.strip() if val.strip() else ""
                )  # make sure blanks are "" not None
                self.table.setItem(r, c, QTableWidgetItem(clean_val))

        self._preview()
