import os
import logging
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
)
from XPS_curvefit.utils.plotting import PlotCanvas, photon_energy_eV, be_to_ke, ke_to_be
from scipy.ndimage import gaussian_filter1d
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QLocale, QLibraryInfo, QSettings


class SmoothTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        layout = QVBoxLayout()

        # Controls at the top
        control_layout = QHBoxLayout()
        self.sigma_input = QDoubleSpinBox()
        self.sigma_input.setDecimals(1)
        self.sigma_input.setMinimum(0.1)
        self.sigma_input.setMaximum(50.0)
        self.sigma_input.setValue(2.0)
        self.sigma_input.setSuffix(" σ")
        self.save_button = QPushButton("Save Smoothed Spectrum")
        self.save_button.clicked.connect(self.save_smoothed_spectrum)

        self.apply_button = QPushButton("Apply Smoothing")
        self.apply_button.clicked.connect(self.apply_smoothing)
        control_layout.addWidget(self.save_button)

        control_layout.addWidget(QLabel("Smoothing width:"))
        control_layout.addWidget(self.sigma_input)
        control_layout.addWidget(self.apply_button)
        self.undo_button = QPushButton("Undo Smoothing")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_smoothing)
        control_layout.addWidget(self.undo_button)
        self._prev_curve = None  # to store previous y_current

        # Plot
        self.coord_label = QLabel("X: -, Y: -")
        self.canvas = PlotCanvas(self, self.coord_label)

        layout.addLayout(control_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.coord_label)
        self.setLayout(layout)

    def refresh(self):
        """Force update when new data is loaded."""
        if self.parent.x is not None and self.parent.y_current is not None:
            self.plot_raw_data()
        self.load_smoothing_settings()

    def plot_raw_data(self):
        """Plot just the raw data."""
        self.canvas.ax1.clear()
        # self.canvas.ax2.clear()
        # self.canvas.ax2 = self.canvas.ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_current, label="Raw Spectrum", color="black"
        )
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.legend()
        self.canvas.ax1.invert_xaxis()
        # self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
        self.canvas.draw()

    def apply_smoothing(self):
        if self.parent.x is None or self.parent.y_current is None:
            QMessageBox.warning(self, "No Data", "Please load a spectrum first.")
            return
        self._prev_curve = self.parent.y_current.copy()
        sigma = self.sigma_input.value()
        y_smoothed = gaussian_filter1d(self.parent.y_current, sigma)
        self.parent.y_smoothed = y_smoothed
        self.parent.y_current = y_smoothed
        self.undo_button.setEnabled(True)

        # Overplot the smoothed spectrum
        self.canvas.ax1.clear()
        # self.canvas.ax2.clear()
        # self.canvas.ax2 = self.canvas.ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_raw, label="Raw Spectrum", color="black"
        )
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.legend()

        self.canvas.ax1.plot(
            self.parent.x, y_smoothed, label=f"Smoothed (σ={sigma:.1f})", color="red"
        )
        self.canvas.ax1.invert_xaxis()
        self.canvas.draw()
        self.save_smoothing_settings()
        self.parent.tabs.widget(2).refresh()
        self.parent.tabs.widget(3).refresh()
        logging.info("Smoothing was applied by:")
        logging.info(f"Smoothed (σ={sigma:.1f})")

    def undo_smoothing(self):
        if self._prev_curve is not None:
            self.parent.y_current = self._prev_curve
            self._prev_curve = None
            self.undo_button.setEnabled(False)

            self.plot_raw_data()
            self.parent.tabs.widget(2).refresh()  # update Background tab if needed
            self.parent.tabs.widget(3).refresh()  # update Fit tab if needed

            logging.info("Smoothing undone")

    def save_smoothed_spectrum(self):
        if not hasattr(self.parent, "y_smoothed") or self.parent.y_smoothed is None:
            QMessageBox.warning(self, "No Data", "You need to apply smoothing first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Smoothed Spectrum", "", "CSV files (*.csv);;Text files (*.txt)"
        )
        if path:
            try:
                data = np.column_stack((self.parent.x, self.parent.y_smoothed))
                np.savetxt(
                    path, data, delimiter=",", header="X, Smoothed Y", comments=""
                )
                QMessageBox.information(
                    self, "Saved", f"Smoothed spectrum saved to:\n{path}"
                )
                logging.info(f"Smoothed spectrum saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
                logging.info(f"Failed to save file")

    def save_smoothing_settings(self):
        settings = QSettings()
        settings.setValue("smoothing_sigma", self.sigma_input.value())
        # settings.setValue("smoothing_method", self.method_combo.currentText())

    def load_smoothing_settings(self):
        settings = QSettings()
        sigma = settings.value("smoothing_sigma", 2.0, type=float)
        self.sigma_input.setValue(sigma)
