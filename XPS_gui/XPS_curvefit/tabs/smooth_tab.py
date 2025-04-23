import os
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
from utils.plotting import PlotCanvas, photon_energy_eV, be_to_ke, ke_to_be
from scipy.ndimage import gaussian_filter1d
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QLocale, QLibraryInfo


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

    def plot_raw_data(self):
        """Plot just the raw data."""
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax2 = self.canvas.ax1.secondary_xaxis(
            "top", functions=(be_to_ke, ke_to_be)
        )
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_current, label="Raw Spectrum", color="black"
        )
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.legend()
        self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
        self.canvas.draw()

    def apply_smoothing(self):
        if self.parent.x is None or self.parent.y_current is None:
            QMessageBox.warning(self, "No Data", "Please load a spectrum first.")
            return

        sigma = self.sigma_input.value()
        y_smoothed = gaussian_filter1d(self.parent.y_current, sigma)
        self.parent.y_smoothed = y_smoothed
        self.parent.y_current = y_smoothed

        # Overplot the smoothed spectrum
        self.canvas.ax1.clear()
        self.canvas.ax2.clear()
        self.canvas.ax2 = self.canvas.ax1.secondary_xaxis(
            "top", functions=(be_to_ke, ke_to_be)
        )
        self.canvas.ax1.plot(
            self.parent.x, self.parent.y_raw, label="Raw Spectrum", color="black"
        )
        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.legend()

        self.canvas.ax1.plot(
            self.parent.x, y_smoothed, label=f"Smoothed (σ={sigma:.1f})", color="red"
        )
        self.canvas.draw()
        self.parent.tabs.widget(2).refresh()
        self.parent.tabs.widget(3).refresh()

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
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
