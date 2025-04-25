import sys
import os
import ast
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QLocale, QSettings, QCoreApplication

from tabs.load_tab import LoadTab
from tabs.smooth_tab import SmoothTab
from tabs.background_tab import BackgroundTab
from tabs.fit_tab import FitTab
from tabs.general_tab import GeneralUtilityTab
from config import load_last_dir, save_last_dir
from utils import plotting


class XPSAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPS Analysis Tool")

        QCoreApplication.setOrganizationName("Synchrotron SOLEIL")
        QCoreApplication.setApplicationName("SXFA - Simple XPS fitting app")

        self.x = None
        self.y = None
        self.y_raw = None
        self.y_current = None
        self.last_dir = load_last_dir()
        self.save_last_dir = save_last_dir

        self.tabs = QTabWidget()
        self.load_tab = LoadTab(self)
        self.smooth_tab = SmoothTab(self)
        self.background_tab = BackgroundTab(self)
        self.fit_tab = FitTab(self)
        self.general_tab = GeneralUtilityTab(self)

        self.tabs.addTab(self.load_tab, "Load Data")
        self.tabs.addTab(self.smooth_tab, "Smoothing")
        self.tabs.addTab(self.background_tab, "Background")
        self.tabs.addTab(self.fit_tab, "Voigt Fitting")
        self.tabs.addTab(self.general_tab, "General Utility")

        self.setCentralWidget(self.tabs)

        settings = QSettings()
        photon_energy = settings.value("photon_energy")
        try:
            self.photon_energy = (
                float(photon_energy) if photon_energy is not None else 300.0
            )
        except ValueError:
            self.photon_energy = 300.0
            print("Invalid saved photon energy; using default.")
        plotting.photon_energy_eV = self.photon_energy
        self.load_tab.energy_le.setText(f"{self.photon_energy:.1f}")
        # Crop values — handle both list and string types
        crop = settings.value("crop_values")
        if crop:
            try:
                if isinstance(crop, str):
                    x1, x2 = map(float, crop.strip("()").split(","))
                else:
                    x1, x2 = map(float, crop)
                self.background_tab.set_crop_values((x1, x2))
            except Exception as e:
                print("Could not restore crop values:", e)

        # print("Loaded photon energy from settings:", self.photon_energy)
        # print("plotting.photon_energy_eV =", plotting.photon_energy_eV)

    def closeEvent(self, event):
        settings = QSettings()
        settings.setValue("photon_energy", self.photon_energy)
        plotting.photon_energy_eV = self.photon_energy

        try:
            settings.setValue(
                "crop_values", list(self.background_tab.get_crop_values())
            )
        except Exception as e:
            print("Error saving crop values:", e)
        print("Saving photon energy:", self.photon_energy)
        super().closeEvent(event)


if __name__ == "__main__":
    QLocale.setDefault(QLocale(QLocale.C))
    app = QApplication(sys.argv)
    win = XPSAnalysisApp()
    win.resize(1000, 800)
    win.show()
    sys.exit(app.exec())
