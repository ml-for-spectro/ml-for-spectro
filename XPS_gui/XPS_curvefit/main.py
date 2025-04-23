import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from tabs.load_tab import LoadTab
from tabs.smooth_tab import SmoothTab
from tabs.background_tab import BackgroundTab
from tabs.fit_tab import FitTab
from config import load_last_dir, save_last_dir

from scipy.ndimage import gaussian_filter1d
from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QLocale, QLibraryInfo
from utils.plotting import PlotCanvas


class XPSAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPS Analysis Tool")
        self.x = None
        self.y = None
        self.y_raw = None  # original
        self.y_current = None  # latest processed version
        self.last_dir = load_last_dir()
        self.save_last_dir = save_last_dir

        self.tabs = QTabWidget()
        self.tabs.addTab(LoadTab(self), "Load Data")
        self.tabs.addTab(SmoothTab(self), "Smoothing")
        self.tabs.addTab(BackgroundTab(self), "Background")
        self.tabs.addTab(FitTab(self), "Voigt Fitting")

        self.setCentralWidget(self.tabs)


if __name__ == "__main__":
    QLocale.setDefault(QLocale(QLocale.C))
    app = QApplication(sys.argv)
    win = XPSAnalysisApp()
    win.resize(1000, 800)
    win.show()
    sys.exit(app.exec())
