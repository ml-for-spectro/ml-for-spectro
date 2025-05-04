import sys
import os
import ast
import logging
from datetime import datetime
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QLocale, QSettings, QCoreApplication

from XPS_curvefit.tabs.load_tab import LoadTab
from XPS_curvefit.tabs.smooth_tab import SmoothTab
from XPS_curvefit.tabs.background_tab import BackgroundTab
from XPS_curvefit.tabs.fit_tab import FitTab
from XPS_curvefit.tabs.general_tab import GeneralUtilityTab
from XPS_curvefit.tabs.log_tab import LogTab
from XPS_curvefit.config import load_last_dir, save_last_dir
from XPS_curvefit.utils import plotting
from XPS_curvefit.utils import logging_utils


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class XPSAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPS Analysis Tool")

        log_path = os.path.join(os.getcwd(), "app.log")  # <--- Step 1

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
        self.log_tab = LogTab(self, log_path)  # <--- Step 2

        self.tabs.addTab(self.load_tab, "Load Data")
        self.tabs.addTab(self.smooth_tab, "Smoothing")
        self.tabs.addTab(self.background_tab, "Background")
        self.tabs.addTab(self.fit_tab, "Voigt Fitting")
        self.tabs.addTab(self.general_tab, "General Utility")
        self.tabs.addTab(self.log_tab, "Log")

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

        # Setup logging
        # Initialize LogTab and store a reference to its QTextEdit

        self.text_edit = self.log_tab.text_edit
        # self.setup_logger()
        self.setup_logger(self.log_tab.text_edit)

        crop = settings.value("crop_values")
        # print(crop)
        if crop:
            try:
                if isinstance(crop, str):
                    x1, x2 = map(float, crop.strip("()").split(","))
                else:
                    x1, x2 = map(float, crop)
                self.background_tab.set_crop_values((x1, x2))
            except Exception as e:
                print("Could not restore crop values:", e)

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

    def setup_logger(self, text_edit_widget):
        # Go one level up from the current file's directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(log_dir, f"log_{timestamp}.txt")

        file_handler = logging.FileHandler(self.log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Attach QTextEdit logger to the GUI
        gui_handler = logging_utils.QTextEditLogger(text_edit_widget)
        logger.addHandler(gui_handler)


# ✅ ADD THIS FUNCTION so it works with setup.py entry_points
def main():
    QLocale.setDefault(QLocale(QLocale.C))
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
    QWidget {
        font-family: 'Helvetica Neue', 'Arial';
        font-size: 12pt;
        color: #2c2c2c;
    }

    QPushButton {
        background-color: #0078d7;
        color: white;
        font-weight: bold;
        border: none;
        padding: 6px 12px;
        border-radius: 5px;
    }

    QPushButton:hover:!disabled {
        background-color: #005fa3;
    }

    QPushButton:disabled {
        background-color: #cccccc;
        color: #666666;
        font-weight: normal;
    }

    QLineEdit, QDoubleSpinBox {
        border: 1px solid #999;
        padding: 4px;
        border-radius: 4px;
        background-color: #fcfcfc;
        font-weight: bold;
    }

    QLineEdit:disabled, QDoubleSpinBox:disabled {
        background-color: #f0f0f0;
        color: #888;
    }

    QLabel {
        font-weight: 600;
    }

    QTabWidget::pane {
        border: 1px solid #aaa;
        border-radius: 6px;
    }

    QTabBar::tab {
        background: #dcdcdc;
        font-weight: bold;
        padding: 8px 16px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }

    QTabBar::tab:selected {
        background: #ffffff;
        border: 1px solid #aaa;
        border-bottom-color: #ffffff;
    }

    QToolTip {
        color: #ffffff;
        background-color: #2c2c2c;
        border: none;
        padding: 5px;
        font-size: 10pt;
    }
"""
    )

    win = XPSAnalysisApp()
    win.resize(1000, 800)
    win.show()
    sys.exit(app.exec())


# ⬇️ This is still useful for direct execution
if __name__ == "__main__":
    main()
