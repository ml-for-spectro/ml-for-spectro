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
    QFormLayout,
    QLineEdit,
)
from utils.plotting import PlotCanvas, photon_energy_eV, be_to_ke, ke_to_be
from scipy.ndimage import gaussian_filter1d
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QLocale, QLibraryInfo


class LoadTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        # Main layout: horizontal (left: controls, right: plot)
        main_layout = QHBoxLayout()

        # Left side controls
        controls_layout = QVBoxLayout()

        # ── New: photon‑energy input ───────────────────────────
        energy_form = QFormLayout()
        self.energy_le = QLineEdit(f"{photon_energy_eV:.1f}")
        self.energy_le.setFixedWidth(80)
        self.energy_le.editingFinished.connect(self._update_energy)
        energy_form.addRow("Photon E (eV):", self.energy_le)
        controls_layout.addLayout(energy_form)
        # ───────────────────────────────────────────────────────

        self.load_button = QPushButton("Load Spectrum")
        self.exit_button = QPushButton("Exit")
        self.load_button.clicked.connect(self.load_spectrum)
        self.exit_button.clicked.connect(QApplication.quit)
        self.help_button = QPushButton("Help")

        self.load_button.setFixedSize(120, 30)
        self.exit_button.setFixedSize(120, 30)
        self.help_button.setFixedSize(120, 30)

        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.exit_button)
        controls_layout.addWidget(self.help_button)
        controls_layout.addStretch()

        self.help_button.clicked.connect(self.show_help)

        # Right side: plot + coordinate label
        right_layout = QVBoxLayout()
        self.path_label = QLabel("No file loaded")
        self.coord_label = QLabel("X: ---, Y: ---")
        font = QFont()
        font.setPointSize(12)
        self.coord_label.setFont(font)
        self.coord_label.setAlignment(Qt.AlignRight)

        self.canvas = PlotCanvas(self, coord_label=self.coord_label)

        right_layout.addWidget(self.path_label)
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.coord_label)

        # Add both layouts
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        # self.energy_le.setText(f"{self.parent.photon_energy:.1f}")

    def _update_energy(self):
        from utils import plotting  # avoid circular import

        try:
            new_val = float(self.energy_le.text())
            plotting.photon_energy_eV = new_val
            self.parent.photon_energy = new_val  # ← Save to main app instance!

            # Refresh current tab’s canvas
            if self.parent.x is not None:
                self.canvas.plot_data(self.parent.x, self.parent.y_current)

            # Refresh smoothing tab if it’s been plotted already
            self.parent.tabs.widget(1).refresh()
        except ValueError:
            QMessageBox.warning(self, "Bad value", "Please enter a number (e.g. 350)")
            self.energy_le.setText(f"{plotting.photon_energy_eV:.1f}")
        # print("User entered photon energy:", new_val)
        # print("Stored in parent:", self.parent.photon_energy)
        # print("plotting.photon_energy_eV =", plotting.photon_energy_eV)

    def load_spectrum(self):
        initial_dir = self.parent.last_dir if self.parent.last_dir else ""
        path, _ = QFileDialog.getOpenFileName(
            None, "Open File", initial_dir, "Text files (*.txt *.csv)"
        )
        if path:
            try:
                data = np.genfromtxt(path, delimiter=",", skip_header=1)
                self.parent.x = data[:, 0]
                self.parent.y_raw = data[:, 1]
                self.parent.y_current = self.parent.y_raw.copy()
                self.path_label.setText(path)
                self.canvas.plot_data(self.parent.x, self.parent.y_current)
                # Store directory for next time
                dir_path = os.path.dirname(path)
                self.parent.last_dir = dir_path
                self.parent.save_last_dir(dir_path)
                # Update smoothing tab
                self.parent.tabs.widget(1).refresh()
                # add this immediately after
                self.parent.tabs.widget(2).refresh()  # Background tab (index 2)
                self.parent.tabs.widget(3).refresh()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load file: {e}")

    def show_help(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        help_path = os.path.join(current_dir, "help_text.txt")

        try:
            with open(help_path, "r") as file:
                help_content = file.read()
        except FileNotFoundError:
            help_content = "Help file not found."

        QMessageBox.information(self, "Help", help_content)
