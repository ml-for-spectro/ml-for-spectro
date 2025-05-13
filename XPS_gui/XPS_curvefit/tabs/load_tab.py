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
    QDialog,
    QScrollArea,
    QDialogButtonBox,
    QInputDialog,
)
import logging
from XPS_curvefit.utils.plotting import PlotCanvas, photon_energy_eV, be_to_ke, ke_to_be
from scipy.ndimage import gaussian_filter1d
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QColor
from PySide6.QtCore import Qt, QLocale, QLibraryInfo, QSettings


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
        from XPS_curvefit.utils import plotting  # avoid circular import

        try:
            new_val = float(self.energy_le.text())
            logging.info("New Photon energy value")
            logging.info(f"{new_val} eV")
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
            logging.info("Bad Photon energy value")

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
                x = ke_to_be(data[:, 0])
                # print(type(x))
                y_all = data[:, 1:]  # could be 1 or more columns
                logging.info("New file loaded")
                logging.info(f"{path}")
                if y_all.shape[1] == 1:
                    # Only one Y column
                    y_raw = y_all[:, 0]
                else:
                    # Multiple Y columns -> ask user which one to pick
                    options = [f"Column {i+1}" for i in range(y_all.shape[1])]
                    item, ok = QInputDialog.getItem(
                        self,
                        "Select Y column",
                        "Choose Y data column:",
                        options,
                        0,
                        False,
                    )
                    if ok and item:
                        idx = options.index(item)
                        y_raw = y_all[:, idx]
                        logging.info("The following column loaded:")
                        logging.info(f"Column {item}")
                    else:
                        # User cancelled
                        logging.info("Loading Cancelled")

                        return

                # Store in parent
                self.parent.x = x
                # print(type(self.parent.x))
                self.parent.y_raw = y_raw
                self.parent.y_current = y_raw.copy()
                self.path_label.setText(path)
                self.canvas.plot_data(self.parent.x, self.parent.y_current)

                # Store directory for next time
                dir_path = os.path.dirname(path)
                self.parent.last_dir = dir_path
                self.parent.save_last_dir(dir_path)

                # Update smoothing tab
                self.parent.tabs.widget(1).refresh()
                # Update background tab
                self.parent.tabs.widget(2).refresh()
                # Update fit tab
                self.parent.tabs.widget(3).refresh()

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load file: {e}")
                logging.info("Failed to load file.")

    def show_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Help")
        dialog.resize(600, 500)

        layout = QVBoxLayout(dialog)

        help_path = os.path.join(os.path.dirname(__file__), "../help_text.txt")
        try:
            with open(help_path, "r") as file:
                help_content = file.read()
                logging.info("Help file loaded")

        except FileNotFoundError:
            help_content = "Help file not found."
            logging.info("Help file missing.")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        label = QLabel(help_content)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        scroll.setWidget(label)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()
