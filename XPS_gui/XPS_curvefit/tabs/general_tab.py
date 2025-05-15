import os
import numpy as np
import logging
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QGroupBox,
    QGridLayout,
    QInputDialog,
)
from XPS_curvefit.utils.plotting import PlotCanvas, photon_energy_eV, ke_to_be, be_to_ke
from PySide6.QtCore import QSettings
import pandas as pd


class GeneralUtilityTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.spectra = []  # List of dicts: {name, x, y, photon_energy, checkbox, ...}

        self.main_layout = QVBoxLayout()
        self.settings = QSettings("Synchrotron SOLEIL", "SXFA - Simple XPS fitting app")

        # Button bar
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Spectrum")
        self.compare_button = QPushButton("Compare Selected")
        self.send_button = QPushButton("Send to Analysis")
        self.send_button.setEnabled(False)
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.setEnabled(False)

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.compare_button)
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.remove_button)
        self.main_layout.addLayout(button_layout)

        # Plot area
        self.canvas = PlotCanvas(self)
        self.main_layout.addWidget(self.canvas)

        # Spectra list (as a group box with grid layout)
        self.spectra_group = QGroupBox("Loaded Spectra")
        self.spectra_layout = QGridLayout()
        self.spectra_group.setLayout(self.spectra_layout)
        self.main_layout.addWidget(self.spectra_group)

        self.setLayout(self.main_layout)

        self.load_button.clicked.connect(self.load_spectrum)
        self.compare_button.clicked.connect(self.compare_selected)
        self.send_button.clicked.connect(self.send_selected_to_analysis)

        self.remove_button.clicked.connect(self.remove_selected)

    def load_spectrum(self):
        if len(self.spectra) >= 6:
            QMessageBox.warning(
                self, "Limit Reached", "Maximum of 6 spectra supported."
            )
            return

        last_dir = self.settings.value("general_tab/last_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Spectrum", last_dir, "Data Files (*.csv *.txt *.dat)"
        )

        if not path:
            return
        self.settings.setValue("general_tab/last_dir", os.path.dirname(path))

        try:
            df = pd.read_csv(path)  # <-- use pandas to load

            x = df.iloc[:, 0].values
            y_all = df.iloc[:, 1:]

            if y_all.shape[1] == 1:
                y = y_all.iloc[:, 0].values
                logging.info("New file loaded")
                logging.info(f"{path}")
            else:
                options = list(y_all.columns)
                item, ok = QInputDialog.getItem(
                    self,
                    "Select Y column",
                    f"Choose Y data column ({len(options)} available):",
                    options,
                    0,
                    False,
                )
                if ok and item:
                    y = y_all[item].values
                    logging.info("The following column loaded:")
                    logging.info(f"Column {item}")
                else:
                    logging.info("Loading Cancelled")
                    return

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")
            return

        spectrum_name = os.path.basename(path)
        row = len(self.spectra) // 2
        col = len(self.spectra) % 2

        container = QWidget()
        container_layout = QHBoxLayout()
        container.setLayout(container_layout)

        checkbox = QCheckBox()
        checkbox.stateChanged.connect(self._update_send_button)
        name_label = QLabel(spectrum_name)
        energy_input = QDoubleSpinBox()
        energy_input.setRange(0, 2000)
        energy_input.setValue(photon_energy_eV)
        energy_input.setSuffix(" eV")
        energy_input.setDecimals(1)
        energy_input.setFixedWidth(100)

        container_layout.addWidget(checkbox)
        container_layout.addWidget(name_label)
        container_layout.addWidget(QLabel("Photon E:"))
        container_layout.addWidget(energy_input)
        container_layout.addStretch()

        self.spectra_layout.addWidget(container, row, col)

        self.spectra.append(
            {
                "name": spectrum_name,
                "x": x,
                "y": y,
                "photon_energy": energy_input,
                "checkbox": checkbox,
                "widget": container,  # Store the container widget reference
            }
        )

        checkbox.stateChanged.connect(
            lambda _: self.plot_selected(x, y, energy_input.value())
        )

    # def ke_to_be_local(ke, photon_energy):
    #   return photon_energy - ke

    def _update_send_button(self):
        selected = [s for s in self.spectra if s["checkbox"].isChecked()]
        self.send_button.setEnabled(len(selected) == 1)
        self.remove_button.setEnabled(len(selected) > 0)
        logging.info("File send for analysis")

    def plot_selected(self, x, y, energy):
        from XPS_curvefit.utils import plotting

        plotting.photon_energy_eV = energy
        filename = next(
            (s["name"] for s in self.spectra if np.array_equal(s["x"], x)), "Spectrum"
        )
        self.canvas.plot_data(x, y, label=filename)

    def compare_selected(self):
        selected = [s for s in self.spectra if s["checkbox"].isChecked()]
        if len(selected) < 2:
            QMessageBox.information(
                self, "Select Spectra", "Select at least two spectra to compare."
            )
            return

        normalize = (
            QMessageBox.question(
                self,
                "Normalize?",
                "Normalize spectra before comparing?",
                QMessageBox.Yes | QMessageBox.No,
            )
            == QMessageBox.Yes
        )

        self.canvas.ax1.clear()
        # self.canvas.ax2.clear()
        # self.canvas.ax2 = self.canvas.ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))

        for s in selected:
            x = s["x"]
            y = s["y"]
            if normalize:
                max_y = np.max(y)
                y = y / max_y if max_y != 0 else y
            label = s["name"]
            from XPS_curvefit.utils import plotting

            plotting.photon_energy_eV = s["photon_energy"].value()
            self.canvas.ax1.plot(x, y, label=label)

        self.canvas.ax1.set_xlabel("Binding Energy (eV)")
        self.canvas.ax1.set_ylabel("Intensity (a.u.)")
        self.canvas.ax1.legend()
        self.canvas.ax1.invert_xaxis()
        # self.canvas.ax2.set_xlabel("Kinetic Energy (eV)")
        self.canvas.draw()

    def send_selected_to_analysis(self):
        def ke_to_be_local(ke, photon_energy):
            return photon_energy - ke

        selected = [s for s in self.spectra if s["checkbox"].isChecked()]
        if len(selected) != 1:
            return

        s = selected[0]
        x_be = ke_to_be_local(s["x"], s["photon_energy"].value())
        self.parent.x = x_be
        # self.parent.x = s["x"]
        self.parent.y_raw = s["y"]
        self.parent.y_current = s["y"].copy()
        from XPS_curvefit.utils import plotting

        plotting.photon_energy_eV = s["photon_energy"].value()
        self.parent.photon_energy = plotting.photon_energy_eV
        self.parent.tabs.widget(1).refresh()
        self.parent.tabs.widget(2).refresh()
        self.parent.tabs.widget(3).refresh()
        QMessageBox.information(self, "Sent", f"{s['name']} sent to main analysis.")

    def remove_selected(self):
        selected_spectra = [s for s in self.spectra if s["checkbox"].isChecked()]

        if not selected_spectra:
            return

        for s in selected_spectra:
            widget = s["widget"]
            self.spectra_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
            self.spectra.remove(s)

        # Clear the plot
        self.canvas.ax1.clear()
        # self.canvas.ax2.clear()
        self.canvas.draw()

        self._update_send_button()
        self._update_remove_button()

    def _update_remove_button(self):
        selected = [s for s in self.spectra if s["checkbox"].isChecked()]
        self.remove_button.setEnabled(len(selected) > 0)
