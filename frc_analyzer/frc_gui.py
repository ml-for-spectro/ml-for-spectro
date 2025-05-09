import os
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.signal import savgol_filter
from numpy import fft
from PIL import Image
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from matplotlib.figure import Figure
from PySide6.QtCore import QLocale, QSettings, QFileInfo
from PySide6.QtGui import QPixmap

# import pyqtgraph as pg


class FRCViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FRC Analyzer")
        self.resize(1000, 800)
        QLocale.setDefault(QLocale.c())
        self.settings = QSettings("Synchrotron SOLEIL", "FRCAnalyzer")
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Top row: Load and Exit buttons ---
        top_row = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.exit_button = QPushButton("Exit")
        self.spin0 = QDoubleSpinBox()
        self.spin0.setPrefix("Pixel size (nm): ")
        self.spin0.setValue(10)
        self.spin0.setLocale(QLocale(QLocale.C))

        self.exit_button.clicked.connect(self.close)
        top_row.addWidget(self.load_button)
        top_row.addWidget(self.exit_button)
        top_row.addWidget(self.spin0)
        main_layout.addLayout(top_row)

        # --- Second row: FRC, Reset, and three spinboxes ---
        second_row = QHBoxLayout()
        self.frc_button = QPushButton("FRC")
        self.save_button = QPushButton("Save Screengrab")

        # Input fields
        self.spin1 = QDoubleSpinBox()
        self.spin1.setPrefix("Max std: ")
        self.spin1.setValue(40)
        self.spin1.setLocale(QLocale(QLocale.C))

        self.spin2 = QDoubleSpinBox()
        self.spin2.setPrefix("Iterations: ")
        self.spin2.setValue(100)
        self.spin2.setLocale(QLocale(QLocale.C))

        self.spin3 = QDoubleSpinBox()
        self.spin3.setPrefix("Increment: ")
        self.spin3.setValue(0.01)
        self.spin3.setLocale(QLocale(QLocale.C))

        second_row.addWidget(self.frc_button)
        second_row.addWidget(self.save_button)
        second_row.addWidget(self.spin1)
        second_row.addWidget(self.spin2)
        second_row.addWidget(self.spin3)
        main_layout.addLayout(second_row)

        # --- Bottom row: two plots side by side ---
        bottom_row = QHBoxLayout()

        self.image_figure = Figure()
        self.image_canvas = FigureCanvas(self.image_figure)
        self.image_ax = self.image_figure.add_subplot(111)
        bottom_row.addWidget(self.image_canvas)

        self.frc_figure = Figure()
        self.frc_canvas = FigureCanvas(self.frc_figure)
        self.frc_ax = self.frc_figure.add_subplot(111)
        self.frc_twin_ax = self.frc_ax.twiny()
        bottom_row.addWidget(self.frc_canvas)

        main_layout.addLayout(bottom_row)

        # Connections
        self.load_button.clicked.connect(self.load_image)
        self.frc_button.clicked.connect(self.calculation_with_iterations)
        self.save_button.clicked.connect(self.save_screengrab)

        # Enable status
        self.spin0.setEnabled(False)
        self.spin1.setEnabled(False)
        self.spin2.setEnabled(False)
        self.spin3.setEnabled(False)
        self.frc_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def read_image(self, path):
        if path.endswith(".txt"):
            my_img = np.loadtxt(path)
            return my_img
        elif path.endswith(".tif") or path.endswith(".tiff"):
            my_img = np.array(Image.open(path))
            return my_img
        else:
            raise ValueError("Unsupported file format")

    def load_image(self):
        last_dir = self.settings.value("last_directory", "")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", last_dir, "TIFF files (*.tif *.tiff, *.txt)"
        )

        if file_path:
            self.settings.setValue(
                "last_directory", QFileInfo(file_path).absolutePath()
            )
            img = self.read_image(file_path)
            self.image_current = img

            self.display_image(self.image_current, file_path.split("/")[-1])

    def display_image(self, img, path):
        self.image_ax.clear()
        self.image_ax.imshow(img, cmap="gray")
        self.image_ax.set_title(path)
        self.image_canvas.draw()
        self.frc_figure.delaxes(self.frc_twin_ax)  # remove old twin axis
        self.frc_ax.clear()
        self.frc_twin_ax = self.frc_ax.twiny()
        self.frc_canvas.draw()

        self.spin0.setEnabled(True)
        self.spin1.setEnabled(True)
        self.spin2.setEnabled(True)
        self.spin3.setEnabled(True)
        self.frc_button.setEnabled(True)
        self.save_button.setEnabled(True)

    """ def run_analysis(self):
        if self.image is None:
            return

        pixnm = self.pixel_size.value()
        iterations = self.iterations.value()
        max_std = self.max_noise.value()
        increment = self.noise_increment.value()
        lim_freq = 1 / (2 * pixnm)

        all_FRC, all_n = [], []
        for _ in range(iterations):
            std = 1
            while std <= max_std:
                noisy = add_gaussian_noise(self.image, std_dev=std)
                im1, im2 = split_image(noisy)
                frc_vals, n_vals = frc(im1, im2)
                frc_smooth = savgol_filter(frc_vals, 7, 1)
                snr = snr_threshold(n_vals)
                inter = find_intersection(
                    np.linspace(0, lim_freq, len(frc_smooth)), frc_smooth, snr
                )
                if inter:
                    all_FRC.append(frc_smooth)
                    all_n.append(n_vals)
                    break
                std += increment

        mean_frc = np.mean(all_FRC, axis=0)
        mean_n = np.mean(all_n, axis=0)
        freqs = np.linspace(0, lim_freq, len(mean_frc))

        self.ax_frc.clear()
        self.ax_frc.plot(freqs, mean_frc, label="FRC", lw=2)
        self.ax_frc.plot(
            freqs, snr_threshold(mean_n), label="SNR Threshold", linestyle="--"
        )
        self.ax_frc.plot(
            freqs, half_bit_threshold(mean_n), label="Half-bit Threshold", linestyle=":"
        )
        self.ax_frc.set_title("FRC Analysis")
        self.ax_frc.set_xlabel("Spatial Frequency (1/nm)")
        self.ax_frc.set_ylabel("FRC")
        self.ax_frc.legend()
        self.ax_frc.set_xlim([0, lim_freq])
        self.ax_frc.set_ylim([0, 1.05])

        self.canvas.draw() """

    # ==============================================
    # 2. Add Noise
    # ==============================================
    def add_gaussian_noise(self, image_array, mean=0, std_dev=0.01):
        """Add Gaussian noise to the image."""
        noise = np.random.normal(mean, std_dev, image_array.shape)
        noisy_image = image_array + noise
        return noisy_image

    # ==============================================
    # 3. Split into Two Sub-images
    # ==============================================
    def sous_images(self, image):
        half_even = image[::2, ::2]
        half_odd = image[1::2, 1::2]
        return half_odd, half_even

    def find_ring_area(self, map_dist, r, width_ring):
        return np.argwhere((map_dist >= r) & (map_dist <= r + width_ring))

    def frc(self, image1, image2, r=3, width_ring=0.8):
        ny, nx = np.shape(image1)
        max_size = min(nx, ny)
        freq_nyq = int(np.floor(max_size / 2.0))
        x = np.arange(-np.floor(max_size / 2.0), np.ceil(max_size / 2.0))
        y = np.arange(-np.floor(max_size / 2.0), np.ceil(max_size / 2.0))
        x, y = np.meshgrid(x, y)
        map_dist = np.sqrt(x * x + y * y)
        fft_image1 = fft.fftshift(fft.fftn(image1))
        fft_image2 = fft.fftshift(fft.fftn(image2))

        C1, C2, C3, n = [], [], [], []
        while r + width_ring < freq_nyq:
            ring = self.find_ring_area(map_dist, r, width_ring)
            aux1, aux2 = (
                fft_image1[ring[:, 0], ring[:, 1]],
                fft_image2[ring[:, 0], ring[:, 1]],
            )
            C1.append(np.sum(aux1 * np.conjugate(aux2)))
            C2.append(np.sum(np.abs(aux1) ** 2))
            C3.append(np.sum(np.abs(aux2) ** 2))
            n.append(len(aux1))
            r += width_ring
        C1, C2, C3 = np.array(C1), np.array(C2), np.array(C3)
        FRC = np.abs(C1) / np.sqrt(C2 * C3)
        FRC = np.where(np.isnan(FRC), 0, FRC)

        return FRC, np.array(n)

    def solutions_snr_polynomial(self, n, FRC):
        e = (FRC * np.sqrt(n) - 1) / (1 - FRC)
        snr_min = []
        snr_max = []
        snr_mean = []
        Roots = []

        for i in range(len(e)):
            a = n[i]
            b = 4 * np.sqrt(n[i])
            c = 4
            d = 0
            constant = -(e[i] ** 2)

            coefficients = [a, b, c, d, constant]

            # Calculer les racines
            roots = np.roots(coefficients)
            # Filtrer les racines réelles
            real_roots = [root.real for root in roots if np.isreal(root)]

            if real_roots:
                # Ajouter les racines réelles si elles existent
                Roots.append(real_roots)
                # Calculer SNR pour min, max et moyenne
                snr_min.append(np.min(real_roots) ** 2)
                snr_max.append(np.max(real_roots) ** 2)
                snr_mean.append(np.mean([root**2 for root in real_roots]))
            else:
                # Si aucune racine réelle n'est trouvée, sélectionner la racine complexe
                # dont la partie imaginaire est la plus proche de zéro
                closest_to_real = min(roots, key=lambda root: abs(root.imag))
                snr_min.append(closest_to_real.real**2)
                snr_max.append(closest_to_real.real**2)
                snr_mean.append(closest_to_real.real**2)
                Roots.append([closest_to_real.real])

        # Convertir les listes en tableaux NumPy
        snr_min = np.array(snr_min)
        snr_max = np.array(snr_max)
        snr_mean = np.array(snr_mean)
        Roots = np.array(Roots, dtype=object)

        # print("SNR Min:", snr_min)
        # print("SNR Max:", snr_max)
        # print("SNR Mean:", snr_mean)
        # print("Roots:", Roots)
        # x=np.linspace(-1,1,1000)
        # p=a*x**4+b*x**3+c*x**2+d*x+constant
        # y=[0 for _ in range(len(x))]
        # plt.plot(x,p)
        # plt.plot(x,y)
        # plt.show()
        return snr_min, snr_max, snr_mean, Roots

    def threshold(self, num, SNR_list):
        T_nested = [
            (SNR + 2 * SNR / np.sqrt(i) + 1 / np.sqrt(i))
            / (1 + SNR + 2 * np.sqrt(SNR) / np.sqrt(i))
            for SNR, i in zip(SNR_list, num)
        ]
        return np.array(T_nested)

    def find_intersections_with_interpolation(self, x, y1, y2):
        """
        Trouver le premier point d'intersection entre deux courbes
        avec interpolation linéaire.

        Args:
            x (array): Les abscisses.
            y1 (array): La première courbe.
            y2 (array): La deuxième courbe.

        Returns:
            float or None: L'abscisse du premier point d'intersection, ou None s'il n'y a pas d'intersection.
        """
        try:
            # Calcul des différences entre les deux courbes
            diff = y1 - y2
            sign_change = np.diff(np.sign(diff))  # Détecte les changements de signe
            intersect_indices = np.where(sign_change != 0)[
                0
            ]  # Indices des intersections

            # Si aucune intersection n'est trouvée
            if len(intersect_indices) == 0:
                # print("No intersection found between the curves.")
                return None

            # Trouver la première intersection avec interpolation linéaire
            i = intersect_indices[0]
            x1, x2 = x[i], x[i + 1]
            y1_diff, y2_diff = diff[i], diff[i + 1]

            # Interpolation linéaire
            x_intersect = x1 - (y1_diff / (y2_diff - y1_diff)) * (x2 - x1)
            return x_intersect

        except Exception as e:
            print(f"Error in find_intersections_with_interpolation: {e}")
            return None

    # ==============================================
    #  6. Incertitude pour la résolution
    # ==============================================
    def calculate_uncertainty(self, value, std_dev):
        """
        Calcule les incertitudes pour une mesure donnée et son écart type.

        Args:
            value (float): La valeur centrale (par exemple, une fréquence).
            std_dev (float): L'écart type ou l'incertitude associée.

        Returns:
            tuple: (uncertainty_plus, uncertainty_minus, max_uncertainty)
                - uncertainty_plus: Incertitude pour value + std_dev.
                - uncertainty_minus: Incertitude pour value - std_dev.
                - max_uncertainty: La plus grande des incertitudes absolues.
        """
        # Calcul des incertitudes en s'assurant d'éviter les valeurs invalides
        if value - std_dev > 0:
            uncertainty_minus = (1 / value) - (1 / (value - std_dev))
        else:
            uncertainty_minus = np.nan  # Valeur invalide, retourne NaN

        if value + std_dev > 0:
            uncertainty_plus = (1 / value) - (1 / (value + std_dev))
        else:
            uncertainty_plus = np.nan  # Valeur invalide, retourne NaN

        # Calcul de l'incertitude maximale
        max_uncertainty = max(np.abs(uncertainty_plus), np.abs(uncertainty_minus))
        return uncertainty_plus, uncertainty_minus, max_uncertainty

    def calculation_with_iterations(self):
        self.frc_figure.delaxes(self.frc_twin_ax)  # remove old twin axis
        self.frc_ax.clear()
        self.frc_twin_ax = self.frc_ax.twiny()
        self.frc_canvas.draw()
        pixnm = self.spin0.value()
        max_std_dev = self.spin1.value()
        iteration = int(self.spin2.value())
        increment = self.spin3.value()

        self.lim_freq = 1 / (2 * pixnm)

        im_array = self.image_current
        all_FRC = []
        all_n = []
        SNR_good = []

        for k in range(iteration):
            std_dev = 1
            found_intersections = False
            inner_counter = 0

            while std_dev <= max_std_dev:
                inner_counter += 1
                if inner_counter > 1000:
                    print(f"Iteration {k} aborted: too many inner loops.")
                    break

                print(f"Iteration n°{k} with std_dev = {std_dev:.3f}")
                noisy_image = self.add_gaussian_noise(im_array, mean=0, std_dev=std_dev)
                sub_image1, sub_image2 = self.sous_images(noisy_image)
                FRC, n = self.frc(sub_image1, sub_image2)

                smoothed_FRC = savgol_filter(FRC, window_length=7, polyorder=1)
                freqs = np.linspace(0, self.lim_freq, len(smoothed_FRC))

                snr_min, snr_max, snr, roots = self.solutions_snr_polynomial(
                    n, smoothed_FRC
                )
                # print(snr_min, snr_max, snr, roots)
                intersections = self.find_intersections_with_interpolation(
                    freqs, smoothed_FRC, snr
                )

                if intersections:
                    found_intersections = True
                    print(f"Intersections found at iteration {k}: {intersections}")
                    all_FRC.append(smoothed_FRC)
                    all_n.append(n)
                    SNR_good.append(snr)
                    break
                else:
                    std_dev += increment
                    print(f"No intersection, increasing std_dev to {std_dev:.3f}")
            if not found_intersections:
                print(
                    f"No intersections found for iteration {k} after reaching max std_dev."
                )

        if not found_intersections:
            print(f"No intersections found after all iterations.")

        all_FRC = np.array(all_FRC)
        all_n = np.array(all_n)
        mean_FRC = np.mean(all_FRC, axis=0)
        mean_n = np.mean(all_n, axis=0)

        SNR = np.mean(SNR_good, axis=0)

        snr_min_frc, snr_max_frc, snr_mean_frc, roots_frc = (
            self.solutions_snr_polynomial(mean_n, mean_FRC)
        )

        h = [0.4142] * len(mean_FRC)
        half_bit = self.threshold(mean_n, h)

        x_h = self.find_intersections_with_interpolation(freqs, mean_FRC, half_bit)
        x_snr = self.find_intersections_with_interpolation(
            freqs, mean_FRC, snr_mean_frc
        )

        # Store for later if needed
        self.all_FRC = np.array(all_FRC)
        self.mean_FRC = np.mean(self.all_FRC, axis=0)
        self.freqs = np.linspace(0, self.lim_freq, len(self.mean_FRC))
        self.snr_mean_frc = snr_mean_frc
        self.half_bit = half_bit
        self.x_h = x_h
        self.x_snr = x_snr
        self.SNR_good = SNR_good
        # self.lim_freq = lim_freq

        # Now plot using the external method
        self.plot_the_results()

    # ------------------ GUI Class ------------------
    # def plot_the_results(self, all_FRC, lim_freq, mean_FRC, snr_mean_frc, half_bit):
    def plot_the_results(self):
        plt.rcParams.update(
            {
                "font.size": 8,
                "axes.titlesize": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
            }
        )

        # Use the existing axes: self.frc_ax and self.frc_twin_ax
        # Assuming self.frc_ax and self.frc_twin_ax are initialized elsewhere
        self.frc_figure.delaxes(self.frc_twin_ax)  # remove old twin axis
        self.frc_ax.clear()
        self.frc_twin_ax = self.frc_ax.twiny()
        frc_ax = self.frc_ax
        frc_twin_ax = self.frc_twin_ax

        # Plot FRC curves
        for FRC in self.all_FRC:
            frc_ax.plot(
                np.linspace(0, self.lim_freq, len(self.mean_FRC)),
                self.mean_FRC,
                color="gray",
                alpha=0.05,
            )

        # Plot smoothed mean FRC and thresholds
        freqs = np.linspace(0, self.lim_freq, len(self.mean_FRC))
        frc_ax.plot(self.freqs, self.mean_FRC, "k-", label="FRC", linewidth=3)
        frc_ax.plot(
            self.freqs,
            self.snr_mean_frc,
            color="blue",
            linestyle="-",
            label="SNR Threshold",
            linewidth=3,
        )
        frc_ax.plot(
            self.freqs,
            self.half_bit,
            color="red",
            linestyle="-.",
            label="Half-bit Threshold",
            linewidth=3,
        )

        # Set limits and labels
        frc_ax.set_xlim([0, self.lim_freq])
        frc_ax.set_ylim([0, 1.05])
        frc_ax.set_xlabel("Spatial Frequency (1/nm)")
        frc_ax.set_ylabel("FRC")
        frc_ax.legend(loc="best")

        # Second axis for spatial resolution
        frc_twin_ax.set_xlim(frc_ax.get_xlim())
        freq_ticks = frc_ax.get_xticks()

        if freq_ticks[-1] > self.lim_freq:
            freq_ticks = np.delete(freq_ticks, -1)

        spatial_resolution_labels = []
        for freq in freq_ticks:
            if freq > 0:
                resolution = 1 / freq
                spatial_resolution_labels.append(f"{resolution:.1f} nm")
            else:
                spatial_resolution_labels.append("")  # No label for freq = 0

        frc_twin_ax.set_xticks(freq_ticks)
        frc_twin_ax.set_xticklabels(spatial_resolution_labels)
        frc_twin_ax.set_xlabel("Spatial Resolution (nm)")

        # Annotations for intersections and uncertainty calculations
        if self.x_h is not None:
            frc_ax.axvline(
                x=self.x_h,
                color="red",
                linestyle=":",
                label=f"Intersection at {self.x_h:.3f} nm⁻¹",
            )
            if self.x_snr is not None:
                delta = self.x_h - self.x_snr
                if delta < 0:
                    position = "right"
                else:
                    position = "left"
            else:
                position = "right"
            all_half_bit_value = []
            for i in range(len(self.all_FRC)):
                x_h_V = self.find_intersections_with_interpolation(
                    self.freqs, self.all_FRC[i], self.half_bit
                )  # Intersection FRC and half-bit
                if x_h_V is not None:
                    all_half_bit_value.append(x_h_V)
            all_half_bit_value = np.array(all_half_bit_value)
            std_half_bit = np.std(all_half_bit_value)
            frc_ax.annotate(
                f" {self.x_h:.3f}±{std_half_bit:.3f}nm$^-$$^1$ ",
                (self.x_h, 0),
                textcoords="offset points",
                xytext=(0, 10),
                va="bottom",
                ha=position,
                color="red",
            )

            # Calculate uncertainties for x_h
            _, _, max_uncertainty_x_h = self.calculate_uncertainty(
                self.x_h, std_half_bit
            )

            # Annotate on the graph
            resolution_x_h = 1 / self.x_h if self.x_h != 0 else np.nan
            frc_twin_ax.annotate(
                f" {resolution_x_h:.1f}±{max_uncertainty_x_h:.1f}nm ",
                (self.x_h, 0.95),
                textcoords="offset points",
                xytext=(0, 10),
                va="top",
                ha=position,
                color="red",
            )

        if self.x_snr is not None:
            frc_ax.axvline(
                x=self.x_snr,
                color="blue",
                linestyle="--",
                label=f"Intersection at {self.x_snr:.3f} nm⁻¹",
            )
            if self.x_h is not None:
                delta = self.x_h - self.x_snr
                if delta > 0:
                    position = "right"
                else:
                    position = "left"
            else:
                position = "left"
            all_SNR_bit_value = []
            for i in range(len(self.all_FRC)):
                x_snr_V = self.find_intersections_with_interpolation(
                    self.freqs, self.all_FRC[i], self.SNR_good[i]
                )  # Intersection FRC and SNR threshold
                if x_snr_V is not None:
                    all_SNR_bit_value.append(x_snr_V)
            all_SNR_bit_value = np.array(all_SNR_bit_value)
            std_SNR_bit = np.std(all_SNR_bit_value)

            frc_ax.annotate(
                f" {self.x_snr:.3f}±{std_SNR_bit:.3f}nm$^-$$^1$ ",
                (self.x_snr, 0),
                textcoords="offset points",
                xytext=(0, 10),
                va="bottom",
                ha=position,
                color="blue",
            )

            # Calculate uncertainties for x_snr
            uncertainty_plus_x_snr, uncertainty_minus_x_snr, max_uncertainty_x_snr = (
                self.calculate_uncertainty(self.x_snr, std_SNR_bit)
            )

            # Annotate on the graph
            resolution_x_snr = 1 / self.x_snr if self.x_snr != 0 else np.nan
            frc_twin_ax.annotate(
                f" {resolution_x_snr:.1f}±{max_uncertainty_x_snr:.1f}nm ",
                (self.x_snr, 0.95),
                textcoords="offset points",
                xytext=(0, 10),
                va="top",
                ha=position,
                color="blue",
            )

        self.frc_canvas.draw()

    def save_screengrab(self):
        # Prompt user for file path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "screenshot.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )

        if file_path:
            # Grab the entire main window (or replace `self` with a specific widget)
            screenshot = self.grab()
            screenshot.save(file_path)


# ------------------ Main ------------------

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    gui = FRCViewer()
    gui.resize(1000, 600)
    gui.show()
    sys.exit(app.exec())
