from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


photon_energy_eV = 300.0  # default


def ke_to_be(ke):
    return photon_energy_eV - ke


def be_to_ke(be):
    return photon_energy_eV - be


def on_mouse_move(event, label):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        label.setText(f"X: {x:.2f}, Y: {y:.2f}")


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, coord_label=None):
        self.fig, self.ax1 = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax2 = self.ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))
        self.coord_label = coord_label

        self.mpl_connect(
            "motion_notify_event", lambda event: on_mouse_move(event, self.coord_label)
        )

    def plot_data(self, x, y, label="Spectrum", color="black"):
        self.ax1.clear()
        self.ax2.clear()  # Also clear secondary axis
        self.ax2 = self.ax1.secondary_xaxis("top", functions=(be_to_ke, ke_to_be))
        self.ax1.plot(x, y, label=label, color=color)
        self.ax1.set_xlabel("Binding Energy (eV)")
        self.ax1.set_ylabel("Intensity (a.u.)")
        self.ax1.legend()

        self.ax2.set_xlabel("Kinetic Energy (eV)")
        self.draw()
