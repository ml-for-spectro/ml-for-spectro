import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from viewer import ImageViewer
from load_tab import LoadTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ptychography Image GUI")
        self.tabs = QTabWidget()
        self.viewer = ImageViewer()
        self.tabs.addTab(LoadTab(self.viewer), "Load & Correct")
        self.tabs.addTab(self.viewer, "View Stack")
        self.setCentralWidget(self.tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())
