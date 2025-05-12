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

        self.setStyleSheet(
            """
        QPushButton {
            padding: 6px 12px;
            background-color: #4CAF50;
            color: white;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid gray;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
        QLabel {
            padding: 2px;
        }
        QLineEdit, QSpinBox {
            padding: 2px;
            min-width: 80px;
        }
    """
        )


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
