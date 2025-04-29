from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit


class LogTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        # self.log_path = log_path

        layout = QVBoxLayout()
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

        self.update_log_view()

    def update_log_view(self):
        try:
            with open(self.log_path, "r") as f:
                self.text_edit.setPlainText(f.read())
        except Exception as e:
            self.text_edit.setPlainText(f"Failed to read log:\n{e}")
