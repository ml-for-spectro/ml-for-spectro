import logging
from PySide6.QtWidgets import QTextEdit


class QTextEditLogger(logging.Handler):
    def __init__(self, widget: QTextEdit):
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    def emit(self, record):
        msg = self.format(record)
        # Ensure GUI thread safety
        self.widget.appendPlainText(msg)
