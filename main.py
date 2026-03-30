import sys

from PySide6.QtWidgets import QApplication

from logging_setup import setup_logging
from ui import MainWindow


if __name__ == "__main__":
    setup_logging()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
