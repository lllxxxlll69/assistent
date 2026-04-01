import sys

from PySide6.QtWidgets import QApplication

from logging_setup import setup_logging
from ui import MainWindow


def main():
    setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("Local PC Assistant")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()