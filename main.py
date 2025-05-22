import sys
from PyQt5.QtWidgets import QApplication
from windows.start_window import StartWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartWindow()
    window.showMaximized()
    sys.exit(app.exec_())
