import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from windows.start_window import StartWindow

# HARRY: Global reference to track active windows that may have training threads
active_windows = []

if __name__ == "__main__":  
    app = QApplication(sys.argv)
    
    # HARRY: Enhanced cleanup on global app exit
    def on_exit():
        # HARRY: First stop all training threads
        for window in active_windows[:]:  # Use slice copy to avoid modification during iteration
            if hasattr(window, 'cleanup_training_thread'):
                window.cleanup_training_thread()
            if hasattr(window, 'nn_designer_window') and window.nn_designer_window:
                window.nn_designer_window.cleanup_training_thread()
            active_windows.remove(window)
    
    # HARRY: Function to manage window registration
    def register_window(window):
        active_windows.append(window)
        
    def unregister_window(window):
        if window in active_windows:
            active_windows.remove(window)
    
    # HARRY: Make window management functions available globally
    app.register_window = register_window
    app.unregister_window = unregister_window
    
    window = StartWindow()
    app.register_window(window)
    window.showMaximized()
    
    app.aboutToQuit.connect(on_exit)
    sys.exit(app.exec_())