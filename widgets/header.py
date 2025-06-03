from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap
import windows.progress_state as progress_state

class ClickableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, 'on_click'):
            self.on_click()

class Header(QWidget):
    tab_changed = pyqtSignal(str)  
    
    def __init__(self, active_page="Start Window", parent_window=None):
        super().__init__()
        self.parent_window = parent_window
        self.active_page = active_page
        self.init_ui()
        
    def init_ui(self):
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#626262"))  
        self.setPalette(palette)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 10, 20, 10)  
        
        # Add logo on the left
        logo = ClickableLabel()
        logo.setPixmap(QPixmap("assets/logo_exolab.png").scaled(90, 90, Qt.KeepAspectRatio))  
        logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo.setCursor(Qt.PointingHandCursor)
        logo.on_click = self.on_logo_clicked  
        main_layout.addWidget(logo)
    
        
        # Add spacer to push tabs to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        main_layout.addItem(spacer)
        
        # Create tabs container with horizontal layout
        tabs_container = QWidget()
        tabs_layout = QHBoxLayout()
        tabs_layout.setSpacing(30)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setAlignment(Qt.AlignCenter)
        
        # Create tabs with white text for better contrast
        self.tabs = {
            "Dataset Builder": self.create_tab("Dataset Builder"),
            "Neural Network Designer": self.create_tab("Neural Network Designer"),
            "Neural Network Evaluator": self.create_tab("Neural Network Evaluator")
        }
        
        # Add tabs to tabs layout
        for tab in self.tabs.values():
            tabs_layout.addWidget(tab)
        
        tabs_container.setLayout(tabs_layout)
        main_layout.addWidget(tabs_container)
        
        self.setLayout(main_layout)
        self.update_active_tab()

    def on_logo_clicked(self):
        # Demande de confirmation avant retour au StartWindow
        reply = QMessageBox.question(
            self,
            "Confirm Return",
            "Are you sure you want to return to the start? All unsaved progress will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return  # Annule le retour

        if self.parent_window:
            parent = self.parent_window.window()
            # HARRY: Cleanup training thread before closing
            if hasattr(parent, "cleanup_training_thread"):
                parent.cleanup_training_thread()

            from windows.start_window import StartWindow
            start_window = StartWindow()
            start_window.showMaximized()

            self.parent_window.window().close()
        
    def create_tab(self, name):
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        tab_label = QLabel(name)
        tab_label.setFont(QFont("Arial", 16, QFont.Bold))
        tab_label.setAlignment(Qt.AlignCenter)
        tab_label.setStyleSheet("color: white; padding: 5px 10px;")
        
        container_layout.addWidget(tab_label)
        
        if name == self.active_page:
            # Add a horizontal bar for the active page
            underline = QFrame()
            underline.setFrameShape(QFrame.HLine)
            underline.setLineWidth(2)
            underline.setFixedHeight(3)
            underline.setStyleSheet("background-color: white; border: none;")
            container_layout.addWidget(underline)
        else:
            # Make the tab clickable if it's not the active page
            tab_label.mousePressEvent = lambda event, name=name: self.on_tab_clicked(name)
            tab_label.setCursor(Qt.PointingHandCursor)
        
        container.setFixedHeight(tab_label.sizeHint().height() + 5)
        return container
    
    def set_tabs_enabled(self, enabled: bool):
        """Enable or disable tabs based on the training state."""
        for tab in self.tabs.values():
            tab_label = tab.layout().itemAt(0).widget()
            tab_label.setEnabled(enabled)

    def on_tab_clicked(self, tab_name):
        if hasattr(self.parent_window, "is_training") and self.parent_window.is_training:
            QMessageBox.warning(self, "Training in progress", "Please wait for training to finish before navigating.")
            return
        current_window = self.parent_window.window()
        if hasattr(current_window, "train_thread") and current_window.train_thread is not None:
            if current_window.train_thread.isRunning():
                current_window.train_thread.quit()
                current_window.train_thread.wait()
        
        current_window = self.parent_window.window()
        # HARRY: Cleanup training thread before closing
        if hasattr(current_window, "cleanup_training_thread"):
            current_window.cleanup_training_thread()
        
        # Get COMPLETE current state
        saved_state = current_window.get_saved_state() if hasattr(current_window, "get_saved_state") else {}
        
        if tab_name == "Neural Network Designer" and not progress_state.dataset_built:
            QMessageBox.warning(self, "Access Denied", "You must complete Dataset Builder first")
            return
            
        new_window = None
        if tab_name == "Dataset Builder":
            from windows.dataset_builder_window import DatasetBuilderWindow
            new_window = DatasetBuilderWindow(saved_state=saved_state)
        elif tab_name == "Neural Network Designer":
            from windows.nn_designer_window import NeuralNetworkDesignerWindow
            new_window = NeuralNetworkDesignerWindow(
                dataset_path=saved_state.get("dataset_path"),
                saved_state=saved_state
            )
        elif tab_name == "Neural Network Evaluator":
            if not progress_state.training_started:
                QMessageBox.warning(self, "Warning", "You must complete training first")
                return
            if not hasattr(self.parent_window, "nn_evaluator_window") or self.parent_window.nn_evaluator_window is None:
                from windows.nn_evaluator_window import NeuralNetworkEvaluator
                self.parent_window.nn_evaluator_window = NeuralNetworkEvaluator(saved_state=saved_state)
                self.parent_window.nn_evaluator_window.parent_window = self.parent_window
            new_window = self.parent_window.nn_evaluator_window
        else:
            QMessageBox.warning(self, "Error", f"Unsupported tab: {tab_name}")
            return

        if new_window:
            new_window.showMaximized()
            current_window.close()

    def update_active_tab(self):
        for name, tab in self.tabs.items():
            tab_label = tab.layout().itemAt(0).widget()
            
            if name == self.active_page:
                tab_label.setStyleSheet("color: white; padding: 5px 10px; font-weight: bold;")  # Keep bold style for active tab
            else:
                tab_label.setStyleSheet("color: white; padding: 5px 10px;")

    def show_warning(self, message):
        QMessageBox.warning(self, "Access Denied", message)