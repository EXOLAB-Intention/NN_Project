from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap
import windows.progress_state as progress_state

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
        logo = QLabel()
        logo.setPixmap(QPixmap("assets/logo_exolab.png").scaled(90, 90, Qt.KeepAspectRatio))  
        logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
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
    
    def on_tab_clicked(self, tab_name):
        if tab_name == "Neural Network Designer" and not progress_state.dataset_built:
            self.show_warning("You must complete 'Dataset Builder' first.")
            return
        if tab_name == "Neural Network Evaluator" and not (progress_state.dataset_built and progress_state.nn_designed):
            self.show_warning("You must complete both 'Dataset Builder' and 'Neural Network Designer' first.")
            return

        self.tab_changed.emit(tab_name)

        if self.parent_window:
            current_window = self.parent_window.window()

            if tab_name == "Dataset Builder":
                from windows.dataset_builder_window import DatasetBuilderWindow
                new_window = DatasetBuilderWindow()
            elif tab_name == "Neural Network Designer":
                from windows.nn_designer_window import NeuralNetworkDesignerWindow
                new_window = NeuralNetworkDesignerWindow()
            elif tab_name == "Neural Network Evaluator":
                from windows.nn_evaluator_window import NeuralNetworkEvaluator
                new_window = NeuralNetworkEvaluator()

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