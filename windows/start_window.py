import os
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFileDialog, QAction, QListWidgetItem, QMessageBox
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QSize, Qt
from tensorflow.keras.models import load_model

from windows.dataset_builder_window import DatasetBuilderWindow
from windows.nn_designer_window import NeuralNetworkDesignerWindow  
from windows.nn_evaluator_window import NeuralNetworkEvaluator
from widgets.header import Header
import windows.progress_state as progress_state

class StartWindow(QMainWindow):
    def __init__(self):
        """
        Initialize the StartWindow, set up the UI and load recent folders.
        """
        super().__init__()
        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(100, 100, 600, 400)

        # Load recent folders from file
        self.recent_folders = self.load_recent_folders()

        # Build the UI
        self.init_ui()

        # Add a centered status bar at the bottom
        status_bar = self.statusBar()
        status_label = QLabel("Data Monitoring Software version 1.0.13")
        status_label.setAlignment(Qt.AlignCenter)  
        status_bar.addPermanentWidget(status_label, 1)  

    def init_ui(self):
        """
        Set up the main UI components: header, titles, buttons, recent folders list, and menu bar.
        """
        # Central widget and main layout (vertical)
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Add the custom header with navigation tabs
        header = Header(active_page="Start Window", parent_window=self)
        layout.addWidget(header)

        # Add the main title "Start"
        title_start = QLabel("Start")
        title_start.setFont(QFont("Arial", 42, QFont.Bold))
        title_start.setContentsMargins(10, 20, 0, 0)  
        layout.addWidget(title_start)

        # Add horizontal layout with "New Dataset" and "Open Dataset" buttons
        btn_layout = QHBoxLayout()
    
        btn_new = QPushButton("New Dataset…")
        btn_new.setIcon(QIcon("assets/new_data.png")) 

        btn_open = QPushButton("Open Dataset…")
        btn_open.setIcon(QIcon("assets/open_data.png"))

        btn_load = QPushButton("Load Model...")
        btn_load.setIcon(QIcon("assets/load.png"))

        btn_new.setIconSize(QSize(36, 36))  
        btn_open.setIconSize(QSize(38, 38))
        btn_load.setIconSize(QSize(33, 33))

        btn_new.setFixedWidth(450)
        btn_open.setFixedWidth(450)
        btn_load.setFixedWidth(450)

        btn_new.setStyleSheet("padding-top: 10px; padding-bottom: 10px;")
        btn_open.setStyleSheet("padding-top: 10px; padding-bottom: 10px;")
        btn_load.setStyleSheet("padding-top: 10px; padding-bottom: 10px;")

        btn_new.clicked.connect(self.create_new_dataset)
        btn_open.clicked.connect(self.open_dataset)
        btn_load.clicked.connect(self.load_existing_model)

        btn_layout.addWidget(btn_new)
        btn_layout.addWidget(btn_open)
        btn_layout.addWidget(btn_load)

        layout.addLayout(btn_layout)

        # Add the "Recent" title
        title_recent = QLabel("Recent")
        title_recent.setFont(QFont("Arial", 40, QFont.Bold))
        title_recent.setContentsMargins(10, 20, 0, 0)  
        layout.addWidget(title_recent)

        # Add the recent folders list widget
        self.recent_folders_list = QListWidget()
        self.recent_folders_list.setStyleSheet("border: 1px solid #dcdcdc; background-color: white;")
        self.recent_folders_list.itemDoubleClicked.connect(self.open_recent_folder)
        layout.addWidget(self.recent_folders_list)

        # Populate the recent folders list
        self.update_recent_folders_list()

        # Set the layout on the central widget
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Create the menu bar and "File" menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Add actions to the File menu
        action_create = QAction("Create new dataset", self)
        action_create.triggered.connect(self.create_new_dataset)

        action_load_dataset = QAction("Load existing dataset", self)
        action_load_dataset.triggered.connect(self.open_dataset)

        action_load_model = QAction("Load existing Model", self)
        action_load_model.triggered.connect(self.load_existing_model)

        file_menu.addAction(action_create)
        file_menu.addAction(action_load_dataset)
        file_menu.addAction(action_load_model)

    def create_new_dataset(self):
        """
        Open the DatasetBuilderWindow for creating a new dataset.
        """
        progress_state.dataset_built = False  # Reset progress state
        progress_state.nn_designed = False
        progress_state.training_started = False  # Reset training state
        self.set_tabs_enabled(False)  # Disable tabs during dataset creation
        self.dataset_builder = DatasetBuilderWindow(start_window_ref=self)  
        self.dataset_builder.showMaximized()
        self.hide()
        self.set_tabs_enabled(True)  # Re-enable tabs after dataset creation

    def open_dataset(self):
        """
        Open a dialog to select a dataset folder, add it to recents, and open it in NeuralNetworkDesignerWindow.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder_path:
            self.add_to_recent_folders(folder_path)
            self.open_neural_network_designer(folder_path)

    def open_recent_folder(self, item):
        """
        Open a recent folder in NeuralNetworkDesignerWindow if it exists, otherwise remove it from recents.
        """
        folder_path = item.text()
        if os.path.exists(folder_path):
            self.open_neural_network_designer(folder_path)
        else:
            self.recent_folders.remove(folder_path)
            self.save_recent_folders()
            self.update_recent_folders_list()

    def open_neural_network_designer(self, folder_path):
        """
        Open the NeuralNetworkDesignerWindow with the given dataset folder.
        """
        self.hide()  
        progress_state.dataset_built = True 
        self.nn_designer_window = NeuralNetworkDesignerWindow(
            dataset_path=folder_path,
            saved_state={
                "dataset_path": folder_path,
                "selected_folder": folder_path,  
                "filtered_files": [f for f in os.listdir(folder_path) if f.endswith('.h5')]  
            }
        )
        self.nn_designer_window.populate_file_list()  
        self.nn_designer_window.showMaximized()

    def load_recent_folders(self):
        """
        Load the list of recent folders from a text file.
        """
        recent_file = "recent_folders.txt"
        if os.path.exists(recent_file):
            with open(recent_file, "r") as file:
                return [line.strip() for line in file.readlines()]
        return []

    def save_recent_folders(self):
        """
        Save the current list of recent folders to a text file.
        """
        recent_file = "recent_folders.txt"
        with open(recent_file, "w") as file:
            file.writelines(f"{folder}\n" for folder in self.recent_folders)

    def update_recent_folders_list(self):
        """
        Update the QListWidget to display the current recent folders with increased font size and padding.
        """
        self.recent_folders_list.clear()
        self.recent_folders_list.setStyleSheet("""
            QListWidget::item {
                padding: 5px 30px;  
            }
        """)
        for folder in self.recent_folders:
            item = QListWidgetItem(folder)
            font = QFont("Arial", 16)  
            item.setFont(font)
            self.recent_folders_list.addItem(item)

    def add_to_recent_folders(self, folder):
        """
        Add a folder to the top of the recent folders list, ensuring no duplicates and a maximum of 10 items.
        """
        if folder in self.recent_folders:
            self.recent_folders.remove(folder)
        self.recent_folders.insert(0, folder)
        self.recent_folders = self.recent_folders[:10]
        self.save_recent_folders()
        self.update_recent_folders_list()

    def load_existing_model(self):
        """Load a previously trained model and open it in the NN Evaluator window."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "H5 Files (*.h5)")
        if not file_path:
            return

        try:
            self.set_tabs_enabled(False)  # Disable tabs during model loading
            model = load_model(file_path)
            progress_state.nn_designed = True
            self.nn_evaluator_window = NeuralNetworkEvaluator()
            self.nn_evaluator_window.model = model
            self.nn_evaluator_window.showMaximized()
            self.hide()
            self.set_tabs_enabled(True)  # Re-enable tabs after model loading
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")

    def set_tabs_enabled(self, enabled: bool):
        """Enable or disable tabs in the header."""
        if hasattr(self, "header"):
            self.header.set_tabs_enabled(enabled)

