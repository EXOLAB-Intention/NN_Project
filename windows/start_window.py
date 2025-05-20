import os
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFileDialog, QAction, QListWidgetItem
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QSize, Qt

from windows.add_data_window import AddDataWindow
from windows.dataset_builder_window import DatasetBuilderWindow
from windows.nn_designer_window import NeuralNetworkDesignerWindow  
from widgets.header import Header


class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(100, 100, 600, 400)

        self.recent_folders = self.load_recent_folders()

        self.init_ui()

        # Add a centered status bar
        status_bar = self.statusBar()
        status_label = QLabel("Data Monitoring Software version 1.0.13")
        status_label.setAlignment(Qt.AlignCenter)  
        status_bar.addPermanentWidget(status_label, 1)  

    def init_ui(self):
        # Central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with tabs
        header = Header(active_page="Start Window", parent_window=self)
        layout.addWidget(header)

        # Title "Start" with larger font
        title_start = QLabel("Start")
        title_start.setFont(QFont("Arial", 40, QFont.Bold))
        title_start.setContentsMargins(0, 20, 0, 0)  
        layout.addWidget(title_start)

        # Horizontal layout for buttons
        btn_layout = QHBoxLayout()
    
        btn_new = QPushButton("New Dataset…")
        btn_new.setIcon(QIcon("assets/new_data.png")) 

        btn_open = QPushButton("Open Dataset…")
        btn_open.setIcon(QIcon("assets/open_data.png"))

        btn_new.setIconSize(QSize(36, 36))  
        btn_open.setIconSize(QSize(38, 38))

        btn_new.setFixedWidth(500)
        btn_open.setFixedWidth(500)

        btn_new.setStyleSheet("padding-top: 10px; padding-bottom: 10px;")
        btn_open.setStyleSheet("padding-top: 10px; padding-bottom: 10px;")

        btn_new.clicked.connect(self.create_new_dataset)
        btn_open.clicked.connect(self.open_dataset)

        btn_layout.addWidget(btn_new)
        btn_layout.addWidget(btn_open)

        layout.addLayout(btn_layout)

        # Title "Recent" with larger font
        title_recent = QLabel("Recent")
        title_recent.setFont(QFont("Arial", 40, QFont.Bold))
        title_recent.setContentsMargins(0, 20, 0, 0)  
        layout.addWidget(title_recent)

        # Recent folders list
        self.recent_folders_list = QListWidget()
        self.recent_folders_list.setStyleSheet("border: 1px solid #dcdcdc; background-color: white;")
        self.recent_folders_list.itemDoubleClicked.connect(self.open_recent_folder)
        layout.addWidget(self.recent_folders_list)

        self.update_recent_folders_list()

        # Set layout
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Active actions
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
        """Open the DatasetBuilderWindow."""
        self.dataset_builder = DatasetBuilderWindow(start_window_ref=self)  # Pass self as a reference
        self.dataset_builder.showMaximized()
        self.hide()  # Hide the StartWindow instead of closing it

    def open_dataset(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder_path:
            self.add_to_recent_folders(folder_path)
            self.open_neural_network_designer(folder_path)

    def open_recent_folder(self, item):
        folder_path = item.text()
        if os.path.exists(folder_path):
            self.open_neural_network_designer(folder_path)
        else:
            self.recent_folders.remove(folder_path)
            self.save_recent_folders()
            self.update_recent_folders_list()

    def open_neural_network_designer(self, folder_path):
        self.hide()  # Use hide instead of close to keep the application running
        self.nn_designer_window = NeuralNetworkDesignerWindow(dataset_path=folder_path)
        self.nn_designer_window.showMaximized()           


    def load_recent_folders(self):
        recent_file = "recent_folders.txt"
        if os.path.exists(recent_file):
            with open(recent_file, "r") as file:
                return [line.strip() for line in file.readlines()]
        return []

    def save_recent_folders(self):
        recent_file = "recent_folders.txt"
        with open(recent_file, "w") as file:
            file.writelines(f"{folder}\n" for folder in self.recent_folders)

    def update_recent_folders_list(self):
        self.recent_folders_list.clear()
        for folder in self.recent_folders:
            item = QListWidgetItem(folder)
            self.recent_folders_list.addItem(item)

    def add_to_recent_folders(self, folder):
        if folder in self.recent_folders:
            self.recent_folders.remove(folder)
        self.recent_folders.insert(0, folder)
        self.recent_folders = self.recent_folders[:10]
        self.save_recent_folders()
        self.update_recent_folders_list()

    def load_existing_model(self):
        print("Load existing model...")

