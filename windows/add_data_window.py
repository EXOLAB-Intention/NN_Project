from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QFileDialog, QHeaderView
)
from PyQt5.QtCore import Qt


class AddDataWindow(QDialog):
    """
    A dialog window for adding data files to the dataset. It allows users to:
    - Select a folder containing data files
    - Filter files by various criteria (motion, subject type, sex, sensor)
    - View and select files in a table
    - Add selected files to the dataset
    """
    
    def __init__(self):
        """Initialize the Add Data window with UI components."""
        super().__init__()
        self.setWindowTitle("Add Data")
        self.setGeometry(200, 200, 800, 400)
        
        # Initialize selected files list
        self.selected_files = []
        
        # Create and set up the main layout
        self.init_ui()

    def init_ui(self):
        """Initialize all UI components."""
        main_layout = QVBoxLayout()
        
        # Create folder selection controls
        self.create_folder_selection(main_layout)
        
        # Create filter controls
        self.create_filters(main_layout)
        
        # Create the data table
        self.create_data_table(main_layout)
        
        # Create the add button
        self.create_add_button(main_layout)
        
        self.setLayout(main_layout)

    def create_folder_selection(self, parent_layout):
        """
        Create the folder selection controls.
        
        Args:
            parent_layout: The layout to add these controls to
        """
        folder_layout = QHBoxLayout()
        
        # Folder address label
        folder_label = QLabel("Folder address")
        
        # Folder path display
        self.folder_search = QLineEdit()
        self.folder_search.setPlaceholderText("Select a folder containing data files")
        
        # Search button
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.open_folder_dialog)
        
        # Add widgets to layout
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_search)
        folder_layout.addWidget(search_button)
        
        parent_layout.addLayout(folder_layout)

    def create_filters(self, parent_layout):
        """
        Create the filter dropdown controls.
        
        Args:
            parent_layout: The layout to add these controls to
        """
        filter_layout = QHBoxLayout()
        
        # Motion filter
        motion_label = QLabel("Motion")
        self.motion_dropdown = QComboBox()
        self.motion_dropdown.addItems([
            "Select the model...", 
            "Level walking", 
            "Sit-to-stand", 
            "Stair ascending"
        ])

        # Subject type filter
        subject_label = QLabel("Subject Type")
        self.subject_dropdown = QComboBox()
        self.subject_dropdown.addItems([
            "Select the model...", 
            "Adult", 
            "Child"
        ])

        # Sex filter
        sex_label = QLabel("Sex")
        self.sex_dropdown = QComboBox()
        self.sex_dropdown.addItems([
            "Select the model...", 
            "Male", 
            "Female"
        ])

        # Sensor filter
        sensor_label = QLabel("Sensor")
        self.sensor_dropdown = QComboBox()
        self.sensor_dropdown.addItems([
            "Select the model...", 
            "Sensor A", 
            "Sensor B", 
            "Sensor C"
        ])

        # Add all filter widgets to layout
        filter_layout.addWidget(motion_label)
        filter_layout.addWidget(self.motion_dropdown)
        filter_layout.addWidget(subject_label)
        filter_layout.addWidget(self.subject_dropdown)
        filter_layout.addWidget(sex_label)
        filter_layout.addWidget(self.sex_dropdown)
        filter_layout.addWidget(sensor_label)
        filter_layout.addWidget(self.sensor_dropdown)
        
        parent_layout.addLayout(filter_layout)

    def create_data_table(self, parent_layout):
        """
        Create and configure the data table.
        
        Args:
            parent_layout: The layout to add the table to
        """
        self.table = QTableWidget()
        
        # Initial table dimensions
        self.table.setRowCount(5)  # Example row count
        self.table.setColumnCount(5)  # Columns for all metadata
        
        # Set column headers
        self.table.setHorizontalHeaderLabels([
            "File name", 
            "Motion", 
            "Subject Type", 
            "Sex", 
            "Sensor"
        ])

        # Configure table sizing behavior
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        parent_layout.addWidget(self.table)

    def create_add_button(self, parent_layout):
        """
        Create the Add button.
        
        Args:
            parent_layout: The layout to add the button to
        """
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_selected_files)
        parent_layout.addWidget(add_button, alignment=Qt.AlignRight)

    def open_folder_dialog(self):
        """
        Open a folder selection dialog and populate the table with files.
        
        When a folder is selected:
        1. Updates the folder path display
        2. Lists all .h5 files in the folder
        3. Populates the table with the found files
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Update folder path display
            self.folder_search.setText(folder_path)

            # List all .h5 files in the folder
            import os
            files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
            
            # Create table data structure (empty metadata columns)
            data = [[f, "", "", "", ""] for f in files]
            
            # Populate the table with the found files
            self.populate_table(data)

    def populate_table(self, data):
        """
        Populate the table with data, sorted by numbers in file names.
        
        Args:
            data (list): A list of lists containing file information
                         Each sublist represents a row: [filename, motion, subject, sex, sensor]
        """
        def extract_number(file_name):
            """
            Helper function to extract the first number from a filename.
            
            Args:
                file_name (str): The filename to process
                
            Returns:
                int: The first number found, or 0 if none found
            """
            import re
            matches = re.findall(r'\d+', file_name)
            return int(matches[0]) if matches else 0

        # Sort data by the numeric part of filenames
        data.sort(key=lambda row: extract_number(row[0]))

        # Set table dimensions
        self.table.setRowCount(len(data))
        
        # Populate each cell
        for row, row_data in enumerate(data):
            for col, value in enumerate(row_data):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def add_selected_files(self):
        """
        Handle the Add button click.
        
        Collects all selected filenames (from first column) and:
        1. Stores them in self.selected_files
        2. Closes the dialog with QDialog.Accepted status
        """
        self.selected_files = []
        
        # Collect all filenames from the first column
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0):  # Check if the row has data
                self.selected_files.append(self.table.item(row, 0).text())
        
        # Close the dialog (returning QDialog.Accepted)
        self.accept()