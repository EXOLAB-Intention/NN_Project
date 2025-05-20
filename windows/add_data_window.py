from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QFileDialog, QHeaderView
)
from PyQt5.QtCore import Qt


class AddDataWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Add Data")
        self.setGeometry(200, 200, 800, 400)

        # Main layout
        layout = QVBoxLayout()

        # Folder address and search bar
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Folder address")
        self.folder_search = QLineEdit()
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.open_folder_dialog)
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_search)
        folder_layout.addWidget(search_button)
        layout.addLayout(folder_layout)

        # Filters (Motion, Subject Type, Sex, Sensor)
        filter_layout = QHBoxLayout()
        motion_label = QLabel("Motion")
        self.motion_dropdown = QComboBox()
        self.motion_dropdown.addItems(["Select the model...", "Level walking", "Sit-to-stand", "Stair ascending"])

        subject_label = QLabel("Subject Type")
        self.subject_dropdown = QComboBox()
        self.subject_dropdown.addItems(["Select the model...", "Adult", "Child"])

        sex_label = QLabel("Sex")
        self.sex_dropdown = QComboBox()
        self.sex_dropdown.addItems(["Select the model...", "Male", "Female"])

        sensor_label = QLabel("Sensor")
        self.sensor_dropdown = QComboBox()
        self.sensor_dropdown.addItems(["Select the model...", "Sensor A", "Sensor B", "Sensor C"])

        filter_layout.addWidget(motion_label)
        filter_layout.addWidget(self.motion_dropdown)
        filter_layout.addWidget(subject_label)
        filter_layout.addWidget(self.subject_dropdown)
        filter_layout.addWidget(sex_label)
        filter_layout.addWidget(self.sex_dropdown)
        filter_layout.addWidget(sensor_label)
        filter_layout.addWidget(self.sensor_dropdown)
        layout.addLayout(filter_layout)

        # Table for data display
        self.table = QTableWidget()
        self.table.setRowCount(5)  # Example row count
        self.table.setColumnCount(5)  # Updated to include the "Sensor" column
        self.table.setHorizontalHeaderLabels(["File name", "Motion", "Subject Type", "Sex", "Sensor"])

        # Make the table stretch to fill the available space
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Example data
        self.populate_table([
            ["250501_walkingtest1.h5", "Level walking", "Adult", "Male", "Sensor A"],
            ["250501_walkingtest2.h5", "Sit-to-stand", "Child", "Female", "Sensor B"],
            ["250501_walkingtest3.h5", "Stair ascending", "Adult", "Male", "Sensor C"],
            ["250501_walkingtest4.h5", "Level walking", "Child", "Female", "Sensor A"],
        ])

        layout.addWidget(self.table)

        # Add button
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_selected_files)
        layout.addWidget(add_button, alignment=Qt.AlignRight)

        self.setLayout(layout)

    def open_folder_dialog(self):
        """Open a folder selection dialog and update the folder address field."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_search.setText(folder_path)

            # List and populate files
            import os
            files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
            data = [[f, "", "", "", ""] for f in files]
            self.populate_table(data)


    def populate_table(self, data):
        """Populate the table with data, sorted by numbers in file names."""
        def extract_number(file_name):
            """Extract the first number from a file name."""
            import re
            matches = re.findall(r'\d+', file_name)
            return int(matches[0]) if matches else 0

        # Sort the data by the extracted number from the file name
        data.sort(key=lambda row: extract_number(row[0]))

        # Populate the table
        self.table.setRowCount(len(data))
        for row, row_data in enumerate(data):
            for col, value in enumerate(row_data):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def add_selected_files(self):
        """Handle the Add button click."""
        self.selected_files = []
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0):  # Check if the row has data
                self.selected_files.append(self.table.item(row, 0).text())
        print(f"Selected files: {self.selected_files}")
        self.accept()  # Close the dialog
