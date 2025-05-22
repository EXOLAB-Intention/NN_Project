from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QProgressDialog,
    QPushButton, QAction, QFileDialog, QDialog, QLineEdit, QListWidget, QListWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QIntValidator, QFont
from windows.add_data_window import AddDataWindow  
from windows.nn_designer_window import NeuralNetworkDesignerWindow
import re
import time
from widgets.header import Header
import windows.progress_state as progress_state

class DatasetBuilderWindow(QMainWindow):
    """
    Main window for the Dataset Builder module of the Neural Network Trainer application.
    Handles file selection, dataset building, and navigation to other modules.
    """
    
    def __init__(self, start_window_ref=None):
        """
        Initialize the Dataset Builder window.
        
        Args:
            start_window_ref (QMainWindow): Reference to the start window for navigation
        """
        super().__init__()
        self.start_window_ref = start_window_ref
        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(100, 100, 1000, 600)

        # Pagination variables
        self.current_page = 0
        self.items_per_page = 100
        self.all_files = []

        # Initialize UI components
        self.init_ui()

    def init_ui(self):
        """Initialize all UI components of the window."""
        self.create_menu_bar()
        self.create_main_layout()
        self.create_header()
        self.create_content_area()
        self.create_status_bar()

    def create_menu_bar(self):
        """Create the menu bar with navigation options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        back_action = QAction("Back to Start", self)
        back_action.triggered.connect(self.go_back_to_start)
        file_menu.addAction(back_action)
        
        # Additional menus (placeholder)
        menubar.addMenu("Edit")
        menubar.addMenu("Options")

    def create_main_layout(self):
        """Create the main layout structure of the window."""
        central_widget = QWidget()
        central_widget.setContentsMargins(0, 0, 0, 0)  
        self.setCentralWidget(central_widget)
        
        # Main vertical layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  
        main_layout.setSpacing(0)  
        central_widget.setLayout(main_layout)
        
        return main_layout

    def create_header(self):
        """Create the application header with progress indicator."""
        main_layout = self.centralWidget().layout()
        header = Header(active_page="Dataset Builder", parent_window=self)
        main_layout.addWidget(header)

    def create_content_area(self):
        """Create the main content area with file selection and input/output panels."""
        main_layout = self.centralWidget().layout()
        
        # Content container with margins
        content_container = QWidget()
        content_container.setContentsMargins(10, 10, 10, 10)
        container_layout = QVBoxLayout(content_container)
        container_layout.setContentsMargins(0, 0, 0, 0)  

        # Add title and progress indicator
        self.create_title_progress_layout(container_layout)
        
        # Main content layout (horizontal)
        content_layout = QHBoxLayout()
        container_layout.addLayout(content_layout)

        # Left panel - File selection
        self.create_file_selection_panel(content_layout)
        
        # Right panel - Input/Output
        self.create_io_panel(content_layout)
        
        # Bottom buttons
        self.create_button_row(container_layout)
        
        main_layout.addWidget(content_container)

    def create_title_progress_layout(self, parent_layout):
        """
        Create the title and progress indicator layout.
        
        Args:
            parent_layout: The layout to which this will be added
        """
        title_and_progress_layout = QHBoxLayout()
        
        # Title label
        title = QLabel("<h1><b>Dataset Builder</b></h1>")
        title_and_progress_layout.addWidget(title)
        
        # Spacer
        title_and_progress_layout.addStretch()
        
        # Progress indicator
        self.top_right_label = QLabel()
        font = QFont("Arial", 10)
        font.setItalic(True)
        self.top_right_label.setFont(font)
        self.top_right_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.top_right_label.setTextFormat(Qt.RichText)
        self.top_right_label.setStyleSheet("""
            background-color: rgba(200, 200, 200, 50);
            padding: 6px 12px;
            border-radius: 8px;
            margin: 5px;
        """)
        self.update_progress_label(active_step="Dataset Builder")
        title_and_progress_layout.addWidget(self.top_right_label)
        
        parent_layout.addLayout(title_and_progress_layout)

    def create_file_selection_panel(self, parent_layout):
        """
        Create the left panel for file selection and management.
        
        Args:
            parent_layout: The layout to which this will be added
        """
        left_panel = QVBoxLayout()
        
        # File selection header with buttons
        file_header = QHBoxLayout()
        file_header.addWidget(QLabel("<b>File Selection</b>"))
        file_header.addStretch()
        
        add_btn = QPushButton("+ Add data")
        delete_btn = QPushButton("- Delete data")
        file_header.addWidget(add_btn)
        file_header.addWidget(delete_btn)
        left_panel.addLayout(file_header)
        
        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_bar = QLineEdit()  
        self.search_bar.setFixedHeight(30)
        self.search_bar.setPlaceholderText("Enter file number...")
        self.search_bar.setValidator(QIntValidator())  
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_bar)
        left_panel.addLayout(search_layout)

        # File list widget
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)  
        left_panel.addWidget(self.file_list)
        
        # Pagination controls
        self.create_pagination_controls(left_panel)
        
        # Connect signals
        add_btn.clicked.connect(self.open_add_data_window)
        delete_btn.clicked.connect(self.delete_data)
        self.search_bar.textChanged.connect(self.filter_file_list)
        
        parent_layout.addLayout(left_panel)

    def create_pagination_controls(self, parent_layout):
        """
        Create pagination controls for navigating through files.
        
        Args:
            parent_layout: The layout to which this will be added
        """
        pagination_layout = QHBoxLayout()
        
        # Previous page button
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_page)
        
        # Next page button
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_page)
        
        # Page input
        self.page_input = QLineEdit()
        self.page_input.setPlaceholderText("Enter page number")
        self.page_input.setFixedWidth(100)
        self.page_input.returnPressed.connect(self.go_to_page)

        # Go button
        go_button = QPushButton("Go")
        go_button.clicked.connect(self.go_to_page)

        # Add widgets to layout
        pagination_layout.addWidget(self.prev_button)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.page_input)
        pagination_layout.addWidget(go_button)
        pagination_layout.addWidget(self.next_button)
        
        parent_layout.addLayout(pagination_layout)
        
        # Update button states
        self.update_pagination_buttons()

    def create_io_panel(self, parent_layout):
        """
        Create the right panel for input/output display.
        
        Args:
            parent_layout: The layout to which this will be added
        """
        right_panel = QVBoxLayout()
        
        # Panel title
        right_panel.addWidget(QLabel("<b>Input/Output Selection</b>"))
        
        # Input text area
        self.input_list = QTextEdit()
        self.input_list.setPlaceholderText("Input")
        right_panel.addWidget(self.input_list)
        
        # Output text area
        self.output_list = QTextEdit()
        self.output_list.setPlaceholderText("Output")
        right_panel.addWidget(self.output_list)
        
        parent_layout.addLayout(right_panel)

    def create_button_row(self, parent_layout):
        """
        Create the bottom row of navigation buttons.
        
        Args:
            parent_layout: The layout to which this will be added
        """
        button_row = QHBoxLayout()
        
        # Back button
        back_btn = QPushButton("← Back to Start")
        back_btn.setMinimumSize(150, 40)
        back_btn.clicked.connect(self.go_back_to_start)
        button_row.addWidget(back_btn)
        
        button_row.addStretch()
        
        # Build dataset button
        build_btn = QPushButton("Build Dataset")
        build_btn.setMinimumSize(150, 40)
        build_btn.clicked.connect(self.build_dataset)
        button_row.addWidget(build_btn)
        
        # Verify files button
        verify_btn = QPushButton("Verify")
        verify_btn.setMinimumSize(150, 40)
        verify_btn.clicked.connect(self.verify_files)
        button_row.addWidget(verify_btn)
        
        parent_layout.addLayout(button_row)

    def create_status_bar(self):
        """Create and configure the status bar."""
        status_bar = self.statusBar()
        status_label = QLabel("Data Monitoring Software version 1.0.13")
        status_label.setAlignment(Qt.AlignCenter)  
        status_bar.addPermanentWidget(status_label, 1)

    def go_back_to_start(self):
        """Return to the start window without exiting the application."""
        self.hide()  
        if self.start_window_ref:
            self.start_window_ref.showMaximized()  
        else:
            from windows.start_window import StartWindow
            self.start_window_ref = StartWindow()
            self.start_window_ref.showMaximized()

    def open_add_data_window(self):
        """Open the Add Data window and update the file list when files are selected."""
        self.add_data_window = AddDataWindow()
        if self.add_data_window.exec_() == QDialog.Accepted:
            selected_files = getattr(self.add_data_window, 'selected_files', [])
            self.all_files.extend(selected_files)
            self.all_files = sorted(self.all_files, key=self.extract_number)

            # Display the first page
            self.current_page = 0
            self.show_page()

    def delete_data(self):
        """Delete selected files from the list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return  
        
        # Remove the selected items from the complete list of files
        for item in selected_items:
            file_name = item.text()
            self.all_files.remove(file_name)  
            self.file_list.takeItem(self.file_list.row(item))  

    def update_progress_label(self, active_step, completed_steps=None):
        """
        Update the progress label showing the current workflow step.
        
        Args:
            active_step (str): The currently active step (highlighted)
            completed_steps (list): List of completed steps (shown in green)
        """
        steps = ["Dataset Builder", "NN Designer", "NN Evaluator"]
        completed_steps = completed_steps or []

        parts = []
        for step in steps:
            if step in completed_steps:
                # Green bold for completed steps
                parts.append(f'<span style="color: green; font-weight:bold;">{step}</span>')
            elif step == active_step:
                # Orange bold for active step
                parts.append(f'<span style="color: orange; font-weight:bold;">{step}</span>')
            else:
                # Normal style for other steps
                parts.append(step)

        text = " → ".join(parts)
        self.top_right_label.setText(f"Progress Statement : {text}")

    def build_dataset(self):
        """Simulate dataset building with progress dialog and open next window when complete."""
        if progress_state.dataset_built:
            reply = QMessageBox.question(
                self,
                "Rebuild Dataset?",
                "You have already built the dataset. Rebuilding it will reset all progress. Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
                
        progress_state.dataset_built = False
        progress_state.nn_designed = False
        
        # Create and show progress dialog
        progress = QProgressDialog("Building dataset...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Simulate progress
        for i in range(101):
            time.sleep(0.01)  
            progress.setValue(i)
            if progress.wasCanceled():
                self.statusBar().showMessage("Dataset building cancelled.", 3000)
                return

        self.statusBar().showMessage("Dataset built successfully!", 3000)
        
        # Update state and open next window
        progress_state.dataset_built = True
        self.open_nn_designer()

    def open_nn_designer(self):
        """Open the Neural Network Designer window and hide this one."""
        self.nn_designer_window = NeuralNetworkDesignerWindow()
        self.nn_designer_window.showMaximized()
        self.hide()  

    def filter_file_list(self):
        """Filter the file list based on the search text in the search bar."""
        search_text = self.search_bar.text().strip()

        # If search is empty, show all files
        if not search_text:
            self.file_list.clear()
            sorted_files = sorted(self.all_files, key=self.extract_number)  
            self.file_list.addItems(sorted_files)
            return

        # Filter files containing the search text
        filtered_files = [file for file in self.all_files if search_text in file]
        filtered_files = sorted(filtered_files, key=self.extract_number)

        # Update the list widget
        self.file_list.clear()
        if not filtered_files:
            self.file_list.addItem("No matching files")
        else:
            self.file_list.addItems(filtered_files)

    def extract_number(self, file_name):
        """
        Extract the first number from a filename for sorting.
        
        Args:
            file_name (str): The filename to process
            
        Returns:
            int: The first number found in the filename, or 0 if none found
        """
        matches = re.findall(r'\d+', file_name)  
        return int(matches[0]) if matches else 0  

    def verify_files(self):
        """Show a dialog with the total count of files in the dataset."""
        file_count = len(self.all_files)
        
        dialog = QDialog(self)
        dialog.setWindowTitle("File Verification")
        layout = QVBoxLayout(dialog)
        
        label = QLabel(f"Number of files: {file_count}")
        layout.addWidget(label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    def update_pagination_buttons(self):
        """Enable/disable pagination buttons based on current page position."""
        total_pages = (len(self.all_files) + self.items_per_page - 1) // self.items_per_page
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < total_pages - 1)

    def show_page(self):
        """Display the current page of files in the list widget."""
        if not self.all_files:
            self.show_error_page("No files available.")
            return
            
        start_index = self.current_page * self.items_per_page
        end_index = start_index + self.items_per_page
        files_to_display = self.all_files[start_index:end_index]
        
        self.file_list.clear()
        self.file_list.addItems(files_to_display)
        self.update_pagination_buttons()

    def show_previous_page(self):
        """Navigate to the previous page of files."""
        if self.current_page > 0:
            self.current_page -= 1
            self.show_page()

    def show_next_page(self):
        """Navigate to the next page of files."""
        total_pages = (len(self.all_files) + self.items_per_page - 1) // self.items_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.show_page()

    def go_to_page(self):
        """Navigate to a specific page number entered by the user."""
        try:
            page_number = int(self.page_input.text()) - 1
            total_pages = (len(self.all_files) + self.items_per_page - 1) // self.items_per_page
            
            if 0 <= page_number < total_pages:
                self.current_page = page_number
                self.show_page()
            else:
                self.show_error_page(f"Page {page_number + 1} does not exist.")
        except ValueError:
            self.show_error_page("Invalid page number.")

    def show_error_page(self, message):
        """
        Display an error message dialog.
        
        Args:
            message (str): The error message to display
        """
        QMessageBox.warning(self, "Error", message)