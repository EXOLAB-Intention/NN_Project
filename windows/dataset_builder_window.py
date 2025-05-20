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
    def __init__(self, start_window_ref=None):
        super().__init__()
        self.start_window_ref = start_window_ref
        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(100, 100, 1000, 600)

        # Pagination variables
        self.current_page = 0
        self.items_per_page = 100
        self.all_files = []

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        back_action = QAction("Back to Start", self)
        back_action.triggered.connect(self.go_back_to_start)
        file_menu.addAction(back_action)
        menubar.addMenu("Edit")
        menubar.addMenu("Options")

        # Central widget - configuration principale
        central_widget = QWidget()
        central_widget.setContentsMargins(0, 0, 0, 0)  # Supprime les marges du widget central
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # Supprime toutes les marges
        main_layout.setSpacing(0)  # Supprime l'espace entre les éléments
        central_widget.setLayout(main_layout)

        # Header - configuration spécifique
        header = Header(active_page="Dataset Builder", parent_window=self)
        main_layout.addWidget(header)

        # Contenu principal - avec des marges appropriées
        content_container = QWidget()
        content_container.setContentsMargins(10, 10, 10, 10)

        # Créez un layout vertical pour le conteneur
        container_layout = QVBoxLayout(content_container)
        container_layout.setContentsMargins(0, 0, 0, 0)  

        title_and_progress_layout = QHBoxLayout()
        container_layout.addLayout(title_and_progress_layout)

        # Titre aligné à gauche
        title = QLabel("<h1><b>Dataset Builder</b></h1>")
        title_and_progress_layout.addWidget(title)

        # Espacement flexible pour aligner la barre de progression à droite
        title_and_progress_layout.addStretch()

        # Barre de progression alignée à droite
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

        # Ajouter la disposition au layout principal
        main_layout.addLayout(title_and_progress_layout)

        # Ensuite ajoutez le layout horizontal pour le contenu principal
        content_layout = QHBoxLayout()
        container_layout.addLayout(content_layout)

        # Enfin ajoutez le content_container au main_layout
        main_layout.addWidget(content_container)

        # Left panel - File Selection
        left_panel = QVBoxLayout()
        
        # File Selection header
        file_header = QHBoxLayout()
        file_header.addWidget(QLabel("<b>File Selection</b>"))
        file_header.addStretch()
        
        add_btn = QPushButton("+ Add data")
        delete_btn = QPushButton("- Delete data")
        file_header.addWidget(add_btn)
        file_header.addWidget(delete_btn)
        
        left_panel.addLayout(file_header)
        
        # Search bar to filter files
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_bar = QLineEdit()  # Replace QTextEdit with QLineEdit
        self.search_bar.setFixedHeight(30)
        self.search_bar.setPlaceholderText("Enter file number...")
        self.search_bar.setValidator(QIntValidator())  # Limit input to numbers only
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_bar)
        left_panel.addLayout(search_layout)

        # File list (replace QTextEdit with QListWidget)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)  # Allow selecting only one file at a time
        left_panel.addWidget(self.file_list)
        
        content_layout.addLayout(left_panel)

        # Pagination controls
        pagination_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_page)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_page)

        # Page input
        self.page_input = QLineEdit()
        self.page_input.setPlaceholderText("Enter page number")
        self.page_input.setFixedWidth(100)
        self.page_input.returnPressed.connect(self.go_to_page)

        go_button = QPushButton("Go")
        go_button.clicked.connect(self.go_to_page)

        pagination_layout.addWidget(self.prev_button)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.page_input)
        pagination_layout.addWidget(go_button)
        pagination_layout.addWidget(self.next_button)
        left_panel.addLayout(pagination_layout)

        # Right panel - Input/Output
        right_panel = QVBoxLayout()
        
        # Input/Output header
        right_panel.addWidget(QLabel("<b>Input/Output Selection</b>"))
        
        # Input field
        self.input_list = QTextEdit()
        self.input_list.setPlaceholderText("Input")
        right_panel.addWidget(self.input_list)
        
        # Output field
        self.output_list = QTextEdit()
        self.output_list.setPlaceholderText("Output")
        right_panel.addWidget(self.output_list)
        
        content_layout.addLayout(right_panel)

        # Button row
        button_row = QHBoxLayout()
        
        # Back button
        back_btn = QPushButton("← Back to Start")
        back_btn.setMinimumSize(150, 40)
        back_btn.clicked.connect(self.go_back_to_start)
        button_row.addWidget(back_btn)
        
        button_row.addStretch()
        
        # Build button
        build_btn = QPushButton("Build Dataset")
        build_btn.setMinimumSize(150, 40)
        button_row.addWidget(build_btn)
        
        # Verification button
        verify_btn = QPushButton("Verify")
        verify_btn.setMinimumSize(150, 40)
        verify_btn.clicked.connect(self.verify_files)  
        button_row.addWidget(verify_btn)
        
        main_layout.addLayout(button_row)

        # Connect signals
        add_btn.clicked.connect(self.open_add_data_window)
        delete_btn.clicked.connect(self.delete_data)
        build_btn.clicked.connect(self.build_dataset)
        self.search_bar.textChanged.connect(self.filter_file_list)

        # Status bar
        status_bar = self.statusBar()
        status_label = QLabel("Data Monitoring Software version 1.0.13")
        status_label.setAlignment(Qt.AlignCenter)  
        status_bar.addPermanentWidget(status_label, 1) 

        # Update pagination buttons
        self.update_pagination_buttons()

    def go_back_to_start(self):
        """Return to the start window without exiting the application."""
        self.hide()  # Hide the current window instead of closing it
        if self.start_window_ref:
            self.start_window_ref.showMaximized()  # Properly maximize the StartWindow
        else:
            from windows.start_window import StartWindow
            self.start_window_ref = StartWindow()
            self.start_window_ref.showMaximized()

    def open_add_data_window(self):
        """Open the Add Data window and update the file list."""
        self.add_data_window = AddDataWindow()
        if self.add_data_window.exec_() == QDialog.Accepted:
            selected_files = getattr(self.add_data_window, 'selected_files', [])
            self.all_files.extend(selected_files)
            self.all_files = sorted(self.all_files, key=self.extract_number)

            # Display the first page
            self.current_page = 0
            self.show_page()

    def delete_data(self):
        """Delete the selected file(s) from the list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return  
        
        # Remove the selected items from the complete list of files
        for item in selected_items:
            file_name = item.text()
            self.all_files.remove(file_name)  
            self.file_list.takeItem(self.file_list.row(item))  

    ''' DEBUT HARRY PROGRESS BAR '''
    def update_progress_label(self, active_step, completed_steps=None):
        """
        Met à jour le label de progression avec :
        - les étapes complètes en vert
        - l’étape active en orange
        - les autres en style normal
        """
        steps = ["Dataset Builder", "NN Designer", "NN Evaluator"]
        completed_steps = completed_steps or []

        parts = []
        for step in steps:
            if step in completed_steps:
                # Vert gras
                parts.append(f'<span style="color: green; font-weight:bold;">{step}</span>')
            elif step == active_step:
                # Orange gras
                parts.append(f'<span style="color: orange; font-weight:bold;">{step}</span>')
            else:
                # Style normal
                parts.append(step)

        text = " → ".join(parts)
        self.top_right_label.setText(f"Progress Statement : {text}")

    def build_dataset(self):
        """Simule la construction du dataset avec une barre de progression."""
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
        progress = QProgressDialog("Building dataset...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i in range(101):
            time.sleep(0.01)  # Simule une tâche lourde
            progress.setValue(i)
            if progress.wasCanceled():
                self.statusBar().showMessage("Dataset building cancelled.", 3000)
                return

        self.statusBar().showMessage("Dataset built successfully!", 3000)
        
        progress_state.dataset_built = True
        # Ensuite, tu peux passer à la fenêtre suivante si nécessaire :
        self.open_nn_designer()

    def open_nn_designer(self):
        """Ouvre la fenêtre suivante après la construction du dataset."""
        self.nn_designer_window = NeuralNetworkDesignerWindow()
        self.nn_designer_window.showMaximized()
        self.hide()  
    ''' FIN HARRY PROGRESS BAR '''


    def filter_file_list(self):
        """Filter the file list based on the search text."""
        search_text = self.search_bar.text().strip()

        # If the complete list of files is not initialized, initialize it
        if not hasattr(self, 'all_files') or not self.all_files:
            self.all_files = [self.file_list.item(i).text() for i in range(self.file_list.count())]

        # If the search bar is empty, redisplay all files
        if not search_text:
            self.file_list.clear()
            sorted_files = sorted(self.all_files, key=self.extract_number)  # Sort by number
            self.file_list.addItems(sorted_files)
            return

        # Filter files containing the search text
        filtered_files = [file for file in self.all_files if search_text in file]

        # Sort the filtered files by number
        filtered_files = sorted(filtered_files, key=self.extract_number)

        # If no files match, display a message
        self.file_list.clear()
        if not filtered_files:
            self.file_list.addItem("No matching files")
        else:
            self.file_list.addItems(filtered_files)

    def extract_number(self, file_name):
        """Extract the first number from a file name for sorting."""
        matches = re.findall(r'\d+', file_name)  # Find all sequences of digits
        return int(matches[0]) if matches else 0  # Return the first number found or 0

    def verify_files(self):
        """Show the total number of files in a dialog."""
        file_count = len(self.all_files)

        # Display the count in a dialog
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
        """Enable or disable pagination buttons."""
        total_pages = (len(self.all_files) + self.items_per_page - 1) // self.items_per_page
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < total_pages - 1)

    def show_page(self):
        """Display the current page of files."""
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
        """Show the previous page of files."""
        if self.current_page > 0:
            self.current_page -= 1
            self.show_page()

    def show_next_page(self):
        """Show the next page of files."""
        total_pages = (len(self.all_files) + self.items_per_page - 1) // self.items_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.show_page()

    def go_to_page(self):
        """Navigate to a specific page based on user input."""
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
        """Display an error message in a dialog box."""
        QMessageBox.warning(self, "Error", message)
