from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QProgressDialog,
    QPushButton, QAction, QFileDialog, QDialog, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QGroupBox, QCheckBox, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor, QIntValidator, QFont
 
import re
import time
from widgets.header import Header
import windows.progress_state as progress_state
import os
import h5py

class DatasetBuilderWindow(QMainWindow):
    """
    Main window for the Dataset Builder module of the Neural Network Trainer application.
    Handles file selection, dataset building, and navigation to other modules.
    """
    
    def __init__(self, start_window_ref=None, saved_state=None):
        """
        Initialize the Dataset Builder window.
        
        Args:
            start_window_ref (QMainWindow): Reference to the start window for navigation
            saved_state (dict): A dictionary containing the saved state of the Dataset Builder.
        """
        super().__init__()
        self.start_window_ref = start_window_ref
        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(100, 100, 1000, 600)

        self.state = {
            "selected_folder": None,
            "all_files": [],
            "selected_inputs": [],
            "filtered_files": [],
            "current_page": 0
        }

        # ...existing code...
        if saved_state:
            self.state.update(saved_state)
            # Toujours restaurer la liste actuelle si elle existe
            if "original_files" in saved_state:
                self.state["all_files"] = list(saved_state["original_files"])
            # NE PAS vider filtered_files si on vient de NN Designer ou d'un modèle chargé
            # On ne le vide que si on vient du StartWindow (par exemple si "from_start_window" dans le state)
            if saved_state.get("from_start_window", False):
                self.state["filtered_files"] = []
            self.state["all_files"] = [f for f in self.state["all_files"] if not os.path.basename(f).startswith("filtered_")]



        # Pagination variables
        self.items_per_page = 100  # Default number of items per page

        # Initialize UI components
        self.init_ui()

        # Restore selected inputs and display files
        self.restore_state()
        self.show_page()

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
        
        # Content container with reduced margins
        content_container = QWidget()
        content_container.setContentsMargins(0, 0, 0, 0)  
        container_layout = QVBoxLayout(content_container)

        # Add title and progress indicator
        self.create_title_progress_layout(container_layout)
        
        # Main content layout (horizontal) for alignment
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10) 

        # Left panel - File selection
        self.create_file_selection_panel(content_layout)
        
        # Right panel - Input/Output
        self.create_io_panel(content_layout)
        
        container_layout.addLayout(content_layout)

        # Bottom buttons
        self.create_button_row(container_layout)
        main_layout.addWidget(content_container)
        
    def apply_filters_to_raw_data(self):
        from functions import process_emg_signal
        import h5py
        import numpy as np

        filtered_data = []
        for file in self.state["all_files"]:
            # Chemin absolu du fichier
            if "selected_folder" in self.state and self.state["selected_folder"]:
                file_path = os.path.join(self.state["selected_folder"], file)
            else:
                file_path = file  # fallback

            # Ouvre le fichier et traite chaque trial
            try:
                with h5py.File(file_path, 'r') as h5f:
                    for trial in h5f.keys():
                        trial_group = h5f[trial]
                        # Exemple : traiter emgL1 si présent
                        if "emgL1" in trial_group:
                            emg = np.array(trial_group["emgL1"])
                            print("Shape du signal EMG:", emg.shape)
                            print("Premiers éléments:", emg[:10])
                            processed = process_emg_signal(emg)
                            filtered_data.append(processed)
            except Exception as e:
                print(f"Erreur lors du traitement de {file_path} : {e}")

        self.state["filtered_files"] = filtered_data


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
        file_title = QLabel("<b>File Selection</b>")
        file_title.setStyleSheet("font-size: 16px; font-weight: bold;")  
        file_header.addWidget(file_title)
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
        Create the right panel with a white background, matching height, and simplified layout for Input/Output Selection.
        """
        right_panel = QVBoxLayout()

        # Add title for Input/Output Selection
        io_title = QLabel("<b>Input/Output Selection</b>")
        io_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        right_panel.addWidget(io_title)

        # Main container styled like the file container
        container = QFrame()
        container.setFrameShape(QFrame.StyledPanel)
        container.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
            }
        """)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(10)

        # Input Section
        input_group = QGroupBox("Input")
        input_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(10, 10, 10, 10)

        # EMG/IMU checkboxes in horizontal layout
        checkbox_row = QHBoxLayout()
        checkbox_row.setAlignment(Qt.AlignTop)  

        # EMG Column
        emg_group = QVBoxLayout()
        emg_group.setAlignment(Qt.AlignTop) 
        emg_label = QLabel("<b>EMG Sensors</b>")
        emg_label.setStyleSheet("margin-bottom: 5px; border: none;")  
        emg_group.addWidget(emg_label)
        self.emg_checkboxes = []
        for i in range(1, 5):
            cb_left = QCheckBox(f"emgL{i}")
            cb_right = QCheckBox(f"emgR{i}")
            emg_group.addWidget(cb_left)
            emg_group.addWidget(cb_right)
            self.emg_checkboxes.extend([cb_left, cb_right])
        checkbox_row.addLayout(emg_group)

        # IMU Column
        imu_group = QVBoxLayout()
        imu_group.setAlignment(Qt.AlignTop)  
        imu_label = QLabel("<b>IMU Sensors</b>")
        imu_label.setStyleSheet("margin-bottom: 5px; border: none;")  
        imu_group.addWidget(imu_label)
        self.imu_checkboxes = []
        for i in range(1, 6):
            cb = QCheckBox(f"imu{i}")
            imu_group.addWidget(cb)
            self.imu_checkboxes.append(cb)
        checkbox_row.addLayout(imu_group)

        input_layout.addLayout(checkbox_row)
        input_group.setLayout(input_layout)
        container_layout.addWidget(input_group)

        # Output Section
        output_group = QGroupBox("Output")
        output_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(10, 10, 10, 10)

        # Output checkbox
        self.output_checkbox = QCheckBox("Button_ok")
        self.output_checkbox.setChecked(True)
        self.output_checkbox.setEnabled(False)
        output_layout.addWidget(self.output_checkbox)

        output_group.setLayout(output_layout)
        container_layout.addWidget(output_group)

        right_panel.addWidget(container)
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
        reply = QMessageBox.question(
            self,
            "Confirm Return",
            "Are you sure you want to return to the start? All unsaved progress in Neural Network Designer will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return  # Annule le retour

        self.hide()
        if hasattr(self, 'parent_window') and self.parent_window is not None:
            self.parent_window.showMaximized()
        else:
            from windows.start_window import StartWindow
            self.start_window = StartWindow()
            self.start_window.showMaximized()

    def open_add_data_window(self):
        """
        Open a dialog to select a folder and add all .h5 files from the folder to the file list.
        Store only file names in self.all_files for proper access.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing .h5 Files")
        if folder_path:
            # Find all .h5 files in the selected folder
            h5_files = [
                file for file in os.listdir(folder_path) if file.endswith(".h5")
            ]

            if not h5_files:
                QMessageBox.warning(self, "No Files Found", "No .h5 files found in the selected folder.")
                return

            # Add only file names to self.all_files
            self.state["all_files"].extend(h5_files)
            self.state["all_files"] = sorted(self.state["all_files"], key=self.extract_number)

            # Store the selected folder path for later use
            self.state["selected_folder"] = folder_path
            self.state["original_files"] = list(self.state["all_files"]) 
            # Display the first page
            self.state["current_page"] = 0
            self.show_page()
        else:
            QMessageBox.information(self, "No Folder Selected", "No folder was selected.")

    def delete_data(self):
        """Delete selected files from the list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return  
        
        # Remove the selected items from the complete list of files
        for item in selected_items:
            file_name = item.text()
            if file_name in self.state["all_files"]:
                self.state["all_files"].remove(file_name)
            self.file_list.takeItem(self.file_list.row(item))
        
        # Met à jour la liste originale pour la persistance/navigation
        self.state["original_files"] = list(self.state["all_files"])

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
        """
        Filter the .h5 files based on selected inputs, save the configuration to a .txt file,
        and navigate to the NN Designer page.
        """

        if not self.state["all_files"]:
            QMessageBox.warning(self, "No Files", "No files available to build the dataset.")
            return

        # ✅ 1. Vérifie si un dataset ou un modèle a déjà été créé/entraîné dans progress_state
        import windows.progress_state as progress_state
        model_trained = getattr(progress_state, "nn_designed", False)
        dataset_built = getattr(progress_state, "dataset_built", False)
        need_reset = False

        if model_trained or dataset_built:
            reply = QMessageBox.question(
                self,
                "Reset All Progress?",
                "A model has already been trained or a dataset built. Rebuilding will reset ALL progress (model, training, évaluation). Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return  # ⛔ Ne continue pas si l'utilisateur refuse
            need_reset = True

        if need_reset:
            # Reset tout l'état global
            progress_state.nn_designed = False
            progress_state.trained_model = None
            progress_state.training_history = None
            progress_state.test_results = None
            progress_state.training_started = False
            progress_state.dataset_built = False
            progress_state.dataset_path = None

             # Reset aussi les fenêtres NN Designer et NN Evaluator si elles existent
            if hasattr(self, "nn_designer") and self.nn_designer:
                self.nn_designer.training_history = None
                self.nn_designer.trained_model = None
                self.nn_designer.test_results = None
                self.nn_designer.training_completed = False
                self.nn_designer.training_stopped = False
                self.nn_designer.training_monitor_text.clear()
                self.nn_designer.nn_text_edit.setText("No parameters saved")
                self.nn_designer.eval_stack.setCurrentIndex(0)
            if hasattr(self, "nn_evaluator") and self.nn_evaluator:
                self.nn_evaluator.test_results = None

        # Debug: Print all file names
        print("Files to process:")
        for file_name in self.state["all_files"]:
            print(file_name)

        # Get selected inputs
        selected_inputs = [cb.text() for cb in self.emg_checkboxes + self.imu_checkboxes if cb.isChecked()]
        if not selected_inputs:
            QMessageBox.warning(self, "No Inputs Selected", "Please select at least one input sensor.")
            return

        # Ensure output (button_ok) is always included
        selected_outputs = ["button_ok"]

        # Create a new dataset folder only if files are processed
        parent_folder = "datasets"
        os.makedirs(parent_folder, exist_ok=True)
        dataset_index = len([name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name)) and name.startswith("dataset")]) + 1
        dataset_folder = os.path.join(parent_folder, f"dataset{dataset_index}")

        # Filter and save the dataset
        filtered_files = []
        for file_name in self.state["all_files"]:
            file_path = os.path.join(self.state["selected_folder"], file_name)  # Use the selected folder path
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "File Missing", f"File not found: {file_name}. Skipping...")
                continue

            try:
                with h5py.File(file_path, 'r') as h5_file:
                    filtered_data = {}

                    # Recursive function to search for selected inputs
                    def search_and_filter(group, trial_data):
                        for key, item in group.items():
                            if isinstance(item, h5py.Group):
                                # Recursively search in sub-groups
                                search_and_filter(item, trial_data)
                             # ''' Harry + TIME Data '''
                            elif isinstance(item, h5py.Dataset) and key in selected_inputs + selected_outputs or key == "time":
                                trial_data[key] = item[()]

                    # === DÉBUT DU BLOC SENSOR/CONTROLLER À COMMENTER ===
                    # from functions import generate_fixed_sequence_labels, convert_labels_to_int, phase_to_int
                    # # ✅ Bloc spécifique pour les fichiers Sensor / Controller
                    # if "Sensor" in h5_file and "Controller" in h5_file:
                    #     sensor_data_full = {}
                    #     controller_data_full = {}
                    #     search_and_filter(h5_file["Sensor"], sensor_data_full)
                    #     search_and_filter(h5_file["Controller"], controller_data_full)
                    #
                    #     if "button_ok" in controller_data_full and "time" in sensor_data_full:
                    #         import numpy as np
                    #         from functions import generate_fixed_sequence_labels, convert_labels_to_int, phase_to_int
                    #
                    #         ok = np.squeeze(controller_data_full["button_ok"])
                    #         time = np.squeeze(sensor_data_full["time"])
                    #         press_indices = np.where(np.diff(ok.astype(int)) == 1)[0] + 1
                    #
                    #         num_trials = len(press_indices) // 4
                    #         for trial_idx in range(num_trials):
                    #             trial_press = press_indices[trial_idx*4:(trial_idx+1)*4]
                    #             if len(trial_press) < 4:
                    #                 continue  # trial incomplet
                    #
                    #             start = trial_press[0] - 500 if trial_press[0] - 500 > 0 else 0
                    #             end = trial_press[3] + 1000 if trial_press[3] + 1000 < len(ok) else len(ok)
                    #
                    #             # Extraire les signaux pour ce trial
                    #             trial_sensor = {k: np.array(sensor_data_full[k])[start:end] for k in sensor_data_full if k != "label" and k != "label_int"}
                    #             trial_controller = {k: np.array(controller_data_full[k])[start:end] for k in controller_data_full if k != "label" and k != "label_int"}
                    #
                    #             # Vérifier qu'il y a exactement 4 pressions dans la fenêtre extraite
                    #             trial_ok = np.squeeze(trial_controller["button_ok"])
                    #             trial_press_in_window = np.where(np.diff(trial_ok.astype(int)) == 1)[0] + 1
                    #             if len(trial_press_in_window) != 4:
                    #                 print(f"⚠️ Trial {trial_idx} ignoré : {len(trial_press_in_window)} pressions trouvées dans la fenêtre")
                    #                 continue
                    #
                    #             # Générer les labels
                    #             try:
                    #                 temp = {
                    #                     "button_ok": trial_controller["button_ok"],
                    #                     "time": trial_sensor["time"]
                    #                 }
                    #                 temp["label"] = generate_fixed_sequence_labels(temp, fs=100)
                    #                 temp = convert_labels_to_int(temp, phase_to_int)
                    #                 trial_controller["label"] = temp["label"]
                    #                 trial_controller["label_int"] = temp["label_int"]
                    #                 trial_sensor["label"] = temp["label"]
                    #                 trial_sensor["label_int"] = temp["label_int"]
                    #             except Exception as e:
                    #                 print(f"⚠️ Erreur lors de la génération des labels pour trial {trial_idx} : {e}")
                    #                 continue
                    #
                    #             # Ajoute chaque trial comme un groupe trial_xxx
                    #             trial_name = f"trial_{trial_idx:03d}"
                    #             filtered_data[trial_name] = {}
                    #             filtered_data[trial_name]["Sensor"] = trial_sensor
                    #             filtered_data[trial_name]["Controller"] = trial_controller
                    # === FIN DU BLOC SENSOR/CONTROLLER À COMMENTER ===

                    # Iterate through trials and apply the recursive search
                    from functions import generate_fixed_sequence_labels, convert_labels_to_int, phase_to_int
                    for trial in h5_file.keys():
                        trial_data = {}
                        search_and_filter(h5_file[trial], trial_data)

                        # Ajout des labels intelligents
                        if "button_ok" in trial_data and "time" in trial_data:
                            try:
                                label_data = {
                                    "button_ok": trial_data["button_ok"],
                                    "time": trial_data["time"]
                                }
                                label_str = generate_fixed_sequence_labels(label_data, fs=100)
                                label_data["label"] = label_str
                                label_data = convert_labels_to_int(label_data, phase_to_int)
                                trial_data["label"] = label_data["label"]
                                trial_data["label_int"] = label_data["label_int"]
                            except Exception as e:
                                print(f"⚠️ Erreur génération de labels dans {trial} : {e}")
                         # === AJOUTE CE BLOC POUR CHAQUE SIGNAL EMG COCHÉ ===
                        from functions import process_emg_signal, normalize_emg  
                        for key in list(trial_data.keys()):
                            if key.startswith("emg"):
                                emg = trial_data[key]
                                try:
                                    emg_filt = process_emg_signal(emg)
                                    emg_norm = normalize_emg(emg_filt)
                                    trial_data[key + "_filt"] = emg_filt
                                    trial_data[key + "_norm"] = emg_norm
                                except Exception as e:
                                    print(f"⚠️ Erreur filtrage/normalisation {key} dans {trial}: {e}")

                        if trial_data:
                            filtered_data[trial] = trial_data


                    # Save the filtered data to a file in the dataset folder
                    os.makedirs(dataset_folder, exist_ok=True)  # Create the dataset folder only when needed
                    filtered_file_path = os.path.join(dataset_folder, f"filtered_{file_name}")
                    with h5py.File(filtered_file_path, 'w') as filtered_h5:
                        for trial, trial_data in filtered_data.items():
                            trial_group = filtered_h5.create_group(trial)
                            for dataset, data in trial_data.items():
                                trial_group.create_dataset(dataset, data=data)
                    # Ajoute le chemin relatif à NN_Project
                    project_root = None
                    path = os.path.abspath(dataset_folder)
                    while path and os.path.basename(path) != "NN_Project":
                        parent = os.path.dirname(path)
                        if parent == path:
                            break
                        path = parent
                    if os.path.basename(path) == "NN_Project":
                        project_root = path
                    if project_root:
                        rel_path = os.path.relpath(filtered_file_path, project_root)
                    else:
                        rel_path = f"filtered_{file_name}"
                    filtered_files.append(str(rel_path))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error processing file {file_name}: {e}")
                continue

        # Save the configuration to a .txt file in the dataset folder
        if filtered_files:
            from datetime import datetime
            # Get current date and time
            creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

            config_file_path = os.path.join(dataset_folder, "dataset_config.txt")
            with open(config_file_path, 'w') as config_file:
                config_file.write(f"Dataset Creation Date: {creation_time}\n\n")
                config_file.write("Selected Inputs:\n")
                config_file.writelines(f"- {sensor}\n" for sensor in selected_inputs)
                config_file.write("\nSelected Outputs:\n")
                config_file.writelines(f"- {output}\n" for output in selected_outputs)
                config_file.write("\nFiltered Files:\n")
                config_file.writelines(f"- {file}\n" for file in filtered_files)

            QMessageBox.information(self, "Dataset Built", f"Dataset built successfully! Configuration saved to {config_file_path}")

            # Prepare complete state transfer
            saved_state = self.get_saved_state()
            saved_state.update({
                "filtered_files": filtered_files,
                "dataset_path": dataset_folder,
                "selected_files": [],
                "training_history": None,
                "trained_model": None,
                "test_results": None,
                "training_completed": False,
                "training_stopped": False,
                "hyperparameters": None,         # <-- reset hyperparams
                "optimizer": "Adam",             # <-- valeur par défaut
                "loss_function": None,
                "summary_text": "No parameters saved",  # <-- reset summary
                "training_monitor_logs": "",
                "training_progress": 0,
                "eval_plot_index": 0,
                "training_started": False
            })

            saved_state["filtered_files"] = [str(f) for f in filtered_files]
            saved_state["original_files"] = [str(f) for f in self.state["all_files"]]
            saved_state["returned_from_nn_designer"] = True
            saved_state["all_files"] = [str(f) for f in filtered_files]
            progress_state.dataset_built = True
            from windows.nn_designer_window import NeuralNetworkDesignerWindow
            self.nn_designer = NeuralNetworkDesignerWindow(
                dataset_path=dataset_folder,
                saved_state=saved_state
            )
            self.nn_designer.showMaximized()
            self.hide()
        else:
            QMessageBox.warning(self, "No Files Processed", "No files were successfully processed.")

    def open_nn_designer(self, filtered_files, dataset_folder):
        saved_state = self.get_saved_state()
        # Si filtered_files est vide, on passe les originaux
        if not filtered_files:
            saved_state["all_files"] = list(self.state["all_files"])
        else:
            saved_state["all_files"] = list(filtered_files)
        print("filtered_files:", filtered_files)
        print("saved_state['all_files']:", saved_state["all_files"])
        from windows.nn_designer_window import NeuralNetworkDesignerWindow
        self.nn_designer_window = NeuralNetworkDesignerWindow(
            dataset_path=dataset_folder,
            saved_state=saved_state
        )
        
        self.nn_designer_window.populate_file_list_with_paths(saved_state["all_files"])
        self.nn_designer_window.showMaximized()
        self.hide()

    def filter_file_list(self):
        """Filter the file list based on the search text in the search bar."""
        search_text = self.search_bar.text().strip()

        # If search is empty, show all files
        if not search_text:
            self.file_list.clear()
            sorted_files = sorted(self.state["all_files"], key=self.extract_number)  
            self.file_list.addItems(sorted_files)
            return

        # Filter files containing the search text
        filtered_files = [file for file in self.state["all_files"] if search_text in file]
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
        file_count = len(self.state["all_files"])
        
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
        total_pages = (len(self.state["all_files"]) + self.items_per_page - 1) // self.items_per_page
        self.prev_button.setEnabled(self.state["current_page"] > 0)
        self.next_button.setEnabled(self.state["current_page"] < total_pages - 1)

    def show_page(self):
        """Affiche les fichiers d'origine (pas les filtrés) dans la liste de gauche."""
        self.file_list.clear()
        files = self.state["all_files"]  # <-- Toujours les fichiers d'origine
        start = self.state["current_page"] * self.items_per_page
        end = start + self.items_per_page
        for i, file in enumerate(files[start:end]):
            item = QListWidgetItem(os.path.basename(file))  # ✅ Affiche nom uniquement
            item.setData(Qt.UserRole, file)                 # ✅ Garde chemin complet pour les traitements internes
            self.file_list.addItem(item)

    def show_previous_page(self):
        """Navigate to the previous page of files."""
        if self.state["current_page"] > 0:
            self.state["current_page"] -= 1
            self.show_page()

    def show_next_page(self):
        """Navigate to the next page of files."""
        total_pages = (len(self.state["all_files"]) + self.items_per_page - 1) // self.items_per_page
        if self.state["current_page"] < total_pages - 1:
            self.state["current_page"] += 1
            self.show_page()

    def go_to_page(self):
        """Navigate to a specific page number entered by the user."""
        try:
            page_number = int(self.page_input.text()) - 1
            total_pages = (len(self.state["all_files"]) + self.items_per_page - 1) // self.items_per_page
            
            if 0 <= page_number < total_pages:
                self.state["current_page"] = page_number
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

    def restore_state(self):
        """Restore UI from saved state."""
        # Restore file list
        if self.state["all_files"]:
            self.file_list.clear()
            # NE GARDE QUE LES FICHIERS NON FILTRÉS
            self.file_list.addItems(self.state["all_files"])

        # Restore input checkboxes
        for checkbox in self.emg_checkboxes + self.imu_checkboxes:
            checkbox.setChecked(checkbox.text() in self.state["selected_inputs"])

    def get_saved_state(self):
        """
        Get the current state including checkbox states.
        """
        self.state["selected_inputs"] = [cb.text() for cb in self.emg_checkboxes + self.imu_checkboxes if cb.isChecked()]
        self.state["original_files"] = list(self.state["all_files"])  # <-- Ajoute ou vérifie cette ligne
        return self.state
