from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGridLayout,
    QPushButton, QComboBox, QTextEdit, QListWidget, QListWidgetItem, QProgressDialog,
    QFrame, QMessageBox, QGroupBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIntValidator, QFont
import random
import os
import time
from widgets.header import Header
import windows.progress_state as progress_state


class NeuralNetworkDesignerWindow(QMainWindow):
    """
    Main window for the Neural Network Designer module.
    Allows users to:
    - Select training/test sets
    - Configure neural network hyperparameters
    - Choose optimizers and loss functions
    - Start/stop training
    - Monitor training progress
    """
    
    def __init__(self, dataset_path=None, saved_state=None):
        """
        Initialize the Neural Network Designer window.
        
        Args:
            dataset_path (str): Path to the dataset directory 
            saved_state (dict): A dictionary containing the saved state of the NN Designer.
        """
        super().__init__()
        # Explicitly set dataset_path as an instance variable
        self.dataset_path = dataset_path

        # Initialize with comprehensive default state
        self.state = {
            "optimizer": "Adam",
            "loss_function": None,
            "hyperparameters": {
                "layer_number": "3",
                "sequence_length": "50",
                "batch_size": "32",
                "epoch_number": "50"
            },
            "selected_files": [],
            "dataset_path": dataset_path,
            "summary_text": "No parameters saved yet"
        }
        
        # Merge with saved state if exists
        if saved_state:
            self.state.update(saved_state)
            # Ensure dataset_path is consistent with the saved state
            self.dataset_path = self.state.get("dataset_path", self.dataset_path)
        
        self.init_ui()
        self.restore_state()
        
        # Ensure summary is always visible
        self.update_summary_display()

    def init_ui(self):
        """Initialize all UI components of the window."""
        # Create scrollable container for responsiveness
        self.create_scrollable_container()
        
        # Create header with navigation
        self.create_header()
        
        # Create progress indicator
        self.create_progress_indicator()
        
        # Create main content sections
        self.create_top_controls()
        self.create_main_content()
        self.create_training_controls()
        self.create_training_monitor()
        
        # Create status bar
        self.create_status_bar()
        
        # Populate file list if dataset path was provided
        if self.dataset_path:
            self.populate_file_list()
        
        # Populate loss function combo boxes using the loss_function_categories dictionary
        self.classification_loss_combo.addItems(self.loss_function_categories["Classification"])
        self.regression_loss_combo.addItems(self.loss_function_categories["Regression"])
        
        # Set default selections
        self.classification_loss_combo.setCurrentText("CrossEntropyLoss")
        self.regression_loss_combo.setCurrentText("MSELoss")
        
        # Connect signals
        self.classification_loss_combo.currentIndexChanged.connect(
            lambda: self.on_loss_function_changed(is_classification=True))
        self.regression_loss_combo.currentIndexChanged.connect(
            lambda: self.on_loss_function_changed(is_classification=False))
        
        # Ensure mutual exclusivity for default selections
        self.on_loss_function_changed(is_classification=True)

    def create_scrollable_container(self):
        """Create the scrollable main container for responsive layout."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        container = QWidget()
        scroll.setWidget(container)
        
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  
        
        self.setCentralWidget(scroll)

    def create_header(self):
        """Create the application header with navigation."""
        header_container = QWidget()
        header_container_layout = QVBoxLayout(header_container)
        header_container_layout.setContentsMargins(0, 0, 0, 0)  
        header_container_layout.setSpacing(0)

        # Add the header widget
        header = Header(active_page="Neural Network Designer", parent_window=self)
        header_container_layout.addWidget(header)

        self.main_layout.addWidget(header_container, alignment=Qt.AlignTop)

    def create_progress_indicator(self):
        """Create the progress indicator showing workflow steps."""
        # Top bar layout (back button, title, progress)
        top_bar_layout = QHBoxLayout()

        # Left: Back button
        left_widget = QWidget()
        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0,0,0,0)
        back_btn = QPushButton("← Back")
        back_btn.setFixedSize(100, 30)
        back_btn.clicked.connect(self.go_back_to_start)
        left_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        left_widget.setLayout(left_layout)

        # Center: Title
        center_widget = QWidget()
        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0,0,0,0)
        title_label = QLabel("Neural Network Designer")
        title_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignLeft)
        center_layout.addWidget(title_label, alignment=Qt.AlignLeft)
        center_widget.setLayout(center_layout)

        # Right: Progress label
        right_widget = QWidget()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0,0,0,0)
        self.top_right_label = QLabel()
        font = QFont("Arial", 10)
        font.setItalic(True)
        self.top_right_label.setFont(font)
        self.top_right_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.top_right_label.setTextFormat(Qt.RichText)
        self.top_right_label.setStyleSheet("""
            background-color: rgba(200, 200, 200, 50);
            padding: 6px 12px;
            border-radius: 8px;
            margin: 5px;
        """)
        right_layout.addWidget(self.top_right_label, alignment=Qt.AlignRight)
        right_widget.setLayout(right_layout)

        # Add all sections to top bar
        top_bar_layout.addWidget(left_widget)
        top_bar_layout.addWidget(center_widget, stretch=1)
        top_bar_layout.addWidget(right_widget)

        self.main_layout.addLayout(top_bar_layout)
        
        # Initialize progress state
        self.update_progress(active_step="NN Designer", completed_steps=["Dataset Builder"])

    def create_top_controls(self):
        """Create the top row of controls (training set selection)."""
        top_row = QHBoxLayout()

        # Training set controls
        left_title_controls = QHBoxLayout()
        training_label = QLabel("Training/Test Set")
        training_label.setStyleSheet("font-weight: bold; font-style: italic;")
        left_title_controls.addWidget(training_label)
        
        # Training percentage input
        self.training_percentage_input = QLineEdit("20%")
        self.training_percentage_input.setMaximumWidth(60)
        left_title_controls.addWidget(self.training_percentage_input)
        
        # Auto-select button
        auto_select_button = QPushButton("Auto Select")
        auto_select_button.setStyleSheet("""
            background-color: #dceaf7; 
            font-size: 12px; 
            font-weight: bold; 
            border: 2px solid white; 
            padding: 4px 4px;
        """)
        auto_select_button.clicked.connect(self.auto_select_files)
        left_title_controls.addWidget(auto_select_button) 
        left_title_controls.addStretch()

        # Container for training controls
        training_container = QWidget()
        training_layout = QVBoxLayout(training_container)
        training_layout.setContentsMargins(7, 0, 0, 0)  
        training_layout.addLayout(left_title_controls)
        
        # Middle and right section labels
        middle_title = QLabel("NN Model")
        middle_title.setStyleSheet("font-weight: bold; font-style: italic;")
        
        right_title = QLabel("Evaluation Plot")
        right_title.setStyleSheet("font-weight: bold; font-style: italic;")

        # Add all to top row
        top_row.addWidget(training_container, 1)
        top_row.addWidget(middle_title, 1, alignment=Qt.AlignLeft)
        top_row.addWidget(right_title, 1, alignment=Qt.AlignLeft)

        self.main_layout.addLayout(top_row)

    def create_main_content(self):
        """Create the main content area with three panels."""
        content_layout = QHBoxLayout()
        
        # Left panel - File selection
        self.create_file_selection_panel(content_layout)
        
        # Middle panel - NN configuration
        self.create_nn_config_panel(content_layout)
        
        # Right panel - Evaluation plot
        self.create_evaluation_panel(content_layout)
        
        self.main_layout.addLayout(content_layout)

    def create_file_selection_panel(self, parent_layout):
        """Create the left panel for file selection."""
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setStyleSheet("background: white")
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)

        # File list widget
        left_layout.addWidget(QLabel("File name"))
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            border: 1px solid #dcdcdc; 
            background-color: white; 
            padding: 4px;
        """)
        
        # Connect the signal for checkbox state changes
        self.file_list.itemChanged.connect(self.on_checkbox_changed)
            
        left_layout.addWidget(self.file_list)

        # Container for proper margins
        left_panel_container = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_container)
        left_panel_layout.setContentsMargins(7, 0, 0, 0) 
        left_panel_layout.addWidget(left_panel)
        
        parent_layout.addWidget(left_panel_container, 1)

    def on_checkbox_changed(self, item):
        """Called automatically when a checkbox state changes."""
        self.save_current_selections()
        print(f"Checkbox changed: {item.text()} -> {'Checked' if item.checkState() == Qt.Checked else 'Unchecked'}")

    def save_current_selections(self):
        """Save the current state of checkboxes in real-time."""
        self.state["selected_files"] = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]
        # Update the complete saved state
        self.saved_state = self.get_saved_state()
        print(f"Saved selected files: {self.state['selected_files']}")

    def show_page(self):
        """Display the current page of files in the list widget."""
        self.file_list.clear()
        for file_name in self.state["all_files"]:
            item = QListWidgetItem(file_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if file_name in self.state["selected_files"] else Qt.Unchecked)
            self.file_list.addItem(item)

    def create_nn_config_panel(self, parent_layout):
        """
        Create the middle panel for NN configuration.
        
        Args:
            parent_layout: The layout to add this panel to
        """
        middle_panel = QFrame()
        middle_panel.setFrameShape(QFrame.StyledPanel)
        middle_panel.setStyleSheet("background: white")
        middle_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        middle_layout = QVBoxLayout(middle_panel)

        # Initialize optimizer combo box
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems([
            "Adam", "SGD", "AdamW"
        ])

        # Define loss function categories (single source of truth)
        self.loss_function_categories = {
            "Classification": ["CrossEntropyLoss", "BCEWithLogitsLoss"],
            "Regression": ["MSELoss", "SmoothL1Loss", "HuberLoss"]
        }

        # Populate loss function combos using the dictionary
        self.classification_loss_combo = QComboBox()
        self.regression_loss_combo = QComboBox()
        self.classification_loss_combo.addItems(self.loss_function_categories["Classification"])
        self.regression_loss_combo.addItems(self.loss_function_categories["Regression"])

        # Set default values
        self.classification_loss_combo.setCurrentText("CrossEntropyLoss")
        self.regression_loss_combo.setCurrentText("MSELoss")

        # Ensure only one loss type is selected
        self.classification_loss_combo.currentIndexChanged.connect(
            lambda i: self.clear_regression_loss() if i != -1 else None
        )
        self.regression_loss_combo.currentIndexChanged.connect(
            lambda i: self.clear_classification_loss() if i != -1 else None
        )

        # Style for combo boxes
        combo_style = """
            QComboBox { 
                border: 1px solid #dcdcdc; 
                padding: 4px; 
                background-color: white; 
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                border-left: 1px solid #cce0ff;
                background-color: #dceaf7; 
                image: url(assets/arrow_down.png);
                min-width: 20px; 
                min-height: 20px; 
                border: none; 
            }        
        """
        self.optimizer_combo.setStyleSheet(combo_style)
        self.classification_loss_combo.setStyleSheet(combo_style)
        self.regression_loss_combo.setStyleSheet(combo_style)

        # Add optimizer and loss sections
        middle_layout.addWidget(QLabel("<b><i>Optimizer</b></i>"))
        middle_layout.addWidget(self.optimizer_combo)
        middle_layout.addWidget(QLabel("<b><i>Classification Loss Function</b></i>"))
        middle_layout.addWidget(self.classification_loss_combo)
        middle_layout.addWidget(QLabel("<b><i>Regression Loss Function</b></i>"))
        middle_layout.addWidget(self.regression_loss_combo)
        middle_layout.addWidget(QLabel("<b><i>Neural Network Model</b></i>"))

        # Hyperparameters group box
        hyper_box = QGroupBox("Hyperparameters")
        hyper_box.setStyleSheet("""
            border: 1px solid #dcdcdc;
            border-radius: 5px;
            padding: 15px;
        """)
        hyper_layout = QGridLayout()  
        hyper_layout.setContentsMargins(0, 10, 10, 10)
        hyper_layout.setVerticalSpacing(5)  

        # Hyperparameter inputs
        self.layer_number_input = QLineEdit("3")
        self.sequence_length_input = QLineEdit("50")
        self.batch_size_input = QLineEdit("32")
        self.epoch_number_input = QLineEdit("50")

        # Style for labels and inputs
        label_style = """
            font-weight: bold;
            color: #333;
            font-size: 14px;
            border: none;
            margin-left: 0px;
            padding-left: 0px;
        """
        input_style = """
            QLineEdit {
                border: 1px solid #dcdcdc;
                padding: 4px;
                font-size: 12px;
                color: #000; 
                background-color: #fff; 
            }
        """

        self.layer_number_input.setStyleSheet(input_style)
        self.sequence_length_input.setStyleSheet(input_style)
        self.batch_size_input.setStyleSheet(input_style)
        self.epoch_number_input.setStyleSheet(input_style)

        # Add hyperparameter fields to grid
        fields = [
            ("Layer Number", self.layer_number_input),
            ("Sequence Length", self.sequence_length_input),
            ("Batch Size", self.batch_size_input),
            ("Epoch Number", self.epoch_number_input)
        ]

        for row, (label_text, input_field) in enumerate(fields):
            lbl = QLabel(label_text)
            lbl.setStyleSheet(label_style)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  
            input_field.setFixedHeight(24)  
            hyper_layout.addWidget(lbl, row, 0) 
            hyper_layout.addWidget(input_field, row, 1) 
            
        hyper_box.setLayout(hyper_layout)
        middle_layout.addWidget(hyper_box)

        # Save button
        save_btn = QPushButton("Save Hyperparameters")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #dceaf7;
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #dcdcdc; 
                padding: 8px 0px;
            }
            QPushButton:hover {
                background-color: #cce0f5; 
            }
            """)
        save_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  
        save_btn.clicked.connect(self.save_hyperparameters)

        save_btn_layout = QVBoxLayout()
        save_btn_layout.addWidget(save_btn)
        middle_layout.addLayout(save_btn_layout)

        # NN summary display
        self.nn_text_edit = QTextEdit()
        self.nn_text_edit.setReadOnly(True)
        self.nn_text_edit.setStyleSheet("""
            border: 1px solid #dcdcdc; 
            background-color: white; 
            padding: 4px;
        """)
        middle_layout.addWidget(self.nn_text_edit, 1)

        parent_layout.addWidget(middle_panel, 1)

    def clear_classification_loss(self):
        """Clear the classification loss combo box selection."""
        self.classification_loss_combo.setCurrentIndex(-1)

    def clear_regression_loss(self):
        """Clear the regression loss combo box selection."""
        self.regression_loss_combo.setCurrentIndex(-1)

    def create_evaluation_panel(self, parent_layout):
        """
        Create the right panel for evaluation visualization.
        
        Args:
            parent_layout: The layout to add this panel to
        """
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setStyleSheet("background: white")
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        
        # Placeholder for evaluation plot
        plot_placeholder = QLabel("Plot will appear here")
        plot_placeholder.setAlignment(Qt.AlignCenter)
        plot_placeholder.setStyleSheet("""
            background: white; 
            border: 1px solid #dcdcdc;
        """)
        right_layout.addWidget(plot_placeholder, 1)

        # Container for proper margins
        right_panel_container = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_container)
        right_panel_layout.setContentsMargins(0, 0, 7, 0)  
        right_panel_layout.addWidget(right_panel)

        parent_layout.addWidget(right_panel_container, 1)

    def create_training_controls(self):
        """Create the training control buttons at the bottom."""
        proc_row = QHBoxLayout()
        
        # Processor label
        proc_label = QLabel("Current processor: CUDA")
        proc_row.addWidget(proc_label, alignment=Qt.AlignLeft)

        # Training control buttons
        button_layout = QHBoxLayout()
        for name in ["Start", "Stop", "Save Model"]:
            btn = QPushButton(name)
            btn.setStyleSheet("""
                background-color: #dceaf7; 
                border: 2px solid white; 
                font-weight: bold;
            """)
            btn.setMinimumWidth(100)
            button_layout.addWidget(btn)
            
            # Connect buttons to appropriate handlers
            if name == "Start":
                btn.clicked.connect(self.start_training)
            elif name == "Stop":
                btn.clicked.connect(self.stop_training)

        proc_row.addStretch()
        proc_row.addLayout(button_layout)
        proc_row.addStretch()

        # Container for proper margins
        proc_container = QWidget()
        proc_layout = QVBoxLayout(proc_container)
        proc_layout.setContentsMargins(7, 0, 0, 0) 
        proc_layout.addLayout(proc_row)
        
        self.main_layout.addWidget(proc_container)

    def create_training_monitor(self):
        """Create the training progress monitor section."""
        # Monitor frame
        monitor_frame = QFrame()
        monitor_frame.setStyleSheet("""
            background: white; 
            border: 1px solid #dcdcdc;
        """)
        monitor_layout = QVBoxLayout(monitor_frame)
        
        # Example epoch label
        epoch_label = QLabel(f"Epoch [{random.randint(1, 50)}/50] Train Loss: 0.{random.randint(30, 45)} Val Loss: 0.{random.randint(30, 45)}")
        epoch_label.setStyleSheet("border: none;")  
        monitor_layout.addWidget(epoch_label)

        # Example loading label
        loading_label = QLabel("Loading bar (~%), Training time")
        loading_label.setStyleSheet("border: none;")  
        monitor_layout.addWidget(loading_label)

        # Container for label
        training_monitor_container = QWidget()
        training_monitor_layout = QVBoxLayout(training_monitor_container)
        training_monitor_layout.setContentsMargins(7, 0, 0, 0) 
        training_monitor_layout.addWidget(
            QLabel("Training Monitor", parent=self, styleSheet="font-weight: bold; font-style: italic;")
        )

        self.main_layout.addWidget(training_monitor_container)

        # Container for monitor frame
        monitor_container = QFrame()
        monitor_container_layout = QVBoxLayout(monitor_container)
        monitor_container_layout.setContentsMargins(7, 0, 7, 7)  
        monitor_container_layout.addWidget(monitor_frame)

        self.main_layout.addWidget(monitor_container)

    def create_status_bar(self):
        """Create and configure the status bar."""
        status_label = QLabel("Data Monitoring Software version 1.0.13")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.statusBar().addPermanentWidget(status_label, 1)

    def update_progress(self, active_step, completed_steps=None):
        """
        Update the progress indicator showing workflow steps.
        
        Args:
            active_step (str): The currently active step (highlighted in orange)
            completed_steps (list): List of completed steps (shown in green)
        """
        steps = ["Dataset Builder", "NN Designer", "NN Evaluator"]
        completed_steps = completed_steps or []

        parts = []
        for step in steps:
            if step in completed_steps:
                # Completed steps in green bold
                parts.append(f'<span style="color: green; font-weight:bold;">{step}</span>')
            elif step == active_step:
                # Active step in orange bold
                parts.append(f'<span style="color: orange; font-weight:bold;">{step}</span>')
            else:
                # Other steps in normal style
                parts.append(step)

        text = " → ".join(parts)
        self.top_right_label.setText(f"Progress Statement : {text}")

    def go_back_to_start(self):
        """
        Return to the Dataset Builder page, restoring the last state if available.
        """
        self.hide()  # Use hide instead of close to keep the application running
        from windows.dataset_builder_window import DatasetBuilderWindow
        
        # Get the current state before switching
        saved_state = self.get_saved_state()
        
        self.dataset_builder_window = DatasetBuilderWindow(
            start_window_ref=None,
            saved_state=saved_state  # Pass the saved state back to DatasetBuilderWindow
        )
        self.dataset_builder_window.showMaximized()

    def start_training(self):
        """Simulate the training process with a progress bar."""
        if progress_state.nn_designed:
            reply = QMessageBox.question(
                self,
                "Retrain Model?",
                "You have already trained the model. Restarting training will reset evaluator progress. Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self.training_stopped = False  # Reset the stop flag
        progress_state.nn_designed = False
        
        # Create progress dialog
        progress = QProgressDialog("Training Model...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Simulate training progress
        for i in range(101):
            if self.training_stopped:  
                self.statusBar().showMessage("Training Model stopped.", 3000)
                return
            time.sleep(0.01) 
            progress.setValue(i)
            if progress.wasCanceled():
                self.statusBar().showMessage("Training Model cancelled.", 3000)
                return

        self.statusBar().showMessage("Training Model successfully!", 3000)
        progress_state.nn_designed = True
        progress_state.training_started = True  
        self.open_nn_evaluator()

    def stop_training(self):
        """Handle stop training button click with confirmation dialog."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Stop")
        msg_box.setText("Are you sure to stop training?")
        msg_box.setIcon(QMessageBox.Question) 

        # Custom buttons
        discard_button = msg_box.addButton("Discard", QMessageBox.RejectRole)
        yes_button = msg_box.addButton("Yes", QMessageBox.YesRole)
        msg_box.setDefaultButton(discard_button)
        
        # Show dialog and handle response
        msg_box.exec_()

        if msg_box.clickedButton() == yes_button:
            self.training_stopped = True
            print("Training stopped")  
        else:
            print("Stop cancelled")

    def open_nn_evaluator(self):
        """Open the Neural Network Evaluator window."""
        from windows.nn_evaluator_window import NeuralNetworkEvaluator
        self.hide()
        self.nn_evaluator_window = NeuralNetworkEvaluator()
        self.nn_evaluator_window.showMaximized()

    def save_hyperparameters(self):
        # Save current UI state
        self.state["hyperparameters"] = {
            "layer_number": self.layer_number_input.text(),
            "sequence_length": self.sequence_length_input.text(),
            "batch_size": self.batch_size_input.text(),
            "epoch_number": self.epoch_number_input.text()
        }
        
        self.state["optimizer"] = self.optimizer_combo.currentText()
        
        # Save loss function
        if self.classification_loss_combo.currentIndex() >= 0:
            self.state["loss_function"] = self.classification_loss_combo.currentText()
            loss_type = "Classification"
        elif self.regression_loss_combo.currentIndex() >= 0:
            self.state["loss_function"] = self.regression_loss_combo.currentText()
            loss_type = "Regression"
        else:
            self.state["loss_function"] = None
            loss_type = "None"
        
        # Save checkbox states
        self.state["selected_files"] = [
            self.file_list.item(i).text() 
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]
        
        # Update summary text in state
        self.state["summary_text"] = f"""Hyperparameters saved:
• Optimizer: {self.state["optimizer"]}
• Loss Function: {self.state["loss_function"]} ({loss_type})
• Layers: {self.state["hyperparameters"]["layer_number"]}
• Seq Length: {self.state["hyperparameters"]["sequence_length"]}
• Batch Size: {self.state["hyperparameters"]["batch_size"]}
• Epochs: {self.state["hyperparameters"]["epoch_number"]}
• Selected Files: {len(self.state["selected_files"])}/{self.file_list.count()}"""
        
        self.update_summary_display()
        progress_state.nn_designed = True
        QMessageBox.information(self, "Saved", "The hyperparameters have been saved successfully!")

    def restore_state(self):
        """Precisely restore all UI elements"""
        # Restore parameters
        hp = self.state["hyperparameters"]
        self.layer_number_input.setText(hp["layer_number"])
        self.sequence_length_input.setText(hp["sequence_length"])
        self.batch_size_input.setText(hp["batch_size"])
        self.epoch_number_input.setText(hp["epoch_number"])
        
        # Restore optimizer
        if self.optimizer_combo.findText(self.state["optimizer"]) >= 0:
            self.optimizer_combo.setCurrentText(self.state["optimizer"])
        
        # Restore loss function
        if self.state["loss_function"]:
            if self.classification_loss_combo.findText(self.state["loss_function"]) >= 0:
                self.classification_loss_combo.setCurrentText(self.state["loss_function"])
                self.regression_loss_combo.setCurrentIndex(-1)
            elif self.regression_loss_combo.findText(self.state["loss_function"]) >= 0:
                self.regression_loss_combo.setCurrentText(self.state["loss_function"])
                self.classification_loss_combo.setCurrentIndex(-1)
        
        # Restore checkboxes after files load
        QTimer.singleShot(100, self.restore_checkboxes)

    def restore_checkboxes(self):
        """Exactly restore previous checkbox states"""
        if not hasattr(self, 'file_list'):
            return
            
        # Create lookup for performance
        selected_files_set = set(self.state["selected_files"])
        
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Checked if item.text() in selected_files_set else Qt.Unchecked)

    def update_summary_display(self):
        """Always keep summary updated with current state"""
        self.nn_text_edit.setText(self.state["summary_text"])

    def display_saved_summary(self):
        """Display the saved summary in the TextEdit widget."""
        if not (self.state["optimizer"] or self.state["loss_function"] or any(self.state["hyperparameters"].values())):
            return
            
        loss_type = ""
        if self.state["loss_function"]:
            if self.state["loss_function"] in self.loss_function_categories["Classification"]:
                loss_type = "Classification"
            elif self.state["loss_function"] in self.loss_function_categories["Regression"]:
                loss_type = "Regression"
        
        summary = (
            f"Hyperparameters:\n"
            f"• Optimizer: {self.state['optimizer'] or 'Not set'}\n"
            f"• Loss Function: {self.state['loss_function'] or 'Not set'} {f'({loss_type})' if loss_type else ''}\n"
            f"• Layer Number: {self.state['hyperparameters'].get('layer_number', 'Not set')}\n"
            f"• Sequence Length: {self.state['hyperparameters'].get('sequence_length', 'Not set')}\n"
            f"• Batch Size: {self.state['hyperparameters'].get('batch_size', 'Not set')}\n"
            f"• Epoch Number: {self.state['hyperparameters'].get('epoch_number', 'Not set')}\n"
            f"• Selected Files: {', '.join(self.state['selected_files']) if self.state['selected_files'] else 'None'}"
        )
        
        self.nn_text_edit.setText(summary)
        print("Displayed saved summary")

    def auto_select_files(self):
        """Modified auto-select that immediately saves state without a pop-up."""
        try:
            percentage = int(self.training_percentage_input.text().strip('%'))
            total = self.file_list.count()
            to_select = max(1, int(total * percentage / 100))
            
            # Get all current items
            all_items = [self.file_list.item(i) for i in range(total)]
            
            # Random selection
            selected = random.sample(all_items, to_select)
            
            # Update UI and state
            self.state["selected_files"] = []
            for item in all_items:
                checked = item in selected
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                if checked:
                    self.state["selected_files"].append(item.text())
                    
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid percentage")

    def populate_file_list(self):
        """
        Populate the file list with .h5 files from the dataset folder.
        Only called if dataset_path was provided during initialization.
        """
        self.file_list.clear()
        if self.dataset_path and os.path.exists(self.dataset_path):
            for filename in os.listdir(self.dataset_path):
                if filename.endswith(".h5"):
                    item = QListWidgetItem(filename)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.file_list.addItem(item)
        else:
            QMessageBox.warning(self, "Error", "Dataset folder does not exist or is invalid.")

    def populate_file_list_with_paths(self, file_paths):
        """
        Populate the file list with the provided file paths.
        
        Args:
            file_paths (list): List of file paths to display in the file list.
        """
        self.file_list.clear()
        for file_path in file_paths:
            item = QListWidgetItem(os.path.basename(file_path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.file_list.addItem(item)

    def get_saved_state(self):
        """Return complete current state"""
        return self.state

    def on_loss_function_changed(self, is_classification):
        """Handle when either loss function combo box changes."""
        if is_classification:
            # Classification loss was selected - clear regression
            if self.classification_loss_combo.currentIndex() >= 0:
                self.regression_loss_combo.blockSignals(True)  # Temporarily block signals
                self.regression_loss_combo.setCurrentIndex(-1)
                self.regression_loss_combo.blockSignals(False)  # Re-enable signals
                self.state["loss_function"] = self.classification_loss_combo.currentText()
        else:
            # Regression loss was selected - clear classification
            if self.regression_loss_combo.currentIndex() >= 0:
                self.classification_loss_combo.blockSignals(True)  # Temporarily block signals
                self.classification_loss_combo.setCurrentIndex(-1)
                self.classification_loss_combo.blockSignals(False)  # Re-enable signals
                self.state["loss_function"] = self.regression_loss_combo.currentText()

    def clear_loss_function_selection(self):
        """Clear both loss function selections"""
        self.classification_loss_combo.setCurrentIndex(-1)
        self.regression_loss_combo.setCurrentIndex(-1)
        self.state["loss_function"] = None