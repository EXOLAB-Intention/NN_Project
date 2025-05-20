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
    def __init__(self, dataset_path=None):
        super().__init__()
        self.dataset_path = dataset_path  
        self.setWindowTitle("Data Monitoring Software")         
        self.resize(1000, 700)

        # Scroll wrapper for responsiveness
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        scroll.setWidget(container)

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 0, 20, 0)  

        self.setCentralWidget(scroll)

        # Header container
        header_container = QWidget()
        header_container_layout = QVBoxLayout(header_container)
        header_container_layout.setContentsMargins(0, 0, 0, 0)  
        header_container_layout.setSpacing(0)

        # Add the header
        header = Header(active_page="Neural Network Designer", parent_window=self)
        header_container_layout.addWidget(header)

        # Add the header container directly to the main layout 
        main_layout.setContentsMargins(0, 0, 0, 0) 

        main_layout.addWidget(header_container, alignment=Qt.AlignTop)

        # -------- TOP BAR --------
        ''' DEBUT HARRY PROGRESS BAR (ALIGNE AU MM NIVEAU QUE LE TITRE) '''
        # -------- TOP BAR with back button, centered title, and top-right progress label --------
        top_bar_layout = QHBoxLayout()

        # Left widget: back button (left aligned)
        left_widget = QWidget()
        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0,0,0,0)
        back_btn = QPushButton("← Back")
        back_btn.setFixedSize(100, 30)
        back_btn.clicked.connect(self.go_back_to_start)
        left_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        left_widget.setLayout(left_layout)

        # Center widget: title (center aligned)
        center_widget = QWidget()
        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(0,0,0,0)
        title_label = QLabel("Neural Network Designer")
        title_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignLeft)
        center_layout.addWidget(title_label, alignment=Qt.AlignLeft)
        center_widget.setLayout(center_layout)

        # Right widget: progress label (right aligned)
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

        # Ajouter les 3 widgets dans le layout principal
        top_bar_layout.addWidget(left_widget)
        top_bar_layout.addWidget(center_widget, stretch=1)  # stretch pour centrer le titre
        top_bar_layout.addWidget(right_widget)

        main_layout.addLayout(top_bar_layout)

        # Initialiser le label de progression
        self.update_progress(active_step="NN Designer", completed_steps=["Dataset Builder"])
        ''' FIN HARRY PROGRESS BAR (ALIGNE AU MM NIVEAU QUE LE TITRE) '''

        # -------- TOP TITLE ROW --------
        top_row = QHBoxLayout()

        left_title_controls = QHBoxLayout()
        training_label = QLabel("Training/Test Set")
        training_label.setStyleSheet("font-weight: bold; font-style: italic;")
        left_title_controls.addWidget(training_label)
        self.training_percentage_input = QLineEdit("20%")
        self.training_percentage_input.setMaximumWidth(60)
        left_title_controls.addWidget(self.training_percentage_input)
        auto_select_button = QPushButton("Auto Select")
        auto_select_button.setStyleSheet("background-color: #dceaf7; font-size: 12px; font-weight: bold; border: 2px solid white; padding: 4px 4px;")
        auto_select_button.clicked.connect(self.auto_select_files)
        left_title_controls.addWidget(auto_select_button) 

        left_title_controls.addStretch()

        training_container = QWidget()
        training_layout = QVBoxLayout(training_container)
        training_layout.setContentsMargins(7, 0, 0, 0)  
        training_layout.addLayout(left_title_controls)
        top_row.addWidget(training_container, 1)

        middle_title = QLabel("NN Model")
        middle_title.setStyleSheet("font-weight: bold; font-style: italic;")
        top_row.addWidget(middle_title, 1, alignment=Qt.AlignLeft)

        right_title = QLabel("Evaluation Plot")
        right_title.setStyleSheet("font-weight: bold; font-style: italic;")
        top_row.addWidget(right_title, 1, alignment=Qt.AlignLeft)

        main_layout.addLayout(top_row)

        # -------- MAIN CONTENT --------
        content_layout = QHBoxLayout()

        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setStyleSheet("background: white")
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("File name"))
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("border: 1px solid #dcdcdc; background-color: white; padding: 4px;")
        for i in range(12):
            item = QListWidgetItem(f"250501_walkingtest{i+1}.h5")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.file_list.addItem(item)
        left_layout.addWidget(self.file_list)

        left_panel_container = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_container)
        left_panel_layout.setContentsMargins(7, 0, 0, 0) 
        left_panel_layout.addWidget(left_panel)
        content_layout.addWidget(left_panel_container, 1)

        middle_panel = QFrame()
        middle_panel.setFrameShape(QFrame.StyledPanel)
        middle_panel.setStyleSheet("background: white")
        middle_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        middle_layout = QVBoxLayout(middle_panel)

        # Initialize optimizer combo box
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems([
            "Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", 
            "AdamW", "Nadam", "Lion", "AdaBelief", "Yogi"
        ])

                # Initialize two loss combo boxes: one for Classification, one for Regression
        self.classification_loss_combo = QComboBox()
        self.regression_loss_combo = QComboBox()

        # Define loss function categories
        self.loss_function_categories = {
            "Classification": [
                "CrossEntropyLoss", "NLLLoss", "BCELoss", "BCEWithLogitsLoss",
                "KLDivLoss", "MultiLabelMarginLoss", "MultiLabelSoftMarginLoss"
            ],
            "Regression": [
                "MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss",
                "MarginRankingLoss", "CosineEmbeddingLoss"
            ]
        }

        # Populate the classification loss combo
        for loss in self.loss_function_categories["Classification"]:
            self.classification_loss_combo.addItem(loss)

        # Populate the regression loss combo
        for loss in self.loss_function_categories["Regression"]:
            self.regression_loss_combo.addItem(loss)

        # Forcer un seul choix actif à la fois
        self.classification_loss_combo.currentIndexChanged.connect(
            lambda i: self.regression_loss_combo.setCurrentIndex(-1) if i != -1 else None
        )
        self.regression_loss_combo.currentIndexChanged.connect(
            lambda i: self.classification_loss_combo.setCurrentIndex(-1) if i != -1 else None
        )

        # Apply style to combo boxes
        combo_style = """
            QComboBox { border: 1px solid #dcdcdc; padding: 4px; background-color: white; }
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

        # Add optimizer and loss combo boxes to the middle panel
        middle_layout.addWidget(QLabel("<b><i>Optimizer</b></i>"))
        middle_layout.addWidget(self.optimizer_combo)

        # Add labels and combo boxes to the layout
        middle_layout.addWidget(QLabel("<b><i>Classification Loss Function</b></i>"))
        middle_layout.addWidget(self.classification_loss_combo)

        middle_layout.addWidget(QLabel("<b><i>Regression Loss Function</b></i>"))
        middle_layout.addWidget(self.regression_loss_combo)

        # Add Neural Network Model section under optimizer and loss function
        middle_layout.addWidget(QLabel("<b><i>Neural Network Model</b></i>"))

        # Hyperparameters
        hyper_box = QGroupBox("Hyperparameters")
        hyper_box.setStyleSheet("""
            border: 1px solid #dcdcdc;
            border-radius: 5px;
            padding: 15px;
        """)
        hyper_layout = QGridLayout()  
        hyper_layout.setContentsMargins(0, 10, 10, 10)
        hyper_layout.setVerticalSpacing(5)  

        self.layer_number_input = QLineEdit("3")
        self.sequence_length_input = QLineEdit("50")
        self.batch_size_input = QLineEdit("32")
        self.epoch_number_input = QLineEdit("50")

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

        # Save Hyperparameters Button
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

        self.nn_text_edit = QTextEdit()
        self.nn_text_edit.setReadOnly(True)
        self.nn_text_edit.setStyleSheet("border: 1px solid #dcdcdc; background-color: white; padding: 4px;")
        middle_layout.addWidget(self.nn_text_edit, 1)

        content_layout.addWidget(middle_panel, 1)

        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setStyleSheet("background: white")
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        plot_placeholder = QLabel("Plot will appear here")
        plot_placeholder.setAlignment(Qt.AlignCenter)
        plot_placeholder.setStyleSheet("background: white; border: 1px solid #dcdcdc;")
        right_layout.addWidget(plot_placeholder, 1)

        right_panel_container = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_container)
        right_panel_layout.setContentsMargins(0, 0, 7, 0)  
        right_panel_layout.addWidget(right_panel)

        content_layout.addWidget(right_panel_container, 1)

        main_layout.addLayout(content_layout)

        # Processor row
        proc_row = QHBoxLayout()
        proc_label = QLabel("Current processor: CUDA")
        proc_row.addWidget(proc_label, alignment=Qt.AlignLeft)

        button_layout = QHBoxLayout()
        for name in ["Start", "Stop", "Save Model"]:
            btn = QPushButton(name)
            btn.setStyleSheet("background-color: #dceaf7; border: 2px solid white; font-weight: bold;")
            btn.setMinimumWidth(100)
            button_layout.addWidget(btn)
            if name == "Start":
                btn.clicked.connect(self.start_training)  # Connect to start_training
            elif name == "Stop":
                btn.clicked.connect(self.stop_training)  # Connect to stop_training

        proc_row.addStretch()
        proc_row.addLayout(button_layout)
        proc_row.addStretch()

        proc_container = QWidget()
        proc_layout = QVBoxLayout(proc_container)
        proc_layout.setContentsMargins(7, 0, 0, 0) 
        proc_layout.addLayout(proc_row)
        main_layout.addWidget(proc_container)

        # Training Monitor
        monitor_frame = QFrame()
        monitor_frame.setStyleSheet("background: white; border: 1px solid #dcdcdc;")
        monitor_layout = QVBoxLayout(monitor_frame)
        epoch_label = QLabel(f"Epoch [{i+1}/50] Train Loss: 0.{random.randint(30, 45)} Val Loss: 0.{random.randint(30, 45)}")
        epoch_label.setStyleSheet("border: none;")  
        monitor_layout.addWidget(epoch_label)

        loading_label = QLabel("Loading bar (~%), Training time")
        loading_label.setStyleSheet("border: none;")  
        monitor_layout.addWidget(loading_label)

        training_monitor_container = QWidget()
        training_monitor_layout = QVBoxLayout(training_monitor_container)
        training_monitor_layout.setContentsMargins(7, 0, 0, 0) 
        training_monitor_layout.addWidget(QLabel("Training Monitor", parent=self, styleSheet="font-weight: bold; font-style: italic;"))

        main_layout.addWidget(training_monitor_container)

        monitor_container = QFrame()
        monitor_container_layout = QVBoxLayout(monitor_container)
        monitor_container_layout.setContentsMargins(7, 0, 7, 7)  
        monitor_container_layout.addWidget(monitor_frame)

        main_layout.addWidget(monitor_container)

        # Status bar
        status_label = QLabel("Data Monitoring Software version 1.0.13")
        status_label.setAlignment(Qt.AlignCenter)

        status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.statusBar().addPermanentWidget(status_label, 1)

        # Populate file list with .h5 files from the dataset folder
        if self.dataset_path:
            self.populate_file_list()
    
    ''' DEBUT HARRY PROGRESS BAR '''
    def update_progress(self, active_step, completed_steps=None):
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
    ''' FIN HARRY PROGRESS BAR '''

    def go_back_to_start(self):
        """Return to the start window."""
        self.hide()
        from windows.start_window import StartWindow
        self.start_window = StartWindow()
        
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
        progress = QProgressDialog("Training Model...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for i in range(101):
            if self.training_stopped:  # Check if training was stopped
                self.statusBar().showMessage("Training Model stopped.", 3000)
                return
            time.sleep(0.01)  # Simulate a heavy task
            progress.setValue(i)
            if progress.wasCanceled():
                self.statusBar().showMessage("Training Model cancelled.", 3000)
                return

        self.statusBar().showMessage("Training Model successfully!", 3000)
        progress_state.nn_designed = True
        self.open_nn_evaluator()

    def stop_training(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirm Stop")
        msg_box.setText("Are you sure to stop training?")
        msg_box.setIcon(QMessageBox.Question) 

        discard_button = msg_box.addButton("Discard", QMessageBox.RejectRole)
        yes_button = msg_box.addButton("Yes", QMessageBox.YesRole)
        msg_box.setDefaultButton(discard_button)
        msg_box.exec_()

        if msg_box.clickedButton() == yes_button:
            print("Training stopped")  
        else:
            print("Stop cancelled")
   

    def open_nn_evaluator(self):
        """Start training and open the evaluator window."""
        from windows.nn_evaluator_window import NeuralNetworkEvaluator
        self.hide()
        self.nn_evaluator_window = NeuralNetworkEvaluator()
        self.nn_evaluator_window.showMaximized()
    ''' FIN HARRY PROGRESS BAR '''

    def go_back_to_start(self):
        self.hide()
        from windows.start_window import StartWindow
        self.start_window = StartWindow()
        self.start_window.showMaximized()

    def save_hyperparameters(self):
        """
        Save the hyperparameters entered by the user and display them.
        """
        # Lire les champs
        layer_number = self.layer_number_input.text()
        sequence_length = self.sequence_length_input.text()
        batch_size = self.batch_size_input.text()
        epoch_number = self.epoch_number_input.text()
        optimizer = self.optimizer_combo.currentText().strip()
        classification_loss = self.classification_loss_combo.currentText().strip()
        regression_loss = self.regression_loss_combo.currentText().strip()

        # Vérifier que tous les champs de base sont remplis
        if not layer_number or not sequence_length or not batch_size or not epoch_number or not optimizer:
            QMessageBox.warning(
                self,
                "Incomplete Hyperparameters",
                "Please fill in all hyperparameters before saving."
            )
            return

        # Vérifier que l'utilisateur a sélectionné exactement une fonction de perte
        if (not classification_loss and not regression_loss) or (classification_loss and regression_loss):
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Please select either a Classification loss or a Regression loss — not both."
            )
            return

        # Sélectionner la fonction de perte active
        if classification_loss:
            loss_function = classification_loss
            loss_type = "Classification"
        else:
            loss_function = regression_loss
            loss_type = "Regression"

        # Afficher un résumé
        summary = (
            f"Hyperparameters saved:\n"
            f"• Optimizer: {optimizer}\n"
            f"• Loss Function: {loss_function} ({loss_type})\n"
            f"• Layer Number: {layer_number}\n"
            f"• Sequence Length: {sequence_length}\n"
            f"• Batch Size: {batch_size}\n"
            f"• Epoch Number: {epoch_number}"
        )

        self.nn_text_edit.setText(summary)
        print(summary)

        QMessageBox.information(self, "Hyperparameters Saved", "The hyperparameters have been saved successfully.")


    def auto_select_files(self):
        try:
            percentage = float(self.training_percentage_input.text().strip().replace('%', ''))
            if not (0 < percentage <= 100):
                raise ValueError
            total = self.file_list.count()
            to_check = round((percentage / 100) * total)
            all_items = [self.file_list.item(i) for i in range(total)]
            selected = random.sample(all_items, to_check)
            for item in all_items:
                item.setCheckState(Qt.Checked if item in selected else Qt.Unchecked)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid percentage between 0 and 100.")

    def populate_file_list(self):
        """
        Populate the file list with .h5 files from the dataset folder.
        """
        self.file_list.clear()
        if os.path.exists(self.dataset_path):
            for filename in os.listdir(self.dataset_path):
                if filename.endswith(".h5"):
                    item = QListWidgetItem(filename)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.file_list.addItem(item)
