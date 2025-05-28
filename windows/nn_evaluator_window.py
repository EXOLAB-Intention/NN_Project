from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QGridLayout, QLabel, QWidget, QSpacerItem, QSizePolicy, QFileDialog, QPushButton, QHBoxLayout, QStackedWidget, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from widgets.header import Header 
import windows.progress_state as progress_state
import matplotlib.pyplot as plt

int_to_phase = {
    0: 'Stand',
    1: 'Stand-to-Sit',
    2: 'Sit',
    3: 'Sit-to-Stand',
    4: 'Stand-to-Walk',
    5: 'Walk',
    6: 'Walk-to-Stand'
}


class NeuralNetworkEvaluator(QMainWindow):
    def __init__(self, saved_state=None):
        """
        Initialize the Neural Network Evaluator window.
        This window allows users to evaluate trained neural networks, visualize results, and compare models.
        """
        
        super().__init__()
        self.setWindowTitle("Data Monitoring Software")
        self.setGeometry(100, 100, 1200, 800)
        
        self.test_results = progress_state.test_results

        # Central widget and layout
        central_widget = QWidget()
        parent_layout = QVBoxLayout()  
        parent_layout.setContentsMargins(0, 0, 0, 0)  

        # Header Section
        header_container = QWidget()
        header_container_layout = QVBoxLayout()

        # Remove margins and spacing for the header container
        header_container_layout.setContentsMargins(0, 0, 0, 0)
        header_container_layout.setSpacing(0)

        header = Header(active_page="Neural Network Evaluator", parent_window=self)
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        header_container_layout.addWidget(header)
        header_container.setLayout(header_container_layout)
        header_container.setContentsMargins(0, 0, 0, 0)  

        parent_layout.addWidget(header_container)  

        # Main Content Section
        main_content_container = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)  
        self.main_layout.setSpacing(10)  

        title = QLabel("Neural Network Evaluator")
        title_font = QFont("Arial", 18)
        title_font.setBold(True)
        title.setFont(title_font)

        # Label to display progress (aligned to the right of the title)
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

        # Add title and progress label in the same row
        header_layout = QHBoxLayout()
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.top_right_label)

        self.main_layout.addLayout(header_layout)

        # Update the progress label
        self.update_progress_label(active_step="NN Evaluator", completed_steps=["Dataset Builder", "NN Designer"])

        # Menu buttons
        self.add_menu_buttons()

        # Add a stacked widget for menu content
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Add an empty page as the default page
        self.add_empty_page()

        # Add pages to the stacked widget
        self.add_training_overview_page()
        self.add_evaluate_test_set_page()
        self.add_compare_models_page()

        # Set the empty page as the default page
        self.stacked_widget.setCurrentIndex(0)

        # Footer
        footer = QLabel("Data Monitoring Software version 1.0.13")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: gray; font-size: 12px; padding: 10px;")
        self.main_layout.addWidget(footer)

        # Add the main content to the container
        main_content_container.setLayout(self.main_layout)
        parent_layout.addWidget(main_content_container)

        # Set layout
        central_widget.setLayout(parent_layout)
        self.setCentralWidget(central_widget)

        # Add menu bar
        self.add_menu_bar()
        self.state = saved_state or {}

    def update_progress_label(self, active_step, completed_steps=None):
        """
        Update the progress label to show the current workflow step.
        Completed steps are displayed in green, the active step in orange, and others in normal style.

        Args:
            active_step (str): The currently active step.
            completed_steps (list): List of completed steps.
        """
        steps = ["Dataset Builder", "NN Designer", "NN Evaluator"]
        completed_steps = completed_steps or []

        parts = []
        for step in steps:
            if step in completed_steps:
                parts.append(f'<span style="color: green; font-weight:bold;">{step}</span>')
            elif step == active_step:
                parts.append(f'<span style="color: orange; font-weight:bold;">{step}</span>')
            else:
                parts.append(step)

        text = " → ".join(parts)
        self.top_right_label.setText(f"Progress Statement : {text}")

    def add_menu_buttons(self):
        """
        Add the main menu buttons for navigating between different evaluation sections.
        """
        menu_button_layout = QHBoxLayout()

        # Training Overview button
        self.training_overview_button = QPushButton("Training Overview")
        self.training_overview_button.setStyleSheet(self.get_button_style())
        self.training_overview_button.clicked.connect(lambda: [self.switch_menu(1), self.update_menu_button_styles(0)])
        menu_button_layout.addWidget(self.training_overview_button)

        # Evaluate on Test Set button
        self.evaluate_test_set_button = QPushButton("Evaluate on Test Set")
        self.evaluate_test_set_button.setStyleSheet(self.get_button_style())
        self.evaluate_test_set_button.clicked.connect(lambda: [self.switch_menu(2), self.update_menu_button_styles(1)])
        menu_button_layout.addWidget(self.evaluate_test_set_button)

        # Compare Models button
        self.compare_models_button = QPushButton("Compare Models")
        self.compare_models_button.setStyleSheet(self.get_button_style())
        self.compare_models_button.clicked.connect(lambda: [self.switch_menu(3), self.update_menu_button_styles(2)])
        menu_button_layout.addWidget(self.compare_models_button)

        self.main_layout.addLayout(menu_button_layout)

    def add_menu_bar(self):
        """
        Add the menu bar with options for file operations and navigation.
        """
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Dataset actions
        create_dataset_action = file_menu.addAction("Create new dataset")
        create_dataset_action.triggered.connect(self.create_new_dataset)

        load_dataset_action = file_menu.addAction("Load existing dataset")
        load_dataset_action.triggered.connect(self.load_existing_dataset)

        file_menu.addSeparator()

        save_dataset_action = file_menu.addAction("Save dataset")
        save_dataset_action.setEnabled(False)  # Disabled by default
        save_dataset_action.triggered.connect(self.save_dataset)

        save_dataset_as_action = file_menu.addAction("Save dataset as...")
        save_dataset_as_action.setEnabled(False)  # Disabled by default
        save_dataset_as_action.triggered.connect(self.save_dataset_as)

        # Model actions
        load_model_action = file_menu.addAction("Load existing Model")
        load_model_action.triggered.connect(self.load_existing_model)

        file_menu.addSeparator()

        save_model_action = file_menu.addAction("Save current Model")
        save_model_action.setEnabled(False)  # Disabled by default
        save_model_action.triggered.connect(self.save_model)

        save_model_as_action = file_menu.addAction("Save current Model as...")
        save_model_as_action.setEnabled(False)  # Disabled by default
        save_model_as_action.triggered.connect(self.save_model_as)

        # Add more options if needed
        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        menubar.addMenu("Edit")
        menubar.addMenu("Options")

    def get_button_style(self):
        """
        Return the style for the menu buttons.
        """
        return """
            QPushButton {
                background-color: #e7f3ff;
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                font-size: 16px;
                padding: 12px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #d0e7ff;
            }
            QPushButton:pressed {
                background-color: #b3d4ff;
            }
        """

    def switch_menu(self, index):
        """
        Switch the displayed menu in the stacked widget.

        Args:
            index (int): Index of the menu to display.
        """
        self.stacked_widget.setCurrentIndex(index)

    def update_menu_button_styles(self, active_index):
        """
        Update the styles of the menu buttons to highlight the active one.

        Args:
            active_index (int): Index of the active button.
        """
        buttons = [self.training_overview_button, self.evaluate_test_set_button, self.compare_models_button]
        for i, button in enumerate(buttons):
            if i == active_index:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: white;
                        border: 2px solid #a6c8ff;
                        border-radius: 5px;
                        font-size: 16px;
                        padding: 12px;
                        min-width: 150px;
                    }
                    QPushButton:hover {
                        background-color: #f0f8ff;
                    }
                    QPushButton:pressed {
                        background-color: #e0f0ff;
                    }
                """)
            else:
                button.setStyleSheet(self.get_button_style())

    def add_training_overview_page(self):
        """
        Add the Training Overview page to the stacked widget.
        This page displays training results and visualizations.
        """
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Add the "Load Model" button
        load_model_button = QPushButton("Load Model")
        load_model_button.setStyleSheet("""
            QPushButton {
                background-color: #e7f3ff;
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                font-size: 16px;
                padding: 12px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #d0e7ff;
            }
            QPushButton:pressed {
                background-color: #b3d4ff;
            }
        """)
        load_model_button.clicked.connect(self.load_existing_model)
        layout.addWidget(load_model_button, alignment=Qt.AlignLeft)

        # Import des données
        ground_truth = progress_state.test_results.get("y_true", [])
        predicted = progress_state.test_results.get("y_pred", [])

        print("DEBUG y_true:", ground_truth[:10])
        print("DEBUG y_pred:", predicted[:10])
        print("DEBUG len(y_true):", len(ground_truth))
        print("DEBUG len(y_pred):", len(predicted))

        # Affichage des courbes via la fonction utilitaire
        plot_canvas = FigureCanvas(Figure(figsize=(12, 5)))
        self.plot_prediction_and_curves(plot_canvas.figure)
        layout.addWidget(plot_canvas)

        page.setLayout(layout)
        self.stacked_widget.addWidget(page)

    def create_new_dataset(self):
        """
        Open the DatasetBuilderWindow to create a new dataset.
        """
        from windows.dataset_builder_window import DatasetBuilderWindow
        self.dataset_builder = DatasetBuilderWindow(start_window_ref=self)
        self.dataset_builder.show()
        self.close()

    def load_existing_dataset(self):
        """
        Open a file dialog to load an existing dataset.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # Open in read-only mode
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Existing Dataset",
            "",
            "Dataset Files (*.csv *.json *.txt);;All Files (*)",
            options=options
        )
        if file_path:
            print(f"Dataset loaded from: {file_path}")
            # Add logic to process the dataset file here

    def save_dataset(self):
        """
        Save the current dataset.
        """
        print("Saving the dataset...")

    def save_dataset_as(self):
        """
        Save the current dataset with a new name.
        """
        print("Saving the dataset as...")

    def load_existing_model(self):
        """
        Open a file dialog to load an existing model.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # Open in read-only mode
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Existing Model",
            "",
            "Model Files (*.h5 *.pt *.onnx);;All Files (*)",
            options=options
        )
        if file_path:
            print(f"Model loaded from: {file_path}")
            # Add logic to process the model file here

    def save_model(self):
        """
        Save the current model.
        """
        print("Saving the current model...")

    def save_model_as(self):
        """
        Save the current model with a new name.
        """
        print("Saving the current model as...")

    '''
    def add_evaluate_test_set_page(self):
        """
        Add the Evaluate on Test Set page to the stacked widget.
        This page allows users to evaluate the model on a test dataset.
        """
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Main grid layout
        main_grid = QGridLayout()
        main_grid.setHorizontalSpacing(20)  
        main_grid.setVerticalSpacing(10)   
        main_grid.setContentsMargins(10, 10, 10, 10) 

        # Classification Section
        classification_label = QLabel("<b><i>Classification</i></b>")
        classification_label.setAlignment(Qt.AlignLeft)
        classification_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) 
        main_grid.addWidget(classification_label, 0, 0, 1, 2)

        # Accuracy row
        accuracy_label = QLabel("<b><i>Accuracy</i></b>")
        accuracy_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        accuracy_value = QLabel("99.8%")
        accuracy_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        accuracy_value.setFixedSize(80, 30)  
        main_grid.addWidget(accuracy_label, 1, 0, alignment=Qt.AlignRight)  
        main_grid.addWidget(accuracy_value, 1, 1, alignment=Qt.AlignLeft)  

        main_grid.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum), 1, 2)

        # Confusion Matrix row
        confusion_matrix_label = QLabel("<b><i>Confusion matrix</i></b>")
        confusion_matrix_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        confusion_matrix_button = QPushButton("Visualize confusion matrix")
        confusion_matrix_button.setFixedSize(120, 120)  
        confusion_matrix_button.setStyleSheet("""
            QPushButton {
                background-color: #add8e6;
                border: 1px solid #000;
                font-size: 10px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #87ceeb;
            }
        """)
        main_grid.addWidget(confusion_matrix_label, 2, 0, alignment=Qt.AlignRight)  
        main_grid.addWidget(confusion_matrix_button, 2, 1, alignment=Qt.AlignLeft)  

        main_grid.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum), 2, 2)

        # Regression Section
        regression_label = QLabel("<b><i>Regression</b></i>")
        regression_label.setAlignment(Qt.AlignLeft)
        regression_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        main_grid.addWidget(regression_label, 3, 0, 1, 2)

        # MSE row
        mse_label = QLabel("<b><i>MSE</i></b>")
        mse_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        mse_value = QLabel("99.8%")
        mse_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        mse_value.setFixedSize(80, 30)  
        main_grid.addWidget(mse_label, 4, 0, alignment=Qt.AlignRight)  
        main_grid.addWidget(mse_value, 4, 1, alignment=Qt.AlignLeft) 

        # RMSE row
        rmse_label = QLabel("<b><i>RMSE</i></b>")
        rmse_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  
        rmse_value = QLabel("99.8%")
        rmse_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        rmse_value.setFixedSize(80, 30) 
        main_grid.addWidget(rmse_label, 5, 0, alignment=Qt.AlignRight) 
        main_grid.addWidget(rmse_value, 5, 1, alignment=Qt.AlignLeft) 

        # Regression plot
        regression_plot = FigureCanvas(Figure(figsize=(12, 10)))  
        regression_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  
        self.plot_prediction_and_curves(regression_plot.figure)

        regression_plot.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

        main_grid.addWidget(regression_plot, 6, 0, 1, 2)  

        main_grid.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), 7, 0, 1, 2)

        # Add the grid layout to the main layout
        layout.addLayout(main_grid)
        layout.setContentsMargins(0, 0, 0, 0)  
        page.setLayout(layout)
        self.stacked_widget.addWidget(page)
    '''
    
    

    def add_evaluate_test_set_page(self):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, accuracy_score
        import numpy as np

        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Main grid layout
        main_grid = QGridLayout()
        main_grid.setHorizontalSpacing(20)
        main_grid.setVerticalSpacing(10)
        main_grid.setContentsMargins(10, 10, 10, 10)

        # === Data
        y_true = np.array(self.test_results.get("y_true", []))
        y_pred = np.array(self.test_results.get("y_pred", []))

        # === Metrics
        if len(y_true) > 0 and len(y_pred) > 0:
            try:
                acc = accuracy_score(y_true, y_pred)
            except:
                acc = 0.0
            try:
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
            except:
                mse = rmse = 0.0
        else:
            acc = mse = rmse = 0.0

        # === Classification Section
        classification_label = QLabel("<b><i>Classification</i></b>")
        classification_label.setAlignment(Qt.AlignLeft)
        classification_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_grid.addWidget(classification_label, 0, 0, 1, 2)

        # Accuracy row
        accuracy_label = QLabel("<b><i>Accuracy</i></b>")
        accuracy_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        accuracy_value = QLabel(f"{acc*100:.2f} %")
        accuracy_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        accuracy_value.setFixedSize(80, 30)
        main_grid.addWidget(accuracy_label, 1, 0, alignment=Qt.AlignRight)
        main_grid.addWidget(accuracy_value, 1, 1, alignment=Qt.AlignLeft)
        main_grid.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum), 1, 2)

        # Confusion Matrix row (centrée sur 2 colonnes)
        confusion_matrix_label = QLabel("<b><i>Confusion matrix</i></b>")
        confusion_matrix_label.setAlignment(Qt.AlignLeft)
        confusion_matrix_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_grid.addWidget(confusion_matrix_label, 2, 0, 1, 2)

        if len(y_true) > 0 and len(y_pred) > 0:
            cm_canvas = self.plot_confusion_matrix(y_true, y_pred)
            main_grid.addWidget(cm_canvas, 3, 0, 1, 2, alignment=Qt.AlignCenter)
        else:
            main_grid.addWidget(QLabel("Confusion matrix not available."), 3, 0, 1, 2)

        # Regression Section
        regression_label = QLabel("<b><i>Regression</b></i>")
        regression_label.setAlignment(Qt.AlignLeft)
        regression_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_grid.addWidget(regression_label, 4, 0, 1, 2)

        # MSE row
        mse_label = QLabel("<b><i>MSE</i></b>")
        mse_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        mse_value = QLabel(f"{mse:.4f}")
        mse_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        mse_value.setFixedSize(80, 30)
        main_grid.addWidget(mse_label, 5, 0, alignment=Qt.AlignRight)
        main_grid.addWidget(mse_value, 5, 1, alignment=Qt.AlignLeft)

        # RMSE row
        rmse_label = QLabel("<b><i>RMSE</i></b>")
        rmse_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rmse_value = QLabel(f"{rmse:.4f}")
        rmse_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white; font-size: 10px;")
        rmse_value.setFixedSize(80, 30)
        main_grid.addWidget(rmse_label, 6, 0, alignment=Qt.AlignRight)
        main_grid.addWidget(rmse_value, 6, 1, alignment=Qt.AlignLeft)

        main_grid.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), 7, 0, 1, 2)

        # Add the grid layout to a horizontal box to center it
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(main_grid)
        hbox.addStretch(1)

        # Replace layout.addLayout(main_grid) by:
        layout.addStretch(1)
        layout.addLayout(hbox)
        layout.addStretch(1)
        layout.setAlignment(Qt.AlignCenter)

        page.setLayout(layout)
        self.stacked_widget.addWidget(page)


    def add_compare_models_page(self):
        """
        Add the Compare Models page to the stacked widget.
        This page allows users to compare different trained models.
        """
        page = QWidget()
        layout = QVBoxLayout()

        comparison_layout = QHBoxLayout()
        
        # Model A
        model_a_layout = QVBoxLayout()
        model_a_label = QLabel("<b><i>Model A</b></i>")
        model_a_layout.addWidget(model_a_label, alignment=Qt.AlignCenter)
        
        model_a_selector = QComboBox()
        model_a_selector.addItems(["Select the model...", "Model 1", "Model 2", "Model 3"])
        model_a_selector.setStyleSheet("""
            QComboBox {
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                padding: 5px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #333;
            }
            QComboBox:hover {
                border: 1px solid #87ceeb;
                background-color: #ffffff;
            }
            QComboBox::drop-down {
                border-left: 1px solid #a6c8ff;
                background-color: #e7f3ff;
            }
            QComboBox::down-arrow {
                image: url(assets/arrow_down.png); /* Replace with your arrow icon path */
                width: 10px;
                height: 10px;
            }
        """)
        model_a_layout.addWidget(model_a_selector)
        
        model_a_plot = FigureCanvas(Figure(figsize=(10, 8)))  
        model_a_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_step_prediction_vs_ground_truth(
            model_a_plot.figure,
            progress_state.test_results.get("y_true", []),
            progress_state.test_results.get("y_pred", [])
        )
        model_a_layout.addWidget(model_a_plot)
        
        model_a_layout.addWidget(QLabel("<b><i>Accuracy</b></i>"), alignment=Qt.AlignLeft)
        model_a_layout.addWidget(QLabel("<b><i>Used data</b></i>"), alignment=Qt.AlignLeft)
        model_a_layout.addWidget(QLabel("<b><i>Summary of Hyperparameters</b></i>"), alignment=Qt.AlignLeft)
        
        comparison_layout.addLayout(model_a_layout)
        
        # Model B
        model_b_layout = QVBoxLayout()
        model_b_label = QLabel("<b><i>Model B</b></i>")
        model_b_layout.addWidget(model_b_label, alignment=Qt.AlignCenter)
        
        model_b_selector = QComboBox()
        model_b_selector.addItems(["Select the model...", "Model 1", "Model 2", "Model 3"])
        model_b_selector.setStyleSheet("""
            QComboBox {
                border: 1px solid #a6c8ff;
                border-radius: 5px;
                padding: 5px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #333;
            }
            QComboBox:hover {
                border: 1px solid #87ceeb;
                background-color: #ffffff;
            }
            QComboBox::drop-down {
                border-left: 1px solid #a6c8ff;
                background-color: #e7f3ff;
            }
            QComboBox::down-arrow {
                image: url(assets/arrow_down.png); /* Replace with your arrow icon path */
                width: 10px;
                height: 10px;
            }
        """)
        model_b_layout.addWidget(model_b_selector)
        
        model_b_plot = FigureCanvas(Figure(figsize=(10, 8))) 
        model_b_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_step_prediction_vs_ground_truth(
            model_b_plot.figure,
            progress_state.test_results.get("y_true", []),
            progress_state.test_results.get("y_pred", [])
        )
        model_b_layout.addWidget(model_b_plot)
        
        model_b_layout.addWidget(QLabel("<b><i>Accuracy</b></i>"), alignment=Qt.AlignLeft)
        model_b_layout.addWidget(QLabel("<b><i>Used data</b></i>"), alignment=Qt.AlignLeft)
        model_b_layout.addWidget(QLabel("<b><i>Summary of Hyperparameters</b></i>"), alignment=Qt.AlignLeft)
        
        comparison_layout.addLayout(model_b_layout)
        
        layout.addLayout(comparison_layout)
        page.setLayout(layout)
        self.stacked_widget.addWidget(page)

    def add_empty_page(self):
        """
        Add an empty page to the stacked widget as the default page.
        """
        empty_page = QWidget()
        self.stacked_widget.addWidget(empty_page)

    def resizeEvent(self, event):
        """
        Handle window resize events to adjust the plots dynamically.

        Args:
            event: The resize event.
        """
        for i in range(self.stacked_widget.count()):
            page = self.stacked_widget.widget(i)
            if page:
                page.updateGeometry()
        super().resizeEvent(event)

    def plot_prediction_and_curves(self, fig):
        import numpy as np
        fig.clear()
        axs = fig.subplots(nrows=1, ncols=2)

        ground_truth = progress_state.test_results.get("y_true", [])
        predicted = progress_state.test_results.get("y_pred", [])

        if len(ground_truth) > 0 and len(predicted) > 0:
            y_true = np.array(ground_truth, dtype=int)
            y_pred = np.array(predicted, dtype=int)
            x = np.arange(len(y_true))

            # 🟦 Scatter plot façon step plot (à gauche)
            axs[0].scatter(x, y_true, s=8, color="black", label="Ground Truth", alpha=0.7)
            axs[0].scatter(x, y_pred, s=8, color="red", label="Prediction", alpha=0.5)
            axs[0].set_yticks(list(int_to_phase.keys()))
            axs[0].set_yticklabels([int_to_phase[i] for i in int_to_phase])
            axs[0].set_xlabel("Time (index)")
            axs[0].set_ylabel("Phase")
            axs[0].set_title("Ground Truth & Prediction (Scatter)")
            axs[0].legend()
            axs[0].grid(True, linestyle='--', alpha=0.3)

            # 🟧 Step plot (à droite)
            axs[1].step(x, y_true, where='post', label='Ground Truth', color='black')
            axs[1].step(x, y_pred, where='post', label='Prediction', color='red', alpha=0.7)
            axs[1].set_yticks(list(int_to_phase.keys()))
            axs[1].set_yticklabels([int_to_phase[i] for i in int_to_phase])
            axs[1].set_xlabel("Time (index)")
            axs[1].set_ylabel("Phase")
            axs[1].set_title("Prediction vs Ground Truth (Step)")
            axs[1].legend()
            axs[1].grid(True, linestyle='--', alpha=0.3)
        else:
            axs[0].text(0.5, 0.5, "Test results not available.\nTrain the model first.", ha='center', va='center')
            axs[1].axis('off')

        fig.tight_layout()


    def get_saved_state(self):
        return self.state if hasattr(self, "state") else {}

    def plot_prediction_vs_ground_truth(self, fig, ground_truth, predicted):
        fig.clear()
        axs = fig.subplots(nrows=1, ncols=2)

        # 🟦 Courbe temporelle
        axs[0].plot(ground_truth, label="Ground Truth", linewidth=1)
        axs[0].plot(predicted, label="Prediction", linestyle="--", linewidth=1)
        axs[0].set_title("Ground Truth vs Prediction")
        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("Forward Speed (m/s)")
        axs[0].legend()
        axs[0].grid(True)

        # 🟫 Scatter plot
        axs[1].scatter(ground_truth, predicted, alpha=0.6, s=10)
        axs[1].plot([min(ground_truth), max(ground_truth)],
                    [min(ground_truth), max(ground_truth)],
                    'k--', linewidth=1)
        axs[1].set_title("Predicted vs Ground Truth")
        axs[1].set_xlabel("Ground Truth (m/s)")
        axs[1].set_ylabel("Prediction (m/s)")
        axs[1].grid(True)

        fig.tight_layout()

    def plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 8), title="Confusion Matrix (Phase Classification)"):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import numpy as np

        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        cm = confusion_matrix(y_true, y_pred, labels=list(int_to_phase.keys()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(int_to_phase.values()))
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        ax.set_title("")  # On met le titre global avec suptitle
        fig.suptitle(title, fontsize=16, y=0.93)
        ax.set_xticklabels(list(int_to_phase.values()), rotation=45, ha="right")
        fig.subplots_adjust(top=0.88, bottom=0.25, right=0.88)
        return FigureCanvas(fig)

    def plot_step_prediction_vs_ground_truth(self, fig, ground_truth, predicted):
        import numpy as np
        fig.clear()
        ax = fig.add_subplot(111)
        y_true = np.array(ground_truth, dtype=int)
        y_pred = np.array(predicted, dtype=int)
        x = np.arange(len(y_true))

        ax.step(x, y_true, where='post', label='Ground Truth', color='black')
        ax.step(x, y_pred, where='post', label='Prediction', color='red', alpha=0.7)
        ax.set_yticks(list(int_to_phase.keys()))
        ax.set_yticklabels([int_to_phase[i] for i in int_to_phase])
        ax.set_xlabel("Time (index)")
        ax.set_ylabel("Phase")
        ax.set_title("Prediction vs Ground Truth (Step)", fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()


