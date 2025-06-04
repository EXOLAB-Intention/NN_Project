from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QGridLayout, QLabel, QWidget, 
    QSpacerItem, QSizePolicy, QFileDialog, QPushButton, QHBoxLayout, 
    QStackedWidget, QComboBox,QMessageBox,QDialog, QTextBrowser,QMenu
)
from PyQt5.QtCore import QSize
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtGui import QPixmap, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
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
        from PyQt5.QtWidgets import QScrollArea, QFrame

        # Create a scrollable page
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)  # Remove border
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Disable horizontal scroll

        # Main container widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(30, 0, 30, 0)  
        container_layout.setSpacing(20)

        # Create a centered content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)  

        # First plot - Prediction vs Ground Truth
        plot1_canvas = FigureCanvas(Figure(figsize=(10, 6)))  # Slightly reduced width
        fig1 = plot1_canvas.figure
        self.plot_prediction_and_curves(fig1)
        
        # Adjust layout with more padding
        fig1.tight_layout(pad=4.0)
        plot1_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create a container with padding
        plot1_container = QWidget()
        plot1_container.setMinimumHeight(500)
        plot1_container_layout = QHBoxLayout(plot1_container)
        plot1_container_layout.setContentsMargins(20, 20, 20, 20)  # Padding around plot
        plot1_container_layout.addWidget(plot1_canvas)
        plot1_container.setMaximumWidth(1800)  # Reduced maximum width
        content_layout.addWidget(plot1_container, 0, Qt.AlignCenter)

        # Second plot - Scatter plot with better label handling
        plot2_canvas = FigureCanvas(Figure(figsize=(10, 6)))  # Slightly reduced width
        fig2 = plot2_canvas.figure
        self.prediction_scatter_plot_canvas(fig2)
        
        # Adjust x-axis labels and layout
        ax2 = fig2.gca()
        labels = [label.replace("-", "-\n") if "-" in label else label for label in [item.get_text() for item in ax2.get_xticklabels()]]
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)  # 0 degree rotation
        
        # Add more padding at the bottom for labels
        fig2.subplots_adjust(bottom=0.1)
        fig2.tight_layout(pad=4.0)
        
        plot2_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create a container with padding
        plot2_container = QWidget()
        plot2_container.setMinimumHeight(500)
        plot2_container_layout = QHBoxLayout(plot2_container)
        plot2_container_layout.setContentsMargins(20, 20, 20, 30)  # Extra bottom padding for labels
        plot2_container_layout.addWidget(plot2_canvas)
        plot2_container.setMaximumWidth(1800)  
        content_layout.addWidget(plot2_container, 0, Qt.AlignCenter)

        # Add stretch to push content up
        content_layout.addStretch()

        # Add the content widget to the container with centering
        container_layout.addWidget(content_widget, 0, Qt.AlignCenter)
        container_layout.addStretch()

        # Set the container as the scroll area's widget
        scroll_area.setWidget(container)

        # Add the scroll area to the stacked widget
        self.stacked_widget.addWidget(scroll_area)

        # Adjust the figure margins when resized
        def resize_plots():
            # First plot adjustments
            fig1 = plot1_canvas.figure
            fig1.tight_layout(pad=3.0)
            
            # Second plot adjustments
            fig2 = plot2_canvas.figure
            ax2 = fig2.gca()
            labels = [label.replace("-", "-\n") if "-" in label else label for label in [item.get_text() for item in ax2.get_xticklabels()]]
            ax2.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
            fig2.subplots_adjust(bottom=0.25)
            fig2.tight_layout(pad=3.0)
            
            plot1_canvas.draw()
            plot2_canvas.draw()

        # Install an event filter for the scroll area
        scroll_area.viewport().installEventFilter(self)

        # Store the resize callback
        self.resize_callback = resize_plots

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

    def add_evaluate_test_set_page(self):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, accuracy_score
        import numpy as np

        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 0)

        # Main grid layout
        main_grid = QGridLayout()
        main_grid.setHorizontalSpacing(10)
        main_grid.setVerticalSpacing(10)
        main_grid.setContentsMargins(50, 10, 20, 10)  

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

        # === Classification Section (moved left 50px)
        classification_label = QLabel("<b><i>Classification</i></b>")
        classification_label.setAlignment(Qt.AlignLeft)
        classification_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        classification_label.setContentsMargins(-50, 0, 0, 0)  
        classification_label.setStyleSheet("font-size: 17px;")  
        main_grid.addWidget(classification_label, 0, 0, 1, 1)

        # Accuracy row - using a single cell with horizontal layout
        accuracy_layout = QHBoxLayout()
        accuracy_layout.setSpacing(20)  # Increased spacing between label and value box
        accuracy_layout.setContentsMargins(50, 0, 0, 0)  # Align with other metrics
    
        accuracy_label = QLabel("<b><i>Accuracy:</i></b>")
        accuracy_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        accuracy_value = QLabel(f"{acc*100:.2f} %")
        accuracy_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white;")
        accuracy_value.setFixedSize(80, 25)  
    
        accuracy_layout.addWidget(accuracy_label)
        accuracy_layout.addWidget(accuracy_value)
        accuracy_layout.addStretch()  
        main_grid.addLayout(accuracy_layout, 1, 0)
        
        # Put label and value in same cell with horizontal layout
        accuracy_layout = QHBoxLayout()
        accuracy_layout.addWidget(accuracy_label)
        accuracy_layout.addWidget(accuracy_value)
        accuracy_layout.setSpacing(5)
        accuracy_layout.setContentsMargins(50, 0, 0, 0)  
        main_grid.addLayout(accuracy_layout, 1, 0, 1, 2)

        # Confusion Matrix row
        confusion_matrix_label = QLabel("<b><i>Confusion matrix</i></b>")
        confusion_matrix_label.setAlignment(Qt.AlignLeft)
        confusion_matrix_label.setContentsMargins(50, 0, 0, 0)  
        main_grid.addWidget(confusion_matrix_label, 2, 0, 1, 2)

        if len(y_true) > 0 and len(y_pred) > 0:
            cm_canvas = self.plot_confusion_matrix(y_true, y_pred)
            cm_canvas.setContentsMargins(50, 0, 0, 0)  
            main_grid.addWidget(cm_canvas, 3, 0, 1, 2, alignment=Qt.AlignLeft)
        else:
            no_data_label = QLabel("Confusion matrix not available.")
            no_data_label.setContentsMargins(50, 0, 0, 0)
            main_grid.addWidget(no_data_label, 3, 0, 1, 2)

        # === Regression Section (moved left 50px)
        regression_label = QLabel("<b><i>Regression</b></i>")
        regression_label.setAlignment(Qt.AlignLeft)
        regression_label.setContentsMargins(-50, 0, 0, 0)  
        regression_label.setStyleSheet("font-size: 17px;")  
        main_grid.addWidget(regression_label, 4, 0, 1, 1)

        # MSE row
        mse_layout = QHBoxLayout()
        mse_layout.setSpacing(20)  # Increased spacing between label and value box
        mse_layout.setContentsMargins(50, 0, 0, 0)
    
        mse_label = QLabel("<b><i>MSE:</i></b>")
        mse_value = QLabel(f"{mse:.4f}")
        mse_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white;")
        mse_value.setFixedSize(80, 25)
    
        mse_layout.addWidget(mse_label)
        mse_layout.addWidget(mse_value)
        mse_layout.addStretch()
        main_grid.addLayout(mse_layout, 5, 0)

        # RMSE row
        rmse_layout = QHBoxLayout()
        rmse_layout.setSpacing(20)  # Increased spacing between label and value box
        rmse_layout.setContentsMargins(50, 0, 0, 0)
    
        rmse_label = QLabel("<b><i>RMSE:</i></b>")
        rmse_value = QLabel(f"{rmse:.4f}")
        rmse_value.setStyleSheet("border: 1px solid gray; padding: 2px; background-color: white;")
        rmse_value.setFixedSize(80, 25)
    
        rmse_layout.addWidget(rmse_label)
        rmse_layout.addWidget(rmse_value)
        rmse_layout.addStretch()
        main_grid.addLayout(rmse_layout, 6, 0)


        # Add spacer with 0 height
        main_grid.addItem(QSpacerItem(20, 0, QSizePolicy.Minimum, QSizePolicy.Expanding), 7, 0, 1, 2)

        # Add the grid layout to a horizontal box to center it
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(main_grid)
        hbox.addStretch(1)

        layout.addStretch(1)
        layout.addLayout(hbox)
        layout.addStretch(1)
        layout.setAlignment(Qt.AlignCenter)

        page.setLayout(layout)
        self.stacked_widget.addWidget(page)


    def add_compare_models_page(self):
        """Add the Compare Models page with dropdown model selection"""
        page = QWidget()
        layout = QVBoxLayout()
        comparison_layout = QHBoxLayout()
        
        # Initialize model manager if not exists
        if not hasattr(self, 'model_manager'):
            from utils.model_manager import ModelManager
            self.model_manager = ModelManager()

        # Add current model if exists
        if hasattr(self, 'test_results') and self.test_results:
            self.model_manager.add_current_model({
                'y_true': self.test_results.get('y_true', []),
                'y_pred': self.test_results.get('y_pred', []),
                'accuracy': self.test_results.get('accuracy', 0.0)
            })

        def create_model_section(side: str):
            section = QWidget()
            layout = QVBoxLayout()

            # Model selection button
            model_select_btn = QPushButton(f"Select Model {side}")
            model_select_btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid #dcdcdc;
                    border-radius: 5px;
                    padding: 5px 30px 5px 10px;
                    background-color: white;
                    font-size: 14px;
                    color: #333;
                    text-align: left;
                    min-width: 200px;
                }
                QPushButton:hover {
                    border: 1px solid #87ceeb;
                    background-color: #f8f8f8;
                }
                QPushButton::menu-indicator {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    image: url(assets/arrow_down.png);
                    min-width: 20px;
                    min-height: 20px;
                    border: none;
                    right: 5px;
                }
            """)

            # Create menu and store actions dictionary
            model_menu = QMenu()
            model_actions = {}
            
            def update_menus():
                """Update both menus with current models"""
                model_menu.clear()
                # Store load_action in the outer scope
                nonlocal load_action  
                load_action = model_menu.addAction("Load Model...")
                model_menu.addSeparator()
                
                model_names = self.model_manager.get_model_names()
                if not model_names:
                    no_models = model_menu.addAction("No models found")
                    no_models.setEnabled(False)
                else:
                    for name in model_names:
                        action = model_menu.addAction(name)
                        model_actions[action] = name

            def on_menu_triggered(action):
                if action.text() == "Load Model...":
                    file_path, _ = QFileDialog.getOpenFileName(
                        self, "Load Model", "", "H5 Files (*.h5)"
                    )
                    if file_path and os.path.exists(file_path):
                        if model_name := self.model_manager.load_model(file_path):
                            model_select_btn.setText(model_name)
                            update_display(model_name)
                            update_menus()  # Update both menus
                elif action.text() != "No models found":
                    model_name = action.text()
                    model_select_btn.setText(model_name)
                    update_display(model_name)

            # Connect menu trigger
            model_menu.triggered.connect(on_menu_triggered)
            model_select_btn.setMenu(model_menu)

            # Declare load_action before initial menu update
            load_action = None
            update_menus()

            # Plot canvas with frame
            canvas_container = QWidget()
            canvas_container.setStyleSheet("""
                QWidget {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background: white;
                }
            """)
            canvas_layout = QVBoxLayout(canvas_container)
            plot_canvas = FigureCanvas(Figure(figsize=(10, 6)))
            canvas_layout.addWidget(plot_canvas)

            # Info section
            info_widget = QWidget()
            info_layout = QVBoxLayout()
            info_widget.setStyleSheet("""
                QWidget {
                    background: #f5f5f5;
                    border-radius: 5px;
                    padding: 10px;
                }
                QLabel {
                    font-size: 13px;
                    padding: 5px;
                }
            """)

            accuracy_label = QLabel("Accuracy: N/A")
            inputs_label = QLabel("Used Data: N/A")
            params_button = QPushButton("View Hyperparameters")
            params_button.setEnabled(False)
            params_button.setStyleSheet("""
                QPushButton {
                    background-color: #e7f3ff;
                    border: 1px solid #a6c8ff;
                    border-radius: 5px;
                    padding: 8px 15px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #d0e7ff;
                }
                QPushButton:disabled {
                    background-color: #f0f0f0;
                    border-color: #ddd;
                    color: #999;
                }
            """)

            info_layout.addWidget(accuracy_label)
            info_layout.addWidget(inputs_label)
            info_layout.addWidget(params_button)
            info_widget.setLayout(info_layout)

            def update_display(model_name):
                if not model_name:
                    return
                    
                model_info = self.model_manager.get_model_info(model_name)
                if model_info:
                    # Update plot
                    plot_canvas.figure.clear()
                    ax = plot_canvas.figure.add_subplot(111)
                    
                    time = np.arange(len(model_info['results']['y_true']))
                    ax.plot(time, model_info['results']['y_true'], 'k-', label='Ground Truth')
                    ax.plot(time, model_info['results']['y_pred'], 'r--', label='Prediction')
                    
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Phase')
                    ax.set_title(f'Model {side} Predictions')
                    ax.grid(True)
                    ax.legend()
                    
                    plot_canvas.draw()

                    # Update info
                    accuracy_label.setText(f"Accuracy: {model_info['accuracy']:.2f}%")
                    inputs_label.setText(f"Used Data: {', '.join(model_info['inputs'])}")
                    params_button.setEnabled(True)
                    
                    def show_params():
                        dialog = QDialog(self)
                        dialog.setWindowTitle("Model Hyperparameters")
                        dialog.setMinimumWidth(400)
                        
                        layout = QVBoxLayout()
                        text = QTextBrowser()
                        text.setStyleSheet("font-family: monospace;")
                        
                        params_text = ["Model Configuration:"]
                        for key, value in model_info['hyperparameters'].items():
                            params_text.append(f"• {key}: {value}")
                            
                        text.setText("\n".join(params_text))
                        layout.addWidget(text)
                        
                        dialog.setLayout(layout)
                        dialog.exec_()
                        
                    params_button.clicked.connect(show_params)

            # Load model from combo selection
            model_select_btn.clicked.connect(update_display)
            
            # Add initial items if any
            model_select_btn.setText("Load Model...")
            model_select_btn.setEnabled(True)
            model_menu.addAction("No models found").setEnabled(False)

            def on_menu_triggered(action):
                if action == load_action:
                    file_path, _ = QFileDialog.getOpenFileName(
                        self, 
                        "Load Model",
                        "",
                        "H5 Files (*.h5)"
                    )
                    if file_path and os.path.exists(file_path):
                        if model_name := self.model_manager.load_model(file_path):
                            # Add new model to menu
                            new_action = model_menu.addAction(model_name)
                            model_actions[new_action] = model_name
                            # Update display
                            model_select_btn.setText(model_name)
                            update_display(model_name)
                else:
                    model_name = model_actions.get(action)
                    if model_name:
                        model_select_btn.setText(model_name)
                        update_display(model_name)

            model_menu.triggered.connect(on_menu_triggered)

            # Layout assembly
            layout.addWidget(model_select_btn)
            layout.addWidget(canvas_container)
            layout.addWidget(info_widget)
            layout.setContentsMargins(10, 10, 10, 10)
            section.setLayout(layout)
            
            return section, model_menu

        # Create sections and get menus
        left_section, left_menu = create_model_section("A")
        right_section, right_menu = create_model_section("B")
        
        comparison_layout.addWidget(left_section)
        comparison_layout.addWidget(right_section)
        
        layout.addLayout(comparison_layout)
        layout.setContentsMargins(20, 20, 20, 20)
        page.setLayout(layout)
        
        self.stacked_widget.addWidget(page)

    def add_empty_page(self):
        """
        Add an empty page to the stacked widget as the default page.
        """
        empty_page = QWidget()
        self.stacked_widget.addWidget(empty_page)

    def plot_prediction_and_curves(self, fig):
        
        """
        Plot prediction vs ground truth using step plots
        
        Args:
            fig: matplotlib figure object
        """
        fig.clear()
        ax = fig.add_subplot(111)  # Single plot

        # Get data and convert to numpy arrays 
        ground_truth = np.array(self.test_results.get("y_true", []), dtype=int) 
        predicted = np.array(self.test_results.get("y_pred", []), dtype=int)

        if len(ground_truth) > 0 and len(predicted) > 0:
            # Create time axis
            t = np.arange(len(ground_truth)) * 10  # Scale time by 10

            # Ground truth plot (black line)
            ax.plot(t, ground_truth, label='Ground Truth', color='black', linewidth=2)

            # Prediction plot (red line)
            ax.plot(t, predicted, label='Prediction', color='red', linewidth=1.5, alpha=0.7)

            # Set y-axis ticks and labels using phase_to_int mapping
            ax.set_yticks(list(int_to_phase.keys()))
            ax.set_yticklabels(list(int_to_phase.values()))
            
            ax.set_xlabel("Time (index)")
            ax.set_ylabel("Phase")
            ax.set_title("Prediction vs Ground Truth")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "Test results not available.\nTrain the model first.",
                    ha='center', va='center')
            ax.set_axis_off()

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

    def prediction_scatter_plot_canvas(self, fig):
        """Plot scatter of predictions vs ground truth on a given figure"""
        import numpy as np
        from collections import Counter
        
        # Get data
        y_true = np.array(progress_state.test_results.get("y_true", []))
        y_pred = np.array(progress_state.test_results.get("y_pred", []))
        
        ax = fig.add_subplot(111)

        if len(y_true) > 0 and len(y_pred) > 0:
            jitter = 0.2
            y_true_plot = y_true + np.random.normal(0, jitter, size=y_true.shape)
            y_pred_plot = y_pred + np.random.normal(0, jitter, size=y_pred.shape)
            sizes = 30
            point_color = "royalblue"

            # Scatter plot
            ax.scatter(y_true_plot, y_pred_plot, alpha=0.4, s=sizes, color="tab:blue")

            # Diagonal line
            ax.plot([-1, 7], [-1, 7], 'k--', linewidth=1)

            # Axis config
            ax.set_xlim(-1, 7)
            ax.set_ylim(-1, 7)
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))

            # Secondary axes for phase names
            ax2 = ax.twinx()
            ax3 = ax.twiny()
            ax2.set_ylim(ax.get_ylim())
            ax3.set_xlim(ax.get_xlim())
            ax2.set_yticks(range(7))
            ax3.set_xticks(range(7))
            ax2.set_yticklabels(list(int_to_phase.values()))
            ax3.set_xticklabels(list(int_to_phase.values()))

            ax.set_xlabel("Ground Truth Phase (integer)")
            ax.set_ylabel("Predicted Phase (integer)")
            ax.set_title("Predicted vs Ground Truth (Scatter)")
            ax.grid(True)

        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_axis_off()

        fig.tight_layout()

    def plot_confusion_matrix(self, y_true, y_pred, figsize=(6, 6)):  # Reduced matrix size
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import numpy as np

        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        cm = confusion_matrix(y_true, y_pred, labels=list(int_to_phase.keys()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(int_to_phase.values()))
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        ax.set_title("")  # Removed title
        ax.set_xticklabels(list(int_to_phase.values()), rotation=45, ha="right")

        # Adjust layout for centering and reduced margins
        fig.subplots_adjust(top=0.9, bottom=0.3, left=0.2, right=0.8)  # Adjusted margins to fit within container
        return FigureCanvas(fig)

    def plot_step_prediction_vs_ground_truth(self, fig, ground_truth=None, predicted=None):
        from functions import phase_to_int
        import matplotlib.ticker as ticker
        import numpy as np

        # Correction ici : forcer en array
        y_true = np.array(self.test_results.get("true_labels", []), dtype=int)
        y_pred = np.array(self.test_results.get("predictions", []), dtype=int)
        time = np.array(self.test_results.get("time", []))  # secondes

        if len(y_true) == 0 or len(y_pred) == 0 or len(time) == 0:
            print("❌ Données incomplètes pour le graphe temporel.")
            return

        # ✅ Tri par temps croissant
        sorted_indices = np.argsort(time)
        time = time[sorted_indices]
        y_true = y_true[sorted_indices]
        y_pred = y_pred[sorted_indices]

        fig.clear()
        ax = fig.add_subplot(111)

        ax.plot(time, y_true, label="Ground Truth", color="black", linewidth=2)
        ax.plot(time, y_pred, label="Prediction", color="red", linewidth=1.5, alpha=0.7)

        ax.set_yticks(ticks=list(phase_to_int.values()))
        ax.set_yticklabels(list(phase_to_int.keys()))

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Phase")
        ax.set_title("Prediction vs Ground Truth (Time-Aligned)")
        ax.legend()
        ax.grid(True)

        ax.set_xlim(time[0], time[-1])
        print(f"[INFO] Time range: {time[0]:.1f}ms → {time[-1]:.1f}ms")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        fig.tight_layout()
