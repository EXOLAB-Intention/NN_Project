import h5py
import os
import numpy as np
import json

class ModelManager:
    def __init__(self):
        self.models = {}
        self.unnamed_count = 1

    def load_model(self, filepath):
        """Load model from H5 file with standardized parameter extraction"""
        try:
            with h5py.File(filepath, 'r') as f:
                print(f"[DEBUG] Loading model from {filepath}")
                model_name = os.path.splitext(os.path.basename(filepath))[0]
                
                # Extract training parameters
                training_params_data = f['training_params'][()]
                if isinstance(training_params_data, bytes):
                    training_params = json.loads(training_params_data.decode('utf-8'))
                    print(f"[DEBUG] Training params: {training_params}")

                # Get selected inputs from training params
                selected_inputs = training_params.get('selected_inputs', [])
                print(f"[DEBUG] Selected inputs: {selected_inputs}")
                
                # Debug accuracy data
                print("\n[DEBUG] Available keys in H5 file:", list(f.keys()))
                print("[DEBUG] Available keys in test_results:", list(f['test_results'].keys()))
                
                # Get accuracy - try multiple possible locations
                accuracy = None
                if 'test_results/accuracy' in f:
                    accuracy_data = f['test_results/accuracy'][()]
                    print(f"[DEBUG] Found accuracy in test_results/accuracy: {accuracy_data}")
                    accuracy = float(accuracy_data)
                elif 'test_results/accuracy_json' in f:
                    accuracy_data = f['test_results/accuracy_json'][()]
                    print(f"[DEBUG] Found accuracy in test_results/accuracy_json: {accuracy_data}")
                    if isinstance(accuracy_data, bytes):
                        accuracy = float(accuracy_data.decode('utf-8'))
                    else:
                        accuracy = float(accuracy_data)
                elif 'accuracy' in f:
                    accuracy_data = f['accuracy'][()]
                    print(f"[DEBUG] Found accuracy in root accuracy: {accuracy_data}")
                    accuracy = float(accuracy_data)
                
                print(f"[DEBUG] Final accuracy value: {accuracy}")
                
                # Extract model info
                model_info = {
                    'filepath': filepath,
                    'accuracy': accuracy * 100 if accuracy is not None else 0.0,
                    'results': {
                        'y_true': np.array(f['test_results/y_true'][:]),
                        'y_pred': np.array(f['test_results/y_pred'][:]),
                        'accuracy': accuracy * 100 if accuracy is not None else 0.0,
                        'time': np.array(f['test_results/time'][:]) if 'test_results/time' in f else None,
                    },
                    'inputs': selected_inputs,  # Use selected_inputs instead of checked_files
                    'hyperparameters': {
                        'optimizer': training_params.get('optimizer', 'Adam'),
                        'loss_function': training_params.get('loss_function', 'CrossEntropyLoss'),
                        'layers': training_params.get('hyperparameters', {}).get('layers', []),
                        'sequence_length': training_params.get('hyperparameters', {}).get('sequence_length', '50'),
                        'stride': training_params.get('hyperparameters', {}).get('stride', '5'),
                        'batch_size': training_params.get('hyperparameters', {}).get('batch_size', '32'),
                        'epoch_number': training_params.get('hyperparameters', {}).get('epoch_number', '10'),
                        'learning_rate': training_params.get('hyperparameters', {}).get('learning_rate', '0.001')
                    }
                }

                print(f"[DEBUG] Model info created: {model_info}")
                self.models[model_name] = model_info
                return model_name
                
        except Exception as e:
            print(f"[ERROR] Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def add_current_model(self, test_results):
        """Add current model to manager with standardized parameters"""
        model_name = f"Model {self.unnamed_count}"
        self.unnamed_count += 1
        
        # Create standardized model info with default parameters
        model_info = {
            'filepath': None,
            'results': test_results,
            'accuracy': test_results.get('accuracy', 0.0),
            'inputs': [],
            'hyperparameters': {
                'optimizer': 'Adam',
                'loss_function': 'CrossEntropyLoss',
                'layers': [],
                'sequence_length': '50',
                'stride': '5',
                'batch_size': '32',
                'epoch_number': '10',
                'learning_rate': '0.001'
            }
        }

        # Add display parameters in the same format as load_model
        model_info['display_params'] = {
            'optimizer': model_info['hyperparameters']['optimizer'],
            'loss_function': model_info['hyperparameters']['loss_function'],
            'layer_types': [layer.get('type', 'Unknown') for layer in model_info['hyperparameters']['layers']],
            'sequence_length': model_info['hyperparameters']['sequence_length'],
            'stride': model_info['hyperparameters']['stride'],
            'batch_size': model_info['hyperparameters']['batch_size'],
            'epochs': model_info['hyperparameters']['epoch_number'],
            'learning_rate': model_info['hyperparameters']['learning_rate']
        }
        
        self.models[model_name] = model_info
        return model_name

    def get_model_info(self, model_name):
        """Get model information"""
        return self.models.get(model_name)

    def get_model_names(self):
        """Get list of model names"""
        return list(self.models.keys())