import os
import h5py
import numpy as np
from scipy.signal import butter, filtfilt, medfilt  # Added medfilt
from datetime import datetime  # Added datetime
import json  # Added json

# === EMG Filters ===
def bandpass_filter(data, lowcut=10, highcut=40, fs=100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff=5, fs=100, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low')
    return filtfilt(b, a, data)

def process_emg_signal(emg, lowcut=10, highcut=40, fs=100, order=4):
    emg_bpf = bandpass_filter(emg, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    emg_rectified = np.abs(emg_bpf)
    return emg_rectified

def normalize_emg(data, method='zscore', mean_val=None, std_val=None, max_val=None):
    if method == 'zscore':
        mean_val = mean_val or np.mean(data)
        std_val = std_val or np.std(data)
        return (data - mean_val) / (std_val + 1e-8)
    elif method == 'max':
        max_val = max_val or np.max(data)
        return data / (max_val + 1e-8)
    else:
        raise ValueError("method must be 'zscore' or 'max'")

# === Label Generation ===
phase_to_int = {
    'Stand': 0,
    'Stand-to-Sit': 1,
    'Sit': 2,
    'Sit-to-Stand': 3,
    'Stand-to-Walk': 4,
    'Walk': 5,
    'Walk-to-Stand': 6
}

# Global constants
PHASE_SEQUENCE = [
    'Stand', 'Stand-to-Sit', 'Sit', 'Sit-to-Stand',
    'Stand', 'Stand-to-Walk', 'Walk', 'Walk-to-Stand', 'Stand'
]

def generate_fixed_sequence_labels(button_ok, fs=100):
    """
    Generate labels for fixed sequence phases based on button presses with debouncing logic.

    Args:
        button_ok (numpy array): Binary button signal.
        fs (int): Sampling frequency (default: 100Hz).

    Returns:
        numpy array: Phase labels for each time step.
    """
    pressed_indices = []
    last_press = -1000  # 1-second debounce (at 100Hz)

    # Detect presses with debouncing
    for idx in np.where(np.diff(button_ok.astype(int)) == 1)[0]:
        if idx - last_press > fs:  # 1-second gap between presses
            pressed_indices.append(idx)
            last_press = idx

    print(f"Detected {len(pressed_indices)} button presses at indices: {pressed_indices}")

    T = len(button_ok)
    labels = np.full(T, 'Stand', dtype=object)

    idx = 0
    for i, press_idx in enumerate(pressed_indices):
        # Duration phase
        n_delay = int(0.2 * fs)  # 200ms delay
        labels[idx:press_idx] = PHASE_SEQUENCE[i * 2]
        end_idx = min(press_idx + n_delay, T)
        labels[press_idx:end_idx] = PHASE_SEQUENCE[i * 2 + 1]
        idx = end_idx

    # Handle cases where button presses exceed the phase sequence length
    if 2 * len(pressed_indices) < len(PHASE_SEQUENCE):
        labels[idx:] = PHASE_SEQUENCE[2 * len(pressed_indices)]
    else:
        print(f"Warning: Too many button presses ({len(pressed_indices)}) for the fixed phase sequence.")
        labels[idx:] = 'Stand'  # Default phase

    return labels

def convert_labels_to_int(trial_data, mapping=None):
    """
    Convert string labels to integer labels using a mapping dictionary.
    """
    if mapping is None:
        mapping = phase_to_int

    label_str = trial_data['label']
    label_int = np.zeros_like(label_str, dtype=int)  # Default to 0 (Stand)

    for i, label in enumerate(label_str):
        if label in mapping:
            label_int[i] = mapping[label]
        else:
            print(f"Warning: Unknown label '{label}' found, defaulting to 'Stand'")
            label_int[i] = 0  # Default to Stand

    trial_data['label_int'] = label_int
    return trial_data

def process_trial(trial):
    """
    Process a single trial by generating labels and converting them to integers.
    """
    try:
        if 'button_ok' not in trial:
            raise ValueError("No button_ok data found in trial.")

        # Generate labels
        trial['label'] = generate_fixed_sequence_labels(trial['button_ok'])

        # Convert to integers
        trial = convert_labels_to_int(trial)

        return trial
    except Exception as e:
        print(f"Error processing trial {trial.get('trial', 'unknown')}: {str(e)}")
        return None  # or handle the error appropriately

# === Trial Processing ===
def process_all_emg(all_trial_data, fs=100, lp_cutoff=5, norm_method='zscore'):
    processed_trials = []
    emg_keys = ['emgL1', 'emgL2', 'emgL3', 'emgL4', 'emgR1', 'emgR2', 'emgR3', 'emgR4']
    imu_keys = ['imu1', 'imu2', 'imu3', 'imu4', 'imu5']

    for trial in all_trial_data:
        trial_copy = {}
        for key, value in trial.items():
            if key in ['file', 'trial']:
                trial_copy[key] = value
            elif key in emg_keys:
                emg_result = process_emg_signal(np.squeeze(value))
                for suffix, data in emg_result.items():
                    trial_copy[f"{key}_{suffix}"] = data
            elif key in imu_keys:
                imu_result = process_imu(np.squeeze(value))
                for suffix, data in imu_result.items():
                    trial_copy[f"{key}_{suffix}"] = data
            else:
                trial_copy[key] = np.squeeze(value)
        processed_trials.append(trial_copy)
    return processed_trials

class DataPreprocessor:
    def __init__(self, fs=100):
        self.fs = fs
        self.button_threshold = 0.5
        self.emg_bandpass = [10, 40]
        self.emg_lowpass = 5
        self.emg_order = 4
        self.imu_lowpass = 20

    def preprocess_button(self, raw_button):
        """Process raw button data to clean binary signal."""
        try:
            button_data = np.squeeze(raw_button).astype(np.float32)
            button_data = (button_data - np.min(button_data)) / (np.max(button_data) - np.min(button_data) + 1e-8)
            binary_signal = (button_data > self.button_threshold).astype(np.int8)
            return binary_signal
        except Exception as e:
            raise ValueError(f"Button processing failed: {str(e)}")

    def process_emg(self, raw_emg):
        """Full EMG processing pipeline."""
        try:
            emg_bpf = bandpass_filter(raw_emg, lowcut=self.emg_bandpass[0], highcut=self.emg_bandpass[1], fs=self.fs, order=self.emg_order)
            emg_rect = np.abs(emg_bpf)
            emg_env = lowpass_filter(emg_rect, cutoff=self.emg_lowpass, fs=self.fs, order=self.emg_order)
            return {
                'raw': raw_emg,
                'filtered': emg_bpf,
                'envelope': emg_env,
                'normalized': normalize_emg(emg_env)
            }
        except Exception as e:
            raise ValueError(f"EMG processing failed: {str(e)}")

    def process_imu(self, raw_imu):
        """IMU processing pipeline."""
        try:
            imu_filt = lowpass_filter(raw_imu, cutoff=self.imu_lowpass, fs=self.fs, order=self.emg_order)
            return {
                'raw': raw_imu,
                'filtered': imu_filt,
                'normalized': normalize_emg(imu_filt)
            }
        except Exception as e:
            raise ValueError(f"IMU processing failed: {str(e)}")

    def process_trial(self, trial_group):
        """
        Process a single trial from HDF5 group, handling both cropped and sensor file structures.

        Args:
            trial_group (h5py.Group): HDF5 group representing a trial.

        Returns:
            dict: Processed trial data or None if processing fails.
        """
        trial_data = {}
        try:
            # Extract button_ok data
            if 'button_ok' in trial_group:
                button_data = trial_group['button_ok'][:]
            elif 'Controller' in trial_group and 'button_ok' in trial_group['Controller']:
                button_data = trial_group['Controller']['button_ok'][:]
            else:
                raise KeyError("No button_ok data found in trial.")

            # Preprocess button_ok data
            trial_data['button_ok'] = self.preprocess_button(button_data)

            # Generate labels
            labels_str = generate_fixed_sequence_labels(trial_data['button_ok'])
            trial_data['label'] = labels_str
            trial_data['label_int'] = np.array([phase_to_int[label] for label in labels_str])

            # Extract sensor data
            sensor_data = trial_group.get('Sensor', trial_group)

            # Process EMG channels
            emg_keys = [k for k in sensor_data.keys() if k.startswith('emg')]
            for key in emg_keys:
                if key in sensor_data:
                    trial_data[key] = sensor_data[key][:]
                else:
                    print(f"Warning: EMG key '{key}' not found in trial.")

            # Process IMU channels
            imu_keys = [k for k in sensor_data.keys() if k.startswith('imu')]
            for key in imu_keys:
                if key in sensor_data:
                    trial_data[key] = sensor_data[key][:]
                else:
                    print(f"Warning: IMU key '{key}' not found in trial.")

            # Extract time data
            if 'time' in trial_group:
                trial_data['time'] = trial_group['time'][:]
            elif 'Time' in trial_group:
                trial_data['time'] = trial_group['Time'][:]
            else:
                print("Warning: Time data not found in trial.")

            return trial_data
        except KeyError as e:
            print(f"Error processing trial: {str(e)}")
            return None
        except IndexError as e:
            print(f"Error processing trial: Index out of range - {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error processing trial: {str(e)}")
            return None

    def generate_labels(self, button_signal, fs, k_ms=[200, 200, 200, 200]):
        """Generate activity labels from button presses."""
        phase_seq = [
            'Stand', 'Stand-to-Sit', 'Sit', 'Sit-to-Stand',
            'Stand', 'Stand-to-Walk', 'Walk', 'Walk-to-Stand', 'Stand'
        ]
        
        pressed_indices = np.where(np.diff(button_signal.astype(int)) == 1)[0] + 1
        
        if len(pressed_indices) < 4:
            print(f"Warning: Expected 4 button presses, got {len(pressed_indices)}. Filling with default phase.")
            labels = np.full(len(button_signal), 'Stand', dtype=object)
            return labels
        
        labels = np.empty(len(button_signal), dtype=object)
        idx = 0
        
        for i, press_idx in enumerate(pressed_indices):
            n_delay = int((k_ms[i] / 1000.0) * fs)
            labels[idx:press_idx] = phase_seq[i * 2]
            end_idx = min(press_idx + n_delay, len(button_signal))
            labels[press_idx:end_idx] = phase_seq[i * 2 + 1]
            idx = end_idx
        
        labels[idx:] = phase_seq[2 * len(pressed_indices)]
        return labels

def preprocess_button_data(button_data, threshold=0.5):
    """
    Preprocess raw button data to clean binary signal.

    Args:
        button_data (numpy array): Raw button signal.
        threshold (float): Threshold for binary conversion.

    Returns:
        numpy array: Clean binary button signal (0 or 1).
    """
    button_data = button_data.astype(np.float32)
    button_data = (button_data - np.min(button_data)) / (np.max(button_data) - np.min(button_data) + 1e-8)
    binary_signal = (button_data > threshold).astype(np.int8)
    return binary_signal

def process_trial(trial_group, file_type="cropped"):
    """
    Process a single trial by generating labels and converting them to integers.
    Handles both 'cropped' and 'sensor' file structures.

    Args:
        trial_group (h5py.Group): HDF5 group representing a trial.
        file_type (str): Type of file structure ('cropped' or 'sensor').

    Returns:
        dict: Processed trial data or None if processing fails.
    """
    trial_data = {}
    try:
        # Extract button_ok data
        if file_type == "cropped":
            if 'button_ok' in trial_group:
                button_data = trial_group['button_ok'][:]
            else:
                raise KeyError("No button_ok data found in cropped trial.")
        elif file_type == "sensor":
            if 'Controller' in trial_group and 'button_ok' in trial_group['Controller']:
                button_data = trial_group['Controller']['button_ok'][:]
            else:
                raise KeyError("No button_ok data found in sensor trial.")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Preprocess button_ok data
        trial_data['button_ok'] = preprocess_button_data(button_data)

        # Generate labels
        labels_str = generate_fixed_sequence_labels(trial_data['button_ok'])
        trial_data['label'] = labels_str
        trial_data['label_int'] = np.array([phase_to_int[label] for label in labels_str])

        # Extract sensor data
        if file_type == "cropped":
            sensor_data = trial_group
        elif file_type == "sensor":
            sensor_data = trial_group.get('Sensor', trial_group)

        # Process EMG channels
        emg_keys = [k for k in sensor_data.keys() if k.startswith('emg')]
        for key in emg_keys:
            trial_data[key] = sensor_data[key][:]

        # Process IMU channels
        imu_keys = [k for k in sensor_data.keys() if k.startswith('imu')]
        for key in imu_keys:
            trial_data[key] = sensor_data[key][:]

        # Extract time data
        if file_type == "cropped":
            if 'time' in trial_group:
                trial_data['time'] = trial_group['time'][:]
            elif 'Time' in trial_group:
                trial_data['time'] = trial_group['Time'][:]
        elif file_type == "sensor":
            if 'Time' in trial_group:
                trial_data['time'] = trial_group['Time'][:]

        return trial_data
    except KeyError as e:
        print(f"Error processing trial: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error processing trial: {str(e)}")
        return None

def inspect_hdf5_file(filepath):
    """
    Inspect the structure of an HDF5 file and check for required keys.

    Args:
        filepath (str): Path to the HDF5 file.

    Returns:
        dict: Summary of the file structure and any missing keys.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"Root keys in {filepath}: {list(f.keys())}")
            
            if 'trial' not in f:
                print(f"Missing 'trial' in {filepath}")
                return {"status": "error", "message": "Missing 'trial' key"}
            
            trial = f['trial']
            if 'button_ok' not in trial or len(trial['button_ok']) == 0:
                print("No button_ok data found in trial.")
                return {"status": "error", "message": "Missing or empty 'button_ok' data"}
            
            # Additional checks can be added here...
            return {"status": "success", "message": "File structure is valid"}
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return {"status": "error", "message": str(e)}

import h5py

def read_cropped_trials(filepath):
    with h5py.File(filepath, 'r') as f:
        trial_keys = [key for key in f.keys() if key.startswith('trial_')]
        print(f"Found trial keys: {trial_keys}")
        
        all_trials = {}
        for trial_key in trial_keys:
            trial_data = {}
            for sensor_key in f[trial_key].keys():
                trial_data[sensor_key] = f[trial_key][sensor_key][:]
            all_trials[trial_key] = trial_data
        return all_trials
def read_sensor_file(filepath):
    with h5py.File(filepath, 'r') as f:
        sensor_data = {}

        # Read EMG
        emg_group = f['Sensor']['EMG']
        for ch in emg_group.keys():
            sensor_data[ch] = emg_group[ch][:]

        # Read IMU
        imu_group = f['Sensor']['IMU']
        for ch in imu_group.keys():
            sensor_data[ch] = imu_group[ch][:]

        # Read Controller
        sensor_data['button_ok'] = f['Controller']['button_ok'][:]

        # Read Time
        sensor_data['time'] = f['Sensor']['Time']['time'][:]

        return sensor_data

def process_imu(raw_imu, lowcut=5, fs=100, order=4):
    """
    Process IMU data using a lowpass filter and normalization.

    Args:
        raw_imu (numpy array): Raw IMU data.
        lowcut (int): Lowpass filter cutoff frequency.
        fs (int): Sampling frequency.
        order (int): Filter order.

    Returns:
        dict: Processed IMU data including raw, filtered, and normalized signals.
    """
    try:
        imu_filt = lowpass_filter(raw_imu, cutoff=lowcut, fs=fs, order=order)
        imu_norm = normalize_emg(imu_filt)
        return {
            'raw': raw_imu,
            'filtered': imu_filt,
            'normalized': imu_norm
        }
    except Exception as e:
        raise ValueError(f"IMU processing failed: {str(e)}")

# Verify the HDF5 file structure
file_path = "data10/sensor_20250520_214656.h5"
result = read_cropped_trials(file_path)
result2 = read_sensor_file(file_path)
print(result)
print(result2)

