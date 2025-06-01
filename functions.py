import os
import glob
import h5py
from scipy.signal import butter, filtfilt
import numpy as np

def load_trials_with_raw_data(file_path):
    trial_data_list = []

    with h5py.File(file_path, 'r') as f:
        for trial_name in f.keys():
            trial = f[trial_name]

            # 개별 센서별로 모두 불러오기
            trial_data = {
                'name': trial_name,
                'emgL1': trial['emgL1'][:],
                'emgL2': trial['emgL2'][:],
                'emgL3': trial['emgL3'][:],
                'emgL4': trial['emgL4'][:],
                'emgR1': trial['emgR1'][:],
                'emgR2': trial['emgR2'][:],
                'emgR3': trial['emgR3'][:],
                'emgR4': trial['emgR4'][:],
                'imu1':  trial['imu1'][:],
                'imu2':  trial['imu2'][:],
                'imu3':  trial['imu3'][:],
                'imu4':  trial['imu4'][:],
                'imu5':  trial['imu5'][:],
                'button_ok': trial['button_ok'][:],
                'time': trial['time'][:]
            }

            trial_data_list.append(trial_data)

    return trial_data_list

def load_all_trials_from_all_files(folder_path, pattern='cropped_*.h5'):
    all_trials = []

    file_list = sorted(glob.glob(os.path.join(folder_path, pattern)))
    print(f"총 {len(file_list)}개 파일을 로딩합니다.")

    for file_path in file_list:
        with h5py.File(file_path, 'r') as f:
            for trial_name in f.keys():
                trial = f[trial_name]

                trial_data = {
                    'file': os.path.basename(file_path),
                    'trial': trial_name,
                    'emgL1': trial['emgL1'][:],
                    'emgL2': trial['emgL2'][:],
                    'emgL3': trial['emgL3'][:],
                    'emgL4': trial['emgL4'][:],
                    'emgR1': trial['emgR1'][:],
                    'emgR2': trial['emgR2'][:],
                    'emgR3': trial['emgR3'][:],
                    'emgR4': trial['emgR4'][:],
                    'imu1':  trial['imu1'][:],
                    'imu2':  trial['imu2'][:],
                    'imu3':  trial['imu3'][:],
                    'imu4':  trial['imu4'][:],
                    'imu5':  trial['imu5'][:],
                    'button_ok': trial['button_ok'][:],
                    'time': trial['time'][:]
                }

                all_trials.append(trial_data)

    print(f"총 {len(all_trials)}개의 trial이 로딩되었습니다.")
    return all_trials

############################################################################################################################################
############################################################################################################################################
####################################################  Label  #########################################################################
############################################################################################################################################
############################################################################################################################################

def generate_fixed_sequence_labels(trial_data, fs=100, k_ms=[200, 200, 200, 200]):
    """
    trial_data: dict with 'button_ok', 'time'
    fs: sampling frequency (Hz)
    k_ms: transition phase duration after OK button (in ms)
    """
    button_ok = np.squeeze(trial_data['button_ok'])
    T = len(button_ok)

    # OK 버튼 눌림 시점 (rising edge)
    pressed_indices = np.where(np.diff(button_ok.astype(int)) == 1)[0] + 1

    if len(pressed_indices) != 4:
        raise ValueError(f"Expected 4 OK button presses, but found {len(pressed_indices)}")

    # 실험 고정 순서
    phase_seq = [
        'Stand',
        'Stand-to-Sit',
        'Sit',
        'Sit-to-Stand',
        'Stand',
        'Stand-to-Walk',
        'Walk',
        'Walk-to-Stand',
        'Stand'
    ]

    labels = np.empty(T, dtype=object)

    idx = 0
    for i, press_idx in enumerate(pressed_indices):
        # 지속 phase



        n_delay = int((k_ms[i] / 1000.0) * fs)


        labels[idx:press_idx] = phase_seq[i * 2]
        # transition phase
        end_idx = min(press_idx + n_delay, T)
        labels[press_idx:end_idx] = phase_seq[i * 2 + 1]
        idx = end_idx

    # 마지막 지속 phase
    labels[idx:] = phase_seq[2 * len(pressed_indices)]

    return labels

phase_to_int = {
    'Stand': 0,
    'Stand-to-Sit': 1,
    'Sit': 2,
    'Sit-to-Stand': 3,
    'Stand-to-Walk': 4,
    'Walk': 5,
    'Walk-to-Stand': 6
}

def convert_labels_to_int(trial_data, mapping):
    label_str = trial_data['label']
    label_int = np.array([mapping[label] for label in label_str])
    trial_data['label_int'] = label_int
    return trial_data


############################################################################################################################################
############################################################################################################################################
####################################################  EMG Filters  #########################################################################
############################################################################################################################################
############################################################################################################################################
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
    emg_bpf = bandpass_filter(emg, lowcut = lowcut, highcut = highcut, fs=fs, order = order)
    emg_rectified = np.abs(emg_bpf)
    return emg_rectified
def normalize_emg(data, method='zscore', mean_val=None, std_val=None, max_val=None):
    if method == 'zscore':
        if mean_val is None:
            mean_val = np.mean(data)
        if std_val is None:
            std_val = np.std(data)
        return (data - mean_val) / (std_val + 1e-8)
    elif method == 'max':
        if max_val is None:
            max_val = np.max(data)
        return data / (max_val + 1e-8)
    else:
        raise ValueError("method must be 'zscore' or 'max'")
    
def apply_filtered_emgL1(all_trial_data, lowcut=10, highcut=40, fs=100):
    all_trial_data_emgL1_fixed = []

    for trial in all_trial_data:
        trial_copy = trial.copy()  # 원본 유지

        emgL1 = fl(trial['emgL1'])
        filtered_emg = process_emg_signal(emgL1, lowcut = lowcut, highcut = highcut, fs=fs)

        trial_copy['emgL1'] = filtered_emg  # 필터링된 값으로 교체

        all_trial_data_emgL1_fixed.append(trial_copy)

    return all_trial_data_emgL1_fixed

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
                raw = np.squeeze(value)
                envelope = lowpass_filter(raw, cutoff=lp_cutoff, fs=fs)
                norm = normalize_emg(envelope, method=norm_method)

                trial_copy[key] = raw  # raw EMG 저장 (예: 'emgL1')
                trial_copy[f"{key}_filt"] = envelope  # envelope 저장 (예: 'emgL1_filt')
                trial_copy[f"{key}_norm"] = norm      # normalized 저장 (예: 'emgL1_norm')
            elif key in imu_keys:
                trial_copy[key] = value  # 그대로
            else:
                trial_copy[key] = np.squeeze(value)  # imu, button_ok, time 등

        processed_trials.append(trial_copy)

    return processed_trials



def fl(data): return np.squeeze(data[:])


def build_lstm_dataset_custom_keys(
    all_trial_data,
    input_keys,
    trial_names_to_use=None,
    window_size=200,
    stride=20,
    pred = 0
):
    """
    Build an LSTM dataset using custom input keys.
    
    Parameters:
    - all_trial_data: list of trial dicts
    - input_keys: list of strings (e.g., ['emgL1_norm', 'imu1', 'imu2'])
    - trial_names_to_use: optional list of (file, trial) keys to include
    - window_size: length of time window
    - stride: step size between windows
    
    Returns:
    - X: (N, window_size, num_features)
    - y: (N,)
    """
    X_list = []
    y_list = []

    for trial in all_trial_data:
        if trial_names_to_use is not None:
            if (trial['file'], trial['trial']) not in trial_names_to_use:
                continue

        inputs = []
        for k in input_keys:
            arr = np.squeeze(trial[k])  # (T,) or (4, T) → (T,), (T, 4)
            
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]  # (T, 1)
            elif arr.ndim == 2:
                if arr.shape[0] < arr.shape[1]:  # (4, T) → transpose
                    arr = arr.T  # → (T, 4)
            else:
                raise ValueError(f"Unsupported shape {arr.shape} for key '{k}'")

            inputs.append(arr)

        input_stack = np.concatenate(inputs, axis=-1)  # (T, D)
        labels = trial['label_int']
        T = len(labels)

        for start in range(0, T - window_size - max(pred, 0), stride):
            x_win = input_stack[start:start + window_size]  # (window_size, D)
            y_label = labels[start + window_size + pred]      # 중앙 프레임 라벨
            X_list.append(x_win)
            y_list.append(y_label)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def build_dataset_from_trial_keys(trial_dict, trial_keys, input_keys, window_size=200, stride=20, pred=0):
    selected_trials = [trial_dict[k] for k in trial_keys]
    X, y = build_lstm_dataset_custom_keys(
        all_trial_data=selected_trials,
        input_keys=input_keys,
        window_size=window_size,
        stride=stride,
        pred=pred
    )
    return X, y

def split_trials_train_val_test(trial_keys, test_ratio=0.2, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    trial_keys = np.random.permutation(trial_keys).tolist()

    n_total = len(trial_keys)
    print(f"Total trial number: {n_total}")
    n_test = int(n_total * test_ratio)
    n_val = int((n_total - n_test) * val_ratio)

    test_keys = trial_keys[:n_test]
    val_keys = trial_keys[n_test:n_test + n_val]
    train_keys = trial_keys[n_test + n_val:]

    return train_keys, val_keys, test_keys
