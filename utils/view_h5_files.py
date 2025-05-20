import os
import h5py

def view_h5_files(directory):
    """
    List and display the content and raw data of .h5 files in the specified directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            filepath = os.path.join(directory, filename)
            print(f"Opening file: {filepath}")
            with h5py.File(filepath, 'r') as h5_file:
                def print_data(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"Dataset: {name}")
                        print(f"Data: {obj[()]}")
                    elif isinstance(obj, h5py.Group):
                        print(f"Group: {name}")
                h5_file.visititems(print_data)

if __name__ == "__main__":
    sensor_data_dir = "sensor_data"  
    if os.path.exists(sensor_data_dir):
        view_h5_files(sensor_data_dir)
    else:
        print(f"Directory '{sensor_data_dir}' does not exist.")
