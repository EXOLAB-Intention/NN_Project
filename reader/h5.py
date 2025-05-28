import h5py

with h5py.File(r"C:\Users\harry\OneDrive - UPEC\Documents\BUT2\Stage\KAIST\NN Package\NN-Package\datasets\dataset1\filtered_cropped_20250520_214656.h5", "r") as f:
    print("Top-level keys:", list(f.keys()))
    trial = f.get("trial_10")
    if trial:
        print("trial_1 keys:", list(trial.keys()))
    else:
        print("trial_1 not found")
