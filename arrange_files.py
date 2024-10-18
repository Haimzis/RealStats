import os
import shutil
import glob

# Define the source directory where the files are located
source_dir = '/home/dsi/haimzis/projects/wavelets_detector/'

# Define the destination base directory
base_destination_dir = '/home/dsi/haimzis/projects/wavelets_detector/experiments/histograms/SDXL/'

# Use glob to find all files matching the pattern *_wave_{wavelet}.png
file_pattern = os.path.join(source_dir, '*_wave_*.png')
file_list = glob.glob(file_pattern)  # This will return the list of all matching files

# Function to extract the wavelet from the filename
def extract_wavelet(filename):
    # Splitting by '_wave_' and '.png' to get the wavelet part
    base_name = os.path.basename(filename)  # Get the file name from the full path
    return base_name.split('_wave_')[1].replace('.png', '')

# Move the files to their respective directories
for file_path in file_list:
    file_name = os.path.basename(file_path)  # Get just the file name
    wavelet = extract_wavelet(file_name)  # Extract the wavelet name from the file
    destination_dir = os.path.join(base_destination_dir, wavelet)  # Create the destination path
    
    # Create the directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Define the full destination path
    destination_path = os.path.join(destination_dir, file_name)
    
    # Move the file
    if os.path.exists(file_path):
        shutil.move(file_path, destination_path)
        print(f"Moved {file_name} to {destination_path}")
    else:
        print(f"File {file_name} does not exist in {source_dir}")

print("File moving completed.")
