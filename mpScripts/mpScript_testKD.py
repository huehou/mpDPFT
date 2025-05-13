import matplotlib.pyplot as plt
import os
import shutil  # For file copying
import pandas as pd  # For reading the data file
import subprocess  # For opening Okular
import glob  # For matching files with patterns
import subprocess  # For running external commands

# Base paths
source_directory = "/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/run/"
destination_directory = "/home/martintrappe/Desktop/PostDoc/Code/mpDPFT/#DATA/KD/testKD/"  # Using data subfolder

# Create the subfolder in destination if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Patterns to match the files (e.g., testK1_xxx.dat, testK2_xxx.dat, testK3_xxx.dat)
file_patterns = ["mpDPFT_testK1_*.dat", "mpDPFT_testK2_*.dat", "mpDPFT_testK3_*.dat"]

# List to store PDF paths for later opening in Okular
pdf_paths = []

# Loop through each file pattern (K1, K2, K3)
for pattern in file_patterns:
    # Loop through each matching file in the source directory for each pattern
    for source_file in glob.glob(os.path.join(source_directory, pattern)):
        file_name = os.path.basename(source_file)  # Get the file name with suffix (e.g., testK1_xxx.dat)
        destination_file = os.path.join(destination_directory, file_name)

        # Copy the file to the 'data' subfolder in the destination directory
        try:
            shutil.copy(source_file, destination_file)
            print(f"File copied from '{source_file}' to '{destination_file}'.")
        except Exception as e:
            print(f"Error copying the file '{file_name}': {e}")
            continue

        # Read the data file
        try:
            # Use delim_whitespace=True for space-separated values
            data = pd.read_csv(destination_file, delim_whitespace=True, header=None)
            data.columns = ['x', 'y']  # Assign column names
        except Exception as e:
            print(f"Error reading the data file '{file_name}': {e}")
            continue

        # Extract x and y values
        x = data['x']
        y = data['y']

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, label=f"y = f(x) for {file_name}", color="blue", linewidth=2)
        ax.set_xlabel("param")
        ax.set_ylabel(file_name.split('_')[1].replace('.dat', ''))  # Extract label from filename (e.g., K1 from testK1_xxx.dat)
        ax.legend()
        ax.grid(True)

        # Save the plot as a PDF in the 'data' subfolder without the timestamp
        pdf_path = os.path.join(destination_directory, f"{file_name.replace('.dat', '')}.pdf")
        fig.savefig(pdf_path)
        print(f"Plot for '{file_name}' saved as '{pdf_path}'.")

        # Add the PDF path to the list
        pdf_paths.append(pdf_path)

# Open all the saved PDFs in Okular in the same window
if pdf_paths:
    try:
        # Use subprocess.Popen to run Okular in the background
        subprocess.Popen(["okular", "--unique"] + pdf_paths)
        print(f"Opened all PDFs in the same Okular window.")
    except Exception as e:
        print(f"Error opening PDFs in Okular: {e}")
