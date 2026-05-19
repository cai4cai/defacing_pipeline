import os
import argparse
import subprocess
import pandas as pd

def reorient_nifti(main_folder):
    """Traverse subfolders and reorient all NIfTI images to NMI152 standard"""

    # Lists to store processed files and their reorientation status
    processed_files = []
    status = []

    # Set FSL environment variables (if needed)
    # os.environ["FSLDIR"] = "/home/lorenagarcia-foncillasmacias/fsl"  # Set FSL installation directory
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"  # Set output file format to NIFTI_GZ

    # Print the starting folder where reorientation will take place
    print(f"Starting reorientation in: {main_folder}")

    # Check if the FSL command is available (do this once to avoid repeated checks)
    fsl_path = subprocess.run(["which", "fslreorient2std"], capture_output=True, text=True)
    if fsl_path.returncode != 0:
        # If the command is not found, notify the user and exit the function
        print("FSL not found in PATH. Ensure it is installed and accessible.")
        return

    # Traverse the subfolders of the provided directory (main_folder)
    for root, dirs, files in os.walk(main_folder):
        print(f"Traversing: {root}")  # Output the current folder being traversed

        # Iterate through each file in the current folder
        for file in files:
            if file.endswith(".nii.gz"):  # Process only NIfTI files (with .nii.gz extension)
                print(f"Found NIfTI: {file}")

                # Construct the full path to the input file
                input_file = os.path.join(root, file)

                # Command to run the fslreorient2std tool
                command = ["fslreorient2std", file, f"reoriented_{file}"]

                try:
                    # Run the command to reorient the file
                    result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=root)
                    print(f"Output: {result.stdout}")
                    print(f"stderr: {result.stderr}")

                except subprocess.CalledProcessError as e:
                    # If there is an error during processing, catch and print the error
                    print(f"Error processing {file}: {e}")
                    print(e.stderr)
                
                # Check if the output file was created (i.e., if the reorientation was successful)
                if os.path.exists(input_file.replace(file, "reoriented_"+file)):
                    processed_files.append(input_file)  # Add file to processed list
                    status.append(True)  # Mark as successfully processed
                    print(f"Successfully reoriented: {file}")
                else:
                    processed_files.append(input_file)  # Add file to processed list
                    status.append(False)  # Mark as failed to process

    # Create a log of the processing results and save it to a CSV file
    data = {
        "file": processed_files,  # File names
        "reorientation_success": status  # Status of reorientation (True/False)
    }
    df = pd.DataFrame(data=data)  # Convert the data to a pandas DataFrame
    log_file = os.path.join(main_folder, "reorientation_log.csv")  # Log file path
    df.to_csv(log_file, index=False)  # Save the DataFrame as a CSV
    print(f"Reorientation log saved at: {log_file}")

# Main function to handle command-line arguments and call the reorientation function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorient NIfTI images in subfolders.")
    parser.add_argument("main_folder", help="Path to the main folder containing NIfTI files.")  # Folder argument
    
    args = parser.parse_args()  # Parse the command-line arguments
    
    # Check if the provided path exists before proceeding
    if os.path.exists(args.main_folder):
        reorient_nifti(args.main_folder)  # Call the reorientation function
    else:
        print(f"Path does not exist: {args.main_folder}")  # Print an error if path doesn't exist

"""
Usage example to execute:
    python reorientation.py "./data/example_input_images"
"""