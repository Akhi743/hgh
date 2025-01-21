import os

def check_files():
    print("Current directory:", os.getcwd())
    print("\nListing all files and directories in current folder:")
    for item in os.listdir():
        print(item)
    
    dataset_path = os.path.join(os.getcwd(), "Dataset")
    if os.path.exists(dataset_path):
        print("\nListing contents of Dataset folder:")
        for item in os.listdir(dataset_path):
            print(item)
    else:
        print("\nDataset folder not found!")

check_files()