import os
import re
import argparse
import pandas as pd

def add_column_to_excel():
    """
    Function 1: Add a third column to Excel files.
    For each Excel file ("src/conf/info-ch.xlsx" and "src/conf/info-en.xlsx"), this function reads
    the first two columns and generates a third column by concatenating the first column, "__", 
    and the second column. Any space in the result is replaced with a dash ("-").
    """
    # List of Excel files to process
    file_list = ["src/conf/info-ch.xlsx", "src/conf/info-en.xlsx"]
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            continue
        
        try:
            # Assuming the Excel file does not have a header, so header=None
            df = pd.read_excel(file_path, header=None)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Add new column: concatenate first and second columns with "__" separator,
        # then replace all spaces with "-"
        df[2] = df.apply(lambda row: f"【{str(row[0])}】{str(row[1])}", axis=1)
        
        try:
            # Save without index and header (since the original file has no header)
            df.to_excel(file_path, index=False, header=False)
            print(f"Successfully updated file: {file_path}")
        except Exception as e:
            print(f"Error saving {file_path}: {e}")

def rename_images():
    """
    Function 2: Rename image files.
    For the directory "images_chinese" and all its subdirectories:
      - Use the third column of the Excel file "src/conf/info-ch.xlsx" to rename image files.
      - Files starting with "幻灯片n" (n starting from 1) will be renamed where the new 
        prefix is the nth entry of the Excel file's third column, while preserving the original
        file extension.
    Similarly, for the "images_english" directory, use "src/conf/info-en.xlsx" for renaming.
    """
    # Mapping for Chinese and English directories and Excel files
    config = [
        {
            "img_dir": "images_chinese",
            "excel_file": "src/conf/info-ch.xlsx",
            "desc": "Chinese"
        },
        {
            "img_dir": "images_english",
            "excel_file": "src/conf/info-en.xlsx",
            "desc": "English"
        }
    ]
    
    for item in config:
        excel_path = item["excel_file"]
        img_dir = item["img_dir"]
        desc = item["desc"]
        
        if not os.path.exists(excel_path):
            print(f"Excel file does not exist: {excel_path}")
            continue
        
        try:
            # Again assuming the Excel file has no header
            df = pd.read_excel(excel_path, header=None)
            # Get the third column as the list of new file names (first row corresponds to index 0)
            new_names = df.iloc[:, 2].tolist()
        except Exception as e:
            print(f"Error reading {excel_path}: {e}")
            continue
        
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
            continue
        
        # Traverse the img_dir and all its subdirectories
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                # Only process files with png or svg extensions (case insensitive)
                if not file.lower().endswith(('.png', '.svg', '.PNG', '.SVG')):
                    continue
                
                # Use regex to match "幻灯片" followed by a number; the number indicates the corresponding Excel row
                match = re.match(r"^(幻灯片)(\d+)", file)
                if match:
                    try:
                        n = int(match.group(2))
                    except ValueError:
                        print(f"Cannot parse number from: {file}")
                        continue
                    # Check if n is within the range of new_names list
                    if 1 <= n <= len(new_names):
                        new_prefix = str(new_names[n-1])
                        # Preserve the original file extension
                        ext = os.path.splitext(file)[1]
                        new_file = new_prefix + ext
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(root, new_file)
                        try:
                            os.rename(old_path, new_path)
                            print(f"[{desc}] Renamed: {old_path} -> {new_path}")
                        except Exception as e:
                            print(f"Error renaming {old_path}: {e}")
                    else:
                        print(f"Number {n} exceeds the number of rows in the Excel file {excel_path} (file: {file}).")
                else:
                    # If the file name does not match the pattern "幻灯片n", skip it
                    continue

def main():
    parser = argparse.ArgumentParser(
        description="Execute one of two functions based on command line parameter: add column to Excel or rename image files."
    )
    parser.add_argument(
        "--func",
        choices=["add_column", "rename_images"],
        required=True,
        help="Choose the function to run: 'add_column' to add a column to Excel, 'rename_images' to rename image files."
    )
    args = parser.parse_args()
    
    if args.func == "add_column":
        add_column_to_excel()
    elif args.func == "rename_images":
        rename_images()

if __name__ == "__main__":
    main()
