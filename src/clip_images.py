from PIL import Image, ImageChops
import os
import shutil

def trim_whitespace(input_path, output_path, padding=10):
    """
    Opens an image, trims the whitespace by comparing with a white background,
    adds the specified padding to the cropped region, and saves the result.
    """
    im = Image.open(input_path)
    width, height = im.size

    # Create a white background image with the same size
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    
    # Get the bounding box of the non-white region
    bbox = diff.getbbox()
    if bbox:
        left = max(bbox[0] - padding, 0)
        upper = max(bbox[1] - padding, 0)
        right = min(bbox[2] + padding, width)
        lower = min(bbox[3] + padding // 2, height)  # Adjust lower padding to avoid covering text
        
        cropped_im = im.crop((left, upper, right, lower))
        cropped_im.save(output_path)
    else:
        im.save(output_path)

def process_directory(input_dir, tmp_dir, padding=10):
    """
    Recursively traverses the input_dir, processes all PNG images,
    and saves the processed images to the tmp_dir preserving the relative path.
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".png"):
                input_path = os.path.join(root, file)
                # Compute the relative path from the input directory
                rel_path = os.path.relpath(input_path, input_dir)
                output_tmp = os.path.join(tmp_dir, rel_path)
                
                # Ensure the target directory exists
                os.makedirs(os.path.dirname(output_tmp), exist_ok=True)
                print(f"Processing {input_path} -> {output_tmp}")
                trim_whitespace(input_path, output_tmp, padding)

def copy_tmp_to_source(input_dir, tmp_dir):
    """
    Copies files from tmp_dir back to input_dir (overwriting original files).
    """
    for root, dirs, files in os.walk(tmp_dir):
        for file in files:
            tmp_file = os.path.join(root, file)
            # Compute the relative path from tmp_dir
            rel_path = os.path.relpath(tmp_file, tmp_dir)
            original_file = os.path.join(input_dir, rel_path)
            
            # Ensure the original directory exists (for safety)
            os.makedirs(os.path.dirname(original_file), exist_ok=True)
            print(f"Overwriting {original_file} with {tmp_file}")
            shutil.copy2(tmp_file, original_file)

if __name__ == "__main__":
    # List of directories to process
    directories = ["images_chinese", "images_english"]
    # Temporary root directory
    tmp_root = ".tmp"
    
    # Delete the tmp_root if it already exists to ensure a clean environment
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    
    # Process each directory and save results to corresponding subdirectory in tmp_root
    for d in directories:
        input_dir = d
        tmp_dir = os.path.join(tmp_root, d)
        process_directory(input_dir, tmp_dir, padding=4)
    
    # Copy the processed files from tmp_root back to the original directories
    for d in directories:
        input_dir = d
        tmp_dir = os.path.join(tmp_root, d)
        copy_tmp_to_source(input_dir, tmp_dir)
    
    # Delete the temporary directory after processing is complete
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    print("Processing complete.")
