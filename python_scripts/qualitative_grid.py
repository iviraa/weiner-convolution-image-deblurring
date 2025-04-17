import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # For more complex layouts
import argparse
import os
import glob # To find files matching a pattern
import re   # To extract sigma from filename
import math # For ceil

def load_image(image_path):
    """Loads an image using OpenCV and converts to RGB."""
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path} (file might be corrupted or invalid format)")
        return None
    # Convert from BGR (OpenCV default) to RGB (Matplotlib default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def find_sigma_pairs(blurred_dir, deblurred_dir, base_name):
    """
    Finds pairs of blurred and deblurred images based on sigma values in filenames.

    Args:
        blurred_dir (str): Directory containing blurred images.
        deblurred_dir (str): Directory containing deblurred images.
        base_name (str): Base name used in filenames.

    Returns:
        dict: A dictionary where keys are sigma values (float) and values are
              dicts containing paths {'blurred': path, 'deblurred': path}.
              Returns None if directories are invalid.
    """
    if not os.path.isdir(blurred_dir):
        print(f"Error: Blurred directory not found: {blurred_dir}")
        return None
    if not os.path.isdir(deblurred_dir):
        print(f"Error: Deblurred directory not found: {deblurred_dir}")
        return None
        
    sigma_files = {}
    # Regex to find sigma value (e.g., 1.5) in filenames like 'basename_blurred_sigma1.5.png'
    pattern_regex = re.compile(f"^{re.escape(base_name)}_blurred_sigma(\\d+\\.\\d+)\\.png$")

    print(f"Scanning for blurred images in: {blurred_dir}")
    
    # Iterate through files in the blurred directory
    for filename in os.listdir(blurred_dir):
        match = pattern_regex.match(filename)
        if match:
            sigma = float(match.group(1))
            sigma_str = f"{sigma:.1f}" # Ensure consistent formatting
            
            blurred_file_path = os.path.join(blurred_dir, filename)
            # Construct expected deblurred filename
            deblurred_filename = f"{base_name}_deblurred_sigma{sigma_str}.png"
            deblurred_file_path = os.path.join(deblurred_dir, deblurred_filename)

            # Check if the corresponding deblurred file exists
            if os.path.exists(deblurred_file_path):
                sigma_files[sigma] = {'blurred': blurred_file_path, 'deblurred': deblurred_file_path}
            else:
                print(f"Warning: Deblurred file missing for sigma {sigma_str}: {deblurred_file_path}")

    if not sigma_files:
         print(f"Warning: No matching blurred/deblurred pairs found for base name '{base_name}'")
         
    return sigma_files


def plot_grid_comparison(original_img, sigma_pairs, main_title, output_filename):
    """
    Creates and saves a wider grid plot comparing original, blurred, and deblurred images 
    for multiple sigma values, arranged in side-by-side blocks.

    Args:
        original_img (numpy.ndarray): The original sharp image (RGB).
        sigma_pairs (dict): Dictionary mapping sigma values to {'blurred': path, 'deblurred': path}.
        main_title (str): The main title for the plot.
        output_filename (str): Path to save the plot image.
    """
    if not sigma_pairs:
        print("No sigma pairs provided to plot.")
        return
        
    # Sort the sigma values for consistent plot order
    sorted_sigmas = sorted(sigma_pairs.keys())
    num_sigmas = len(sorted_sigmas)
    
    # --- Determine Grid Layout ---
    # We want blocks of 2 columns (Blurred | Deblurred)
    # Arrange these blocks side-by-side. Let's aim for 2 blocks wide (4 columns total)
    pairs_per_block_col = 4 # Target number of pairs vertically in each block
    num_block_cols = 2 # Number of side-by-side blocks (each block is 2 image columns wide)
    grid_cols = num_block_cols * 2 # Total columns for pairs (Blurred | Deblurred | Blurred | Deblurred)
    # Calculate rows needed for pairs: ceiling(num_sigmas / num_block_cols)
    grid_rows_pairs = math.ceil(num_sigmas / num_block_cols) 
    
    # Total rows = 1 (for original) + rows needed for pairs
    total_rows = 1 + grid_rows_pairs 

    # Use a visually appealing style
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Create Figure and GridSpec Layout ---
    # Adjust figsize: make it wider, height depends on pair rows
    fig_width = 18 # Wider to accommodate 4 image columns
    fig_height = 4 + grid_rows_pairs * 3 # Base height + height per pair row
    fig = plt.figure(figsize=(fig_width, fig_height)) 

    # Create the main grid spec
    gs = gridspec.GridSpec(total_rows, grid_cols, figure=fig, 
                           hspace=0.3, # Adjust vertical space
                           wspace=0.1) # Adjust horizontal space

    # --- Plot Original Image (Top Row, Spanning All Columns) ---
    ax_orig = fig.add_subplot(gs[0, :]) # Span all columns in row 0
    if original_img is not None:
        ax_orig.imshow(original_img)
        ax_orig.set_title('Original Sharp Image', fontsize=18, pad=15) # Larger padding
    else:
        ax_orig.set_title('Original Image Not Loaded', fontsize=18, pad=15, color='red')
    ax_orig.axis('off')

    # --- Plot Blurred/Deblurred Pairs in the Grid ---
    print(f"Arranging {num_sigmas} pairs into {grid_rows_pairs}x{num_block_cols} blocks (each 1x2 images)")
    for i, sigma in enumerate(sorted_sigmas):
        
        # Calculate position in the grid
        # block_col_index determines which vertical block (0 or 1)
        block_col_index = i % num_block_cols 
        # row_in_block determines the row within that block's vertical space
        row_in_block = i // num_block_cols 
        
        # Calculate actual grid coordinates
        target_row = row_in_block + 1 # Add 1 because original image is in row 0
        target_col_start = block_col_index * 2 # Each block takes 2 columns

        paths = sigma_pairs[sigma]
        blurred_img = load_image(paths['blurred'])
        deblurred_img = load_image(paths['deblurred'])

        # Plot Blurred Image
        ax_blur = fig.add_subplot(gs[target_row, target_col_start]) 
        if blurred_img is not None:
            ax_blur.imshow(blurred_img)
            ax_blur.set_title(f'Blurred (Sigma={sigma:.1f})', fontsize=12)
        else:
             ax_blur.set_title(f'Blurred (Sigma={sigma:.1f})\nNot Loaded', fontsize=12, color='red')
        ax_blur.axis('off')

        # Plot Deblurred Image
        ax_deblur = fig.add_subplot(gs[target_row, target_col_start + 1]) # Next column over
        if deblurred_img is not None:
            ax_deblur.imshow(deblurred_img)
            ax_deblur.set_title(f'Deblurred (Sigma={sigma:.1f})', fontsize=12)
        else:
             ax_deblur.set_title(f'Deblurred (Sigma={sigma:.1f})\nNot Loaded', fontsize=12, color='red')
        ax_deblur.axis('off')


    # Add a main title for the entire figure
    fig.suptitle(main_title, fontsize=22, y=.93) # Adjust y position if needed

    # Adjust layout AFTER adding titles
    # plt.tight_layout(rect=[0, 0.01, 1, 0.99]) # Adjust rect to make space for suptitle

    # Save the plot
    try:
        plt.savefig(output_filename, bbox_inches='tight', dpi=350) # Use dpi=150 or higher
        print(f"Comparison grid plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")

    # Optionally display the plot
    # plt.show()
    
    # Close the plot figure to free memory
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a grid comparison plot of original, blurred, and deblurred images for all found sigma values."
    )
    parser.add_argument("--original_image", required=True, help="Path to the original sharp input image.")
    parser.add_argument("--blurred_dir", required=True, help="Directory containing the blurred images.")
    parser.add_argument("--deblurred_dir", required=True, help="Directory containing the deblurred output images.")
    parser.add_argument("--base_name", required=True, help="Base name used for the image files (e.g., 'pokhara').")
    parser.add_argument("--output_plot", required=True, help="Path to save the output plot grid image (e.g., 'comparison_grid.png').")

    args = parser.parse_args()

    # Find matching image pairs
    sigma_pairs = find_sigma_pairs(args.blurred_dir, args.deblurred_dir, args.base_name)

    if not sigma_pairs:
        print("No valid image pairs found. Exiting.")
        return
        
    print(f"Found {len(sigma_pairs)} sigma pairs to plot.")

    # Load original image once
    original_img = load_image(args.original_image)
    if original_img is None:
        print("Warning: Could not load original image. Plot will proceed without it.")

    # Create the plot
    plot_title = f"Wiener Deconvolution Results Grid: {args.base_name.capitalize()}"
    plot_grid_comparison(original_img, sigma_pairs, plot_title, args.output_plot)

if __name__ == "__main__":
    main()
