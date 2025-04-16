import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # For more complex layouts
import argparse
import os
import glob # To find files matching a pattern
import re   # To extract sigma from filename

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
                # print(f"Found pair for sigma {sigma_str}") # Optional: verbosity
            else:
                print(f"Warning: Deblurred file missing for sigma {sigma_str}: {deblurred_file_path}")

    if not sigma_files:
         print(f"Warning: No matching blurred/deblurred pairs found for base name '{base_name}'")
         
    return sigma_files


def plot_grid_comparison(original_img, sigma_pairs, main_title, output_filename):
    """
    Creates and saves a grid plot comparing original, blurred, and deblurred images for multiple sigma values.

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

    # Use a visually appealing style
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Create Figure and GridSpec Layout ---
    # Need num_sigmas rows for pairs + 1 row for the original image
    # 2 columns for blurred/deblurred pairs
    # Adjust figsize: width 10 is reasonable for 2 columns, height depends on number of sigmas
    fig_height = 3 + num_sigmas * 2.5 # Base height + height per sigma row
    fig = plt.figure(figsize=(10, fig_height)) 

    # Create a grid: (num_sigmas + 1) rows, 2 columns
    # The first row is for the original image, spanning both columns
    # The subsequent rows are for blurred/deblurred pairs
    gs = gridspec.GridSpec(num_sigmas +1 , 2, figure=fig, hspace=0.2, wspace=0.0) # Add spacing

    # --- Plot Original Image (Top Row) ---
    ax_orig = fig.add_subplot(gs[0, :]) # Span columns 0 and 1 in row 0
    if original_img is not None:
        ax_orig.imshow(original_img)
        ax_orig.set_title('Original Sharp Image', fontsize=16, pad=10) # Add padding
    else:
        ax_orig.set_title('Original Image Not Loaded', fontsize=16, pad=10, color='red')
    ax_orig.axis('off')

    # --- Plot Blurred/Deblurred Pairs ---
    for i, sigma in enumerate(sorted_sigmas):
        row_index = i + 1 # Start plotting pairs from the second row (index 1)
        
        paths = sigma_pairs[sigma]
        blurred_img = load_image(paths['blurred'])
        deblurred_img = load_image(paths['deblurred'])

        # Plot Blurred Image
        ax_blur = fig.add_subplot(gs[row_index, 0]) # Row i+1, Column 0
        if blurred_img is not None:
            ax_blur.imshow(blurred_img)
            ax_blur.set_title(f'Blurred (Sigma={sigma:.1f})', fontsize=12)
        else:
             ax_blur.set_title(f'Blurred (Sigma={sigma:.1f})\nNot Loaded', fontsize=12, color='red')
        ax_blur.axis('off')

        # Plot Deblurred Image
        ax_deblur = fig.add_subplot(gs[row_index, 1]) # Row i+1, Column 1
        if deblurred_img is not None:
            ax_deblur.imshow(deblurred_img)
            ax_deblur.set_title(f'Deblurred (Sigma={sigma:.1f})', fontsize=12)
        else:
             ax_deblur.set_title(f'Deblurred (Sigma={sigma:.1f})\nNot Loaded', fontsize=12, color='red')
        ax_deblur.axis('off')


    # Add a main title for the entire figure
    fig.suptitle(main_title, fontsize=20, y=.9) # Adjust y based on figsize/layout

    # Adjust layout slightly AFTER adding titles
    # gs.tight_layout(fig, rect=[0, 0, 1, 0.97]) # Adjust rect to make space for suptitle
    # plt.tight_layout(rect=[0, 0.01, 1, 0.97]) # Give suptitle slightly more space

    # Save the plot
    try:
        plt.savefig(output_filename, bbox_inches='tight', dpi=500)
        print(f"Comparison grid plot saved successfully to {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")

    # Optionally display the plot
    plt.show()
    
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
    # Removed --sigma argument as we now find all sigmas
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
        # Decide if you want to exit here or plot without the original
        # return 

    # Create the plot
    plot_title = f"Wiener Deconvolution Results Grid: {args.base_name.capitalize()}"
    plot_grid_comparison(original_img, sigma_pairs, plot_title, args.output_plot)

if __name__ == "__main__":
    main()
