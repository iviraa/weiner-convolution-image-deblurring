import cv2
import numpy as np
import argparse
import os

def create_gaussian_psf(shape, sigma):
    """
    Generates a 2D Gaussian Point Spread Function (PSF).

    Args:
        shape (tuple): The desired shape (height, width) of the PSF.
        sigma (float): The standard deviation of the Gaussian blur.

    Returns:
        numpy.ndarray: A 2D float array representing the normalized PSF.
                       Returns None if sigma is non-positive.
    """
    if sigma <= 0:
        print(f"Warning: Sigma must be positive. Skipping sigma={sigma}")
        return None

    h, w = shape
    center_y, center_x = h // 2, w // 2

    # Create coordinate grids
    y, x = np.ogrid[-center_y:h - center_y, -center_x:w - center_x]

    # Calculate Gaussian function values
    psf = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

    # Normalize the PSF so that it sums to 1.0
    psf_sum = np.sum(psf)
    if psf_sum > 1e-9:
        psf /= psf_sum
    else:
        print(f"Warning: PSF sum is near zero for sigma={sigma}. Creating delta function.")
        psf = np.zeros((h, w))
        psf[center_y, center_x] = 1.0

    return psf

def save_psf_as_image(psf, filename):
    """
    Saves a float PSF array as an 8-bit grayscale image.

    Args:
        psf (numpy.ndarray): The float PSF array.
        filename (str): The path to save the image file.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if psf is None:
        print(f"Error: Cannot save None PSF for {filename}")
        return False
        
    # Scale the PSF for visualization (peak becomes 255)
    peak_value = np.max(psf)
    if peak_value > 1e-9:
         psf_u8 = (psf * 255.0 / peak_value).clip(0, 255).astype(np.uint8)
    else:
         psf_u8 = np.zeros(psf.shape, dtype=np.uint8)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, psf_u8)
        # print(f"PSF image saved to {filename}") # Reduce verbosity in loop
        return True
    except Exception as e:
        print(f"Error saving PSF image to {filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Apply Gaussian blur for a range of sigma values to an image (max 1400px) and generate corresponding PSFs."
    )
    parser.add_argument("input_image", help="Path to the sharp input image.")
    parser.add_argument("output_dir", help="Directory to save the blurred images and PSFs.")
    parser.add_argument("--sigma_start", type=float, default=0.5, help="Starting sigma value (default: 0.5).")
    parser.add_argument("--sigma_end", type=float, default=4.5, help="Ending sigma value (default: 4.5).")
    parser.add_argument("--sigma_step", type=float, default=0.5, help="Step for sigma values (default: 0.5).")
    parser.add_argument("--base_name", type=str, default="output", help="Base name for output files (default: 'output').")
    parser.add_argument("--max_dim", type=int, default=1400, help="Maximum dimension (width or height) allowed before resizing (default: 1400).")


    args = parser.parse_args()

    # --- 1. Load the sharp image ---
    try:
        img_to_process = cv2.imread(args.input_image)
        if img_to_process is None:
            raise IOError(f"Could not read input image: {args.input_image}")
        orig_h, orig_w = img_to_process.shape[:2]
        print(f"Loaded sharp image: {args.input_image} (Original size: {orig_w}x{orig_h})")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # --- 1b. Conditionally Resize Image ---
    img_h, img_w = orig_h, orig_w
    if img_h > args.max_dim or img_w > args.max_dim:
        print(f"Image dimensions ({img_w}x{img_h}) exceed max limit ({args.max_dim}px). Resizing...")
        # Calculate scaling factor based on the largest dimension
        scale = args.max_dim / max(img_h, img_w)
        # Calculate new dimensions, ensuring they are integers
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        try:
            # Resize the image using INTER_AREA for downscaling
            img_to_process = cv2.resize(img_to_process, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_h, img_w = new_h, new_w # Update dimensions
            print(f"Resized image to: {img_w}x{img_h}")
        except Exception as e:
            print(f"Error resizing image: {e}")
            return # Stop if resizing fails
    else:
        print(f"Image dimensions ({img_w}x{img_h}) are within the limit ({args.max_dim}px). No resizing needed.")


    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved in: {args.output_dir}")

    # --- Loop through Sigma Values ---
    # Use np.linspace for potentially more robust floating point steps
    num_steps = int(round((args.sigma_end - args.sigma_start) / args.sigma_step)) + 1
    sigmas = np.linspace(args.sigma_start, args.sigma_end, num_steps)
    
    print(f"Generating data for sigmas: {np.round(sigmas, 2)}")

    for sigma in sigmas:
        current_sigma = round(sigma, 2) # Round for processing and filename consistency
        print(f"--- Processing Sigma = {current_sigma} ---")

        # --- 2. Apply Gaussian Blur (to potentially resized image) ---
        if current_sigma <= 0:
            print("Sigma must be positive, skipping.")
            continue
        try:
            # Use the potentially resized image 'img_to_process'
            blurred_img = cv2.GaussianBlur(img_to_process, (0, 0), sigmaX=current_sigma, sigmaY=current_sigma, borderType=cv2.BORDER_DEFAULT)
        except Exception as e:
            print(f"Error applying blur for sigma={current_sigma}: {e}")
            continue # Skip to next sigma

        # --- 3. Save the blurred image ---
        # Use 1 decimal place for sigma in filename for consistency
        blurred_filename = os.path.join(args.output_dir, f"{args.base_name}_blurred_sigma{current_sigma:.1f}.png") 
        try:
            cv2.imwrite(blurred_filename, blurred_img)
            print(f"  Blurred image saved: {blurred_filename}")
        except Exception as e:
            print(f"  Error saving blurred image for sigma={current_sigma}: {e}")

        # --- 4. Generate the PSF ---
        # Use the potentially updated dimensions (img_h, img_w)
        psf_float = create_gaussian_psf((img_h, img_w), current_sigma)

        # --- 5. Save the PSF image ---
        if psf_float is not None:
            # Use 1 decimal place for sigma in filename
            psf_filename = os.path.join(args.output_dir, f"{args.base_name}_psf_sigma{current_sigma:.1f}.png") 
            save_psf_as_image(psf_float, psf_filename)
            print(f"  PSF image saved:     {psf_filename}")
        else:
            print(f"  PSF generation failed for sigma={current_sigma}.")

    print("--- Batch processing finished ---")


if __name__ == "__main__":
    main()

