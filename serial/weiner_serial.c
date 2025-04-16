/* * Example: Serial Wiener deconvolution for an RGB image.
 * * This program performs image deblurring using the Wiener algorithm
 * for an RGB image, processing each channel independently on a single core.
 * * - Uses STB libraries for loading/saving images
 * - Assumes input and PSF are the same size initially
 * - Processes the entire image on one process (Rank 0 if run via mpirun,
 * or just normally if compiled without MPI).
 * * Requirements:
 * - FFTW3 library (http://www.fftw.org/)
 * - STB single-file header libraries:
 * https://github.com/nothings/stb/blob/master/stb_image.h
 * https://github.com/nothings/stb/blob/master/stb_image_write.h
 * (Download and place them in your project directory or include path)
 * - A C compiler (like gcc)
 * * Compilation (without MPI):
 * gcc wiener_serial.c -lfftw3 -lm -o wiener_serial
 * * Compilation (with MPI, but still runs serially on Rank 0):
 * mpicc wiener_serial.c -lfftw3 -lm -o wiener_serial_mpi
 * * Execution:
 * ./wiener_serial input.png psf.png output.png [k_value]
 * or
 * mpirun -np 1 ./wiener_serial_mpi input.png psf.png output.png [k_value] 
 * (Using more than 1 process with mpirun won't speed it up in this version)
 */

#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h> // Include MPI for rank check, but logic runs on rank 0

/* STB libraries:
   1) Place these #defines before including the .h files to enable implementations.
   2) Download stb_image.h, stb_image_write.h from https://github.com/nothings/stb
*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"


/* * read_image_rgb:
 * Reads a 3-channel RGB image from file and returns float data in range [0,1].
 *
 * Parameters:
 * - path: path to the input image file (PNG, JPG, etc.)
 * - out_w, out_h: pointers to store the image width and height
 *
 * Returns:
 * - A float array of size (3 * width * height) containing RGB in row-major order,
 * channels interleaved (RGBRGBRGB...).
 * - NULL on failure
 */
static float *read_image_rgb(const char *path, int *out_w, int *out_h)
{
    int channels_in_file;
    // Force 3 channels (RGB)
    unsigned char *data_u8 = stbi_load(path, out_w, out_h, &channels_in_file, 3); 
    if (!data_u8) {
        fprintf(stderr, "Error: Could not load image %s\n", path);
        return NULL;
    }
    int w = *out_w;
    int h = *out_h;

    /* Allocate float array for 3 channels. */
    float *data_f = (float *)malloc(sizeof(float) * w * h * 3);
    if (!data_f) {
        fprintf(stderr, "Error: Could not allocate memory for image.\n");
        stbi_image_free(data_u8);
        return NULL;
    }

    /* Convert from unsigned char [0..255] to float [0..1]. */
    int i;
    for (i = 0; i < w * h * 3; i++) {
        data_f[i] = (float)(data_u8[i]) / 255.0f;
    }

    stbi_image_free(data_u8);
    return data_f;
}

/*
 * read_image_gray:
 * Loads a grayscale image and converts it to floating-point format.
 *
 * Parameters:
 * - path: file path to the input image
 * - out_w, out_h: pointers to receive image dimensions
 *
 * Returns:
 * - Float array containing normalized [0..1] pixel values
 * - NULL if loading fails
 */
static float *read_image_gray(const char *path, int *out_w, int *out_h)
{
    int channels_in_file;
    // Force 1 channel (grayscale)
    unsigned char *data_u8 = stbi_load(path, out_w, out_h, &channels_in_file, 1); 
    if (!data_u8) {
        fprintf(stderr, "Error: Could not load image %s\n", path);
        return NULL;
    }
    int w = *out_w;
    int h = *out_h;

    float *data_f = (float *)malloc(sizeof(float) * w * h);
    if (!data_f) {
        fprintf(stderr, "Error: Could not allocate memory for PSF.\n");
        stbi_image_free(data_u8);
        return NULL;
    }

    int i;
    for (i = 0; i < w * h; i++) {
        data_f[i] = (float)(data_u8[i]) / 255.0f;
    }

    stbi_image_free(data_u8);
    return data_f;
}

/*
 * write_image_rgb:
 * Saves a floating-point RGB image (interleaved channels) as a PNG file.
 *
 * Parameters:
 * - path: output file path
 * - data: float array with RGB values in range [0..1] (RGBRGB...)
 * - w, h: width and height of the image
 *
 * Returns:
 * - 1 on success, 0 on failure
 *
 * Notes:
 * - Values outside [0..1] range are clamped
 */
static int write_image_rgb(const char *path, const float *data, int w, int h)
{
    /* Convert from float [0..1] to unsigned char [0..255]. */
    unsigned char *out_u8 = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 3);
    if (!out_u8) {
        fprintf(stderr, "Error: cannot allocate memory for output image.\n");
        return 0;
    }
    int i;
    for (i = 0; i < w * h * 3; i++) {
        float val = data[i];
        // Clamp values to [0, 1] range before converting
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        out_u8[i] = (unsigned char)(val * 255.0f + 0.5f); // Add 0.5 for rounding
    }

    /* stbi_write_png expects row-major with 3 channels (RGBRGB...). */
    int stride_bytes = w * 3; // Bytes per row
    int ret = stbi_write_png(path, w, h, 3, out_u8, stride_bytes);
    free(out_u8);
    if (!ret) {
        fprintf(stderr, "Error: stbi_write_png failed for %s\n", path);
        return 0;
    }
    return 1;
}

/*
 * center_psf:
 * Rearranges PSF quadrants using fftshift logic to center it for FFT processing.
 * The low-frequency components should be at the corners after centering
 * for correct convolution in the frequency domain using FFT.
 *
 * Parameters:
 * - psf: the point spread function data to be centered
 * - w, h: dimensions of the PSF image
 *
 * Notes:
 * - Modifies the PSF in-place
 * - Required for proper frequency-domain processing
 */
static void center_psf(float *psf, int w, int h) 
{
    float *temp = (float *)malloc(w * h * sizeof(float));
    if (!temp) {
        fprintf(stderr, "Error: Out of memory in center_psf\n");
        // Consider aborting or returning an error code
        return; 
    }
    
    // Copy to temporary buffer first to avoid overwriting needed values
    memcpy(temp, psf, w * h * sizeof(float));

    int half_w = w / 2;
    int half_h = h / 2;
    int i, j;

    // Perform the quadrant swap (fftshift)
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            int src_i = i;
            int src_j = j;
            int dst_i = (i + half_h) % h;
            int dst_j = (j + half_w) % w;
            psf[dst_i * w + dst_j] = temp[src_i * w + src_j];
        }
    }
    
    free(temp);
}


/*
 * normalize_psf:
 * Scales the PSF values so they sum to 1.0 (energy conservation).
 *
 * Parameters:
 * - psf: the point spread function data to normalize
 * - w, h: dimensions of the PSF image
 *
 * Notes:
 * - Modifies the PSF in-place
 */
static void normalize_psf(float *psf, int w, int h)
{
    double sum = 0.0;
    int i;
    
    /* Calculate sum */
    for (i = 0; i < w * h; i++) {
        sum += psf[i];
    }
    
    /* Avoid division by zero or near-zero */
    if (fabs(sum) > 1e-10) { // Use fabs for floating point comparison
        for (i = 0; i < w * h; i++) {
            psf[i] /= (float)sum; // Cast sum back to float if needed
        }
    } else {
        fprintf(stderr, "Warning: PSF sum is close to zero (%e). Normalization skipped.\n", sum);
        // Optional: Set PSF to a delta function or handle error
    }
}

/*
 * wiener_deconv_channel:
 * Performs Wiener deconvolution on a single-channel image.
 * * Parameters:
 * - channel_in:  Input blurred image data for one channel (float [0..1])
 * - full_psf:    The *full* centered and normalized PSF data (float [0..1])
 * - channel_out: Buffer for the deconvolved result for this channel (float)
 * - img_w, img_h: Dimensions of the image and PSF (must be the same)
 * - K:           Wiener regularization parameter (controls noise sensitivity)
 *
 * Notes:
 * - Uses FFTW3 for FFT operations.
 * - K parameter balances deblurring vs. noise amplification. Good values
 * often range from 0.0001 to 0.1, depending on noise.
 */
static void wiener_deconv_channel(
    const float *channel_in,
    const float *full_psf, // Use the full PSF here
    float *channel_out,
    int img_w,
    int img_h,
    double K
) {
    int N = img_w * img_h; // Total number of pixels
    int i;
    
    /* Allocate double arrays for FFTW. FFTW works best with double precision. */
    // Use fftw_malloc for potentially aligned memory, which can be faster
    double *in_spatial  = (double *)fftw_malloc(sizeof(double) * N);
    double *psf_spatial = (double *)fftw_malloc(sizeof(double) * N);
    double *out_spatial = (double *)fftw_malloc(sizeof(double) * N); // For inverse FFT result

    if (!in_spatial || !psf_spatial || !out_spatial) {
        fprintf(stderr, "Error: FFTW malloc failed in wiener_deconv_channel.\n");
        // Free any successfully allocated buffers before returning
        fftw_free(in_spatial); 
        fftw_free(psf_spatial);
        fftw_free(out_spatial);
        // You might want to signal an error more formally
        return; 
    }
    
    /* Copy input channel and full PSF data to double arrays */
    for (i = 0; i < N; i++) {
        in_spatial[i]  = (double)channel_in[i];
        psf_spatial[i] = (double)full_psf[i]; // Use the full PSF passed in
    }
    
    /* Calculate the size of the complex output array for r2c transform */
    // For a real input of size H x W, the complex output has H x (W/2 + 1) elements
    int n_complex_out = img_h * (img_w / 2 + 1); 

    /* Allocate frequency-domain (complex) buffers */
    fftw_complex *freq_in  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);
    fftw_complex *freq_psf = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);
    fftw_complex *freq_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);

    if (!freq_in || !freq_psf || !freq_out) {
        fprintf(stderr, "Error: FFTW malloc failed for complex buffers.\n");
        fftw_free(in_spatial); fftw_free(psf_spatial); fftw_free(out_spatial);
        fftw_free(freq_in); fftw_free(freq_psf); // Free partially allocated complex buffers
        return;
    }

    /* Create FFTW plans */
    // Plan for forward FFT (Real to Complex) for the input image channel
    fftw_plan p_fwd_in = fftw_plan_dft_r2c_2d(img_h, img_w,
                                              in_spatial, freq_in,
                                              FFTW_ESTIMATE); // Use ESTIMATE or MEASURE

    // Plan for forward FFT (Real to Complex) for the PSF
    fftw_plan p_fwd_psf = fftw_plan_dft_r2c_2d(img_h, img_w,
                                               psf_spatial, freq_psf,
                                               FFTW_ESTIMATE);

    // Plan for inverse FFT (Complex to Real) for the result
    // Note: Output buffer is `out_spatial` (must be double*)
    fftw_plan p_inv_out = fftw_plan_dft_c2r_2d(img_h, img_w,
                                               freq_out, out_spatial, 
                                               FFTW_ESTIMATE);

    if (!p_fwd_in || !p_fwd_psf || !p_inv_out) {
         fprintf(stderr, "Error: Failed to create FFTW plans.\n");
         // Cleanup allocated memory and plans
         fftw_destroy_plan(p_fwd_in); // Safe to call on NULL
         fftw_destroy_plan(p_fwd_psf);
         fftw_destroy_plan(p_inv_out);
         fftw_free(in_spatial); fftw_free(psf_spatial); fftw_free(out_spatial);
         fftw_free(freq_in); fftw_free(freq_psf); fftw_free(freq_out);
         return;
    }


    /* Execute forward FFTs */
    fftw_execute(p_fwd_in);
    fftw_execute(p_fwd_psf);

    /* Apply Wiener filter in the frequency domain:
     * F_out(k) = F_in(k) * conjugate(H(k)) / (|H(k)|^2 + K)
     * where:
     * F_out is the frequency domain representation of the estimated deblurred image
     * F_in  is the frequency domain representation of the blurred input image
     * H     is the frequency domain representation of the PSF (Optical Transfer Function)
     * K     is the regularization parameter (noise-to-signal power ratio estimate)
     * k     represents frequency coordinates
     * conjugate(H(k)) is the complex conjugate of H(k)
     * |H(k)|^2 is the squared magnitude of H(k)
     */
    for (i = 0; i < n_complex_out; i++) {
        // Get real and imaginary parts of input FFT and PSF FFT
        double B_r = freq_in[i][0];  // Real part of blurred image FFT
        double B_i = freq_in[i][1];  // Imaginary part of blurred image FFT
        double H_r = freq_psf[i][0]; // Real part of PSF FFT (OTF)
        double H_i = freq_psf[i][1]; // Imaginary part of PSF FFT (OTF)

        // Calculate the squared magnitude of the PSF FFT: |H(k)|^2 = H_r^2 + H_i^2
        double H_mag_sq = H_r * H_r + H_i * H_i; 

        // Calculate the denominator: |H(k)|^2 + K
        double denom = H_mag_sq + K;

        // Calculate the numerator: F_in(k) * conjugate(H(k))
        // conjugate(H(k)) = H_r - j*H_i
        // (B_r + j*B_i) * (H_r - j*H_i) = (B_r*H_r + B_i*H_i) + j*(B_i*H_r - B_r*H_i)
        double num_r = B_r * H_r + B_i * H_i;
        double num_i = B_i * H_r - B_r * H_i;

        // Calculate the filtered frequency components F_out(k) = numerator / denominator
        // Avoid division by zero, although K should prevent this if K > 0
        if (fabs(denom) < 1e-15) { // Use a small tolerance for floating point comparison
            freq_out[i][0] = 0.0;
            freq_out[i][1] = 0.0;
        } else {
            freq_out[i][0] = num_r / denom; // Real part of result
            freq_out[i][1] = num_i / denom; // Imaginary part of result
        }
    }

    /* Execute inverse FFT to get the deblurred spatial domain result */
    // Result is placed in `out_spatial`
    fftw_execute(p_inv_out);

    /* Normalize the result of the inverse FFT and copy back to the float output buffer.
     * FFTW's c2r transform does not normalize, so we need to divide by N.
     */
    double scale = 1.0 / (double)N;
    for (i = 0; i < N; i++) {
        // Ensure the output is properly scaled
        double val = out_spatial[i] * scale; 
        channel_out[i] = (float)val;
    }

    /* Cleanup FFTW plans and allocated memory */
    fftw_destroy_plan(p_fwd_in);
    fftw_destroy_plan(p_fwd_psf);
    fftw_destroy_plan(p_inv_out);

    fftw_free(freq_in);
    fftw_free(freq_psf);
    fftw_free(freq_out);
    fftw_free(in_spatial);
    fftw_free(psf_spatial);
    fftw_free(out_spatial);
    
    // No need to free full_psf here, it's managed outside
}


/*
 * main:
 * Entry point for the serial Wiener deconvolution program.
 *
 * Usage:
 * ./wiener_serial input.png psf.png output.png [k_value]
 *
 * Steps:
 * 1) Initialize MPI (optional, allows running via mpirun but logic is serial)
 * 2) Check rank, only proceed if rank is 0.
 * 3) Load input RGB image and grayscale PSF.
 * 4) Verify dimensions match.
 * 5) Center and normalize the PSF.
 * 6) Allocate memory for output image and separate R, G, B channels.
 * 7) Extract R, G, B channels from the input image.
 * 8) Perform Wiener deconvolution on each channel using the full PSF.
 * 9) Combine the processed R, G, B channels into the output image buffer.
 * 10) Save the final deblurred image to disk.
 * 11) Clean up allocated memory.
 * 12) Finalize MPI.
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv); // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    // --- Only Rank 0 performs the work ---
    if (rank == 0) { 
        if (argc < 4) {
            fprintf(stderr, "Usage: %s <input_rgb_image> <psf_gray_image> <output_image> [k_value]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort all MPI processes
            return 1; // Should not be reached after Abort
        }

        const char *input_path  = argv[1];
        const char *psf_path    = argv[2];
        const char *output_path = argv[3];

        /* Wiener regularization constant K. 
         * Controls the trade-off between deblurring and noise amplification.
         * Smaller K = more deblurring, more noise. Larger K = less deblurring, less noise.
         * Good starting values are often between 0.0001 and 0.1.
         * Can be passed as 4th argument or use a default. 
         */
        double K = 0.001; // Default K value - potentially adjust this
        if (argc > 4) {
            K = atof(argv[4]);
            if (K < 0) {
                fprintf(stderr, "Warning: K value cannot be negative. Using default K = %f\n", 0.001);
                K = 0.001;
            }
        }
        printf("Using Wiener K value: %f\n", K);

        /* Image dimensions */
        int img_w = 0, img_h = 0;
        int psf_w = 0, psf_h = 0;

        /* Load the input color image (RGB interleaved) */
        float *img_in_rgb = read_image_rgb(input_path, &img_w, &img_h);
        if (!img_in_rgb) {
            fprintf(stderr, "Error: Failed to load input image '%s'.\n", input_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        printf("Loaded input image: %d x %d pixels\n", img_w, img_h);

        /* Load the grayscale PSF */
        float *psf_gray = read_image_gray(psf_path, &psf_w, &psf_h);
        if (!psf_gray) {
            fprintf(stderr, "Error: Failed to load PSF image '%s'.\n", psf_path);
            free(img_in_rgb); // Clean up previously allocated memory
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        printf("Loaded PSF image: %d x %d pixels\n", psf_w, psf_h);


        /* --- Sanity Checks --- */
        if (img_w <= 0 || img_h <= 0) {
             fprintf(stderr, "Error: Invalid image dimensions loaded (%dx%d).\n", img_w, img_h);
             free(img_in_rgb); free(psf_gray);
             MPI_Abort(MPI_COMM_WORLD, 1);
             return 1;
        }
        // Wiener deconvolution typically requires the PSF and image to have the same dimensions
        // for standard FFT-based methods.
        if (psf_w != img_w || psf_h != img_h) {
            fprintf(stderr, 
                    "Error: PSF dimensions (%dx%d) must match image dimensions (%dx%d) for this implementation.\n",
                    psf_w, psf_h, img_w, img_h);
            // You could potentially pad/crop the PSF here, but this example assumes matching sizes.
            free(img_in_rgb);
            free(psf_gray);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* --- Prepare PSF --- */
        // 1. Center the PSF (fftshift)
        center_psf(psf_gray, psf_w, psf_h); 
        // 2. Normalize the PSF (sum to 1)
        normalize_psf(psf_gray, psf_w, psf_h);
        printf("PSF centered and normalized.\n");


        /* --- Allocate Memory --- */
        int N = img_w * img_h; // Pixels per channel
        // Allocate buffer for the final RGB output image
        float *img_out_rgb = (float *)malloc(sizeof(float) * N * 3); 
        // Allocate separate buffers for each input color channel
        float *channel_r_in = (float *)malloc(sizeof(float) * N);
        float *channel_g_in = (float *)malloc(sizeof(float) * N);
        float *channel_b_in = (float *)malloc(sizeof(float) * N);
        // Allocate separate buffers for each output color channel
        float *channel_r_out = (float *)malloc(sizeof(float) * N);
        float *channel_g_out = (float *)malloc(sizeof(float) * N);
        float *channel_b_out = (float *)malloc(sizeof(float) * N);

        if (!img_out_rgb || !channel_r_in || !channel_g_in || !channel_b_in ||
            !channel_r_out || !channel_g_out || !channel_b_out) 
        {
            fprintf(stderr, "Error: Failed to allocate memory for channel processing.\n");
            // Free everything allocated so far
            free(img_in_rgb); free(psf_gray); free(img_out_rgb);
            free(channel_r_in); free(channel_g_in); free(channel_b_in);
            free(channel_r_out); free(channel_g_out); free(channel_b_out);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        printf("Memory allocated for processing.\n");

        /* --- Separate Color Channels --- */
        // Input image is RGBRGBRGB...
        int i, pixel_idx;
        for (i = 0; i < N; ++i) {
            pixel_idx = i * 3; // Base index for the i-th pixel in the RGB buffer
            channel_r_in[i] = img_in_rgb[pixel_idx + 0]; // Red
            channel_g_in[i] = img_in_rgb[pixel_idx + 1]; // Green
            channel_b_in[i] = img_in_rgb[pixel_idx + 2]; // Blue
        }
        printf("Color channels separated.\n");

        /* --- Perform Wiener Deconvolution on each channel --- */
        printf("Starting deconvolution for Red channel...\n");
        wiener_deconv_channel(channel_r_in, psf_gray, channel_r_out, img_w, img_h, K);
        printf("Starting deconvolution for Green channel...\n");
        wiener_deconv_channel(channel_g_in, psf_gray, channel_g_out, img_w, img_h, K);
        printf("Starting deconvolution for Blue channel...\n");
        wiener_deconv_channel(channel_b_in, psf_gray, channel_b_out, img_w, img_h, K);
        printf("Deconvolution finished for all channels.\n");

        /* --- Combine Channels into Output Image --- */
        // Output image needs to be RGBRGBRGB...
        for (i = 0; i < N; ++i) {
            pixel_idx = i * 3; // Base index for the i-th pixel in the RGB buffer
            img_out_rgb[pixel_idx + 0] = channel_r_out[i]; // Red
            img_out_rgb[pixel_idx + 1] = channel_g_out[i]; // Green
            img_out_rgb[pixel_idx + 2] = channel_b_out[i]; // Blue
        }
        printf("Processed channels combined.\n");

        /* --- Save Output Image --- */
        if (!write_image_rgb(output_path, img_out_rgb, img_w, img_h)) {
            fprintf(stderr, "Error: Failed to write output image to %s\n", output_path);
            // Continue to cleanup even if writing fails
        } else {
            printf("Successfully wrote deblurred image to %s\n", output_path);
        }

        /* --- Cleanup --- */
        printf("Cleaning up memory...\n");
        free(img_in_rgb);
        free(psf_gray);
        free(img_out_rgb);
        free(channel_r_in);
        free(channel_g_in);
        free(channel_b_in);
        free(channel_r_out);
        free(channel_g_out);
        free(channel_b_out);
        printf("Cleanup complete.\n");

    } // End of rank == 0 block

    // All processes call Finalize
    MPI_Finalize(); 
    return 0;
}
