/* * Example: MPI-based Wiener deconvolution for batches of images.
 * * This program performs image deblurring using the Wiener algorithm
 * processing multiple image/PSF pairs in parallel using MPI master/worker model.
 * * - Rank 0 finds image pairs (based on sigma in filename) and distributes tasks.
 * - Worker ranks (1 to N-1) request tasks, perform serial deconvolution on
 * the assigned image pair, and save the result.
 * - Uses STB libraries for loading/saving images.
 * - Uses POSIX dirent.h for directory scanning (Linux/macOS).
 * * Requirements:
 * - MPI Library (e.g., OpenMPI, MPICH, MS-MPI)
 * - FFTW3 library (http://www.fftw.org/)
 * - STB single-file header libraries (stb_image.h, stb_image_write.h)
 * - A C compiler supporting C99+ (for dirent.h, snprintf)
 * * Compilation:
 * mpicc wiener_parallel_batch.c -lfftw3 -lm -o wiener_parallel_batch -std=c99
 * * Execution:
 * mpirun -np <num_processes> ./wiener_parallel_batch <input_dir> <output_dir> <base_name> [k_value]
 * e.g., mpirun -np 5 ./wiener_parallel_batch output_data output_images pokhara 0.001
 * * <num_processes> should ideally be >= 2 (1 master + 1 or more workers)
 * <input_dir> : Directory containing blurred_*_sigmaX.Y.png and psf_*_sigmaX.Y.png files.
 * <output_dir>: Directory where deblurred_*_sigmaX.Y.png files will be saved.
 * <base_name> : The base name used in the filenames (e.g., 'pokhara').
 * [k_value]   : Optional Wiener K value (default: 0.001).
 */

#include <mpi.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h> // For directory scanning (POSIX)
#include <sys/stat.h> // For mkdir (POSIX)
#include <sys/types.h> // For mkdir (POSIX)
#include <errno.h> // For errno
#include <float.h> // For DBL_MAX

/* STB libraries */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Define tags for MPI communication
#define TAG_TASK_REQUEST 1
#define TAG_TASK_SIGMA   2
#define TAG_TERMINATE    3

// Termination signal for sigma value
#define TERMINATION_SIGMA -1.0

// --- Function Prototypes for Image/FFT Operations (reuse from serial) ---
static float *read_image_rgb(const char *path, int *out_w, int *out_h);
static float *read_image_gray(const char *path, int *out_w, int *out_h);
static int write_image_rgb(const char *path, const float *data, int w, int h);
static void center_psf(float *psf, int w, int h);
static void normalize_psf(float *psf, int w, int h);
static int wiener_deconv_channel(
    const float *channel_in,
    const float *full_psf,
    float *channel_out,
    int img_w,
    int img_h,
    double K); // Changed return type to int for error checking
static int perform_deconvolution_for_sigma(
    double sigma, 
    const char* input_dir, 
    const char* output_dir, 
    const char* base_name, 
    double K,
    int rank); // New function for worker task

// --- Helper Function ---
// Simple function to create directory (POSIX specific)
int create_directory(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        // Use mode 0755 (rwxr-xr-x)
        if (mkdir(path, 0755) != 0) {
             if (errno != EEXIST) { // Don't error if it already exists
                perror("Error creating directory");
                return 0; // Failure
             }
        }
    } else if (!S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Error: %s exists but is not a directory\n", path);
        return 0; // Failure
    }
    return 1; // Success or already exists as directory
}


// ======================== MAIN ========================
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires at least 2 MPI processes (1 master, 1+ workers).\n");
        }
        MPI_Finalize();
        return 1;
    }

    // --- Argument Parsing and Broadcasting ---
    char input_dir[FILENAME_MAX];
    char output_dir[FILENAME_MAX];
    char base_name[FILENAME_MAX];
    double K = 0.001; // Default K value

    if (rank == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s <input_dir> <output_dir> <base_name> [k_value]\n", argv[0]);
            // Signal all processes to abort
            MPI_Abort(MPI_COMM_WORLD, 1); 
        }
        strncpy(input_dir, argv[1], FILENAME_MAX - 1);
        strncpy(output_dir, argv[2], FILENAME_MAX - 1);
        strncpy(base_name, argv[3], FILENAME_MAX - 1);
        input_dir[FILENAME_MAX - 1] = '\0'; // Ensure null termination
        output_dir[FILENAME_MAX - 1] = '\0';
        base_name[FILENAME_MAX - 1] = '\0';

        if (argc > 4) {
            K = atof(argv[4]);
            if (K < 0) {
                fprintf(stderr, "Warning: K value cannot be negative. Using default K = %f\n", 0.001);
                K = 0.001;
            }
        }
        printf("Master (Rank 0): Input='%s', Output='%s', Base='%s', K=%.4f\n", input_dir, output_dir, base_name, K);
        
        // Create output directory
        if (!create_directory(output_dir)) {
             fprintf(stderr, "Master (Rank 0): Failed to create or access output directory '%s'. Aborting.\n", output_dir);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast necessary arguments from Rank 0 to all other ranks
    MPI_Bcast(input_dir, FILENAME_MAX, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_dir, FILENAME_MAX, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(base_name, FILENAME_MAX, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- Master Logic (Rank 0) ---
    if (rank == 0) {
        DIR *dir;
        struct dirent *entry;
        double *sigma_tasks = NULL; // Dynamically allocated array for sigma values
        int task_count = 0;
        int task_capacity = 0;
        char pattern_blur[FILENAME_MAX];
        char pattern_psf[FILENAME_MAX];

        // Prepare filename patterns for sscanf
        snprintf(pattern_blur, FILENAME_MAX, "%s_blurred_sigma%%lf.png", base_name);
        snprintf(pattern_psf, FILENAME_MAX, "%s_psf_sigma%%lf.png", base_name);

        printf("Master: Scanning input directory '%s' for tasks...\n", input_dir);
        dir = opendir(input_dir);
        if (!dir) {
            perror("Master: Error opening input directory");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // --- Scan Directory to Find Tasks ---
        while ((entry = readdir(dir)) != NULL) {
            double current_sigma;
            // Try matching the blurred filename pattern
            if (sscanf(entry->d_name, pattern_blur, &current_sigma) == 1) {
                // Found a potential blurred file, check for corresponding PSF
                char psf_filename[FILENAME_MAX];
                char psf_filepath[FILENAME_MAX];
                snprintf(psf_filename, FILENAME_MAX, "%s_psf_sigma%.1f.png", base_name, current_sigma);
                snprintf(psf_filepath, FILENAME_MAX, "%s/%s", input_dir, psf_filename);

                struct stat st;
                if (stat(psf_filepath, &st) == 0 && S_ISREG(st.st_mode)) {
                    // PSF file exists, add sigma to task list if not already present
                    int found = 0;
                    for(int i = 0; i < task_count; ++i) {
                         // Use a tolerance for float comparison
                         if (fabs(sigma_tasks[i] - current_sigma) < 1e-5) {
                              found = 1;
                              break;
                         }
                    }
                    if (!found) {
                         // Resize task array if needed
                         if (task_count >= task_capacity) {
                              task_capacity = (task_capacity == 0) ? 10 : task_capacity * 2;
                              double *new_tasks = (double*)realloc(sigma_tasks, task_capacity * sizeof(double));
                              if (!new_tasks) {
                                   fprintf(stderr, "Master: Failed to allocate memory for tasks.\n");
                                   free(sigma_tasks); // Free previous allocation
                                   closedir(dir);
                                   MPI_Abort(MPI_COMM_WORLD, 1);
                              }
                              sigma_tasks = new_tasks;
                         }
                         sigma_tasks[task_count++] = current_sigma;
                         printf("Master: Found valid task pair for sigma %.1f\n", current_sigma);
                    }
                }
            }
        }
        closedir(dir);
        printf("Master: Found %d tasks.\n", task_count);

        if (task_count == 0) {
             printf("Master: No valid blurred/PSF pairs found. Telling workers to terminate.\n");
        }

        // --- Distribute Tasks ---
        int tasks_assigned = 0;
        int workers_terminated = 0;
        MPI_Status status;

        // While there are workers still running
        while (workers_terminated < (size - 1)) {
            int worker_rank;
            // Wait for a message from any worker
            MPI_Recv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, TAG_TASK_REQUEST, MPI_COMM_WORLD, &status);

            // Check if there are tasks left to assign
            if (tasks_assigned < task_count) {
                double sigma_to_send = sigma_tasks[tasks_assigned];
                MPI_Send(&sigma_to_send, 1, MPI_DOUBLE, status.MPI_SOURCE, TAG_TASK_SIGMA, MPI_COMM_WORLD);
                tasks_assigned++;
            } else {
                // No tasks left, tell the worker to terminate
                double terminate_signal = TERMINATION_SIGMA;
                MPI_Send(&terminate_signal, 1, MPI_DOUBLE, status.MPI_SOURCE, TAG_TERMINATE, MPI_COMM_WORLD);
                workers_terminated++;
            }
        }
        printf("Master: All tasks assigned and workers terminated.\n");
        free(sigma_tasks); // Free the task list

    } 
    // --- Worker Logic (Ranks > 0) ---
    else { 
        int my_rank = rank; // For clarity in messages
        printf("Worker (Rank %d): Ready for tasks.\n", my_rank);

        while (1) {
            // Request a task from the master
            MPI_Send(&my_rank, 1, MPI_INT, 0, TAG_TASK_REQUEST, MPI_COMM_WORLD);

            // Receive sigma value or termination signal
            double received_sigma;
            MPI_Status status;
            MPI_Recv(&received_sigma, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            // Check if it's the termination signal
            if (status.MPI_TAG == TAG_TERMINATE || received_sigma == TERMINATION_SIGMA) {
                printf("Worker (Rank %d): Received termination signal. Exiting.\n", my_rank);
                break; // Exit the loop
            }

            // Process the assigned sigma value
            if (status.MPI_TAG == TAG_TASK_SIGMA) {
                printf("Worker (Rank %d): Received task for sigma %.1f\n", my_rank, received_sigma);
                
                // Perform the actual deconvolution work
                int success = perform_deconvolution_for_sigma(received_sigma, input_dir, output_dir, base_name, K, my_rank);
                
                if (success) {
                     printf("Worker (Rank %d): Successfully processed sigma %.1f\n", my_rank, received_sigma);
                } else {
                     fprintf(stderr, "Worker (Rank %d): Failed to process sigma %.1f\n", my_rank, received_sigma);
                     // Decide if failure should abort or just continue
                }
            } else {
                 // Unexpected tag - should not happen in this logic
                 fprintf(stderr, "Worker (Rank %d): Received message with unexpected tag %d. Terminating.\n", my_rank, status.MPI_TAG);
                 break;
            }
        } // End of worker loop
    } // End of worker logic

    MPI_Finalize();
    return 0;
}


// ======================== Deconvolution Task Function ========================
/*
 * perform_deconvolution_for_sigma:
 * Loads the relevant images for a given sigma, performs Wiener deconvolution,
 * and saves the result. Called by worker processes.
 *
 * Returns: 1 on success, 0 on failure.
 */
static int perform_deconvolution_for_sigma(
    double sigma, 
    const char* input_dir, 
    const char* output_dir, 
    const char* base_name, 
    double K,
    int rank // Pass rank for logging
) {
    char blurred_filename[FILENAME_MAX];
    char psf_filename[FILENAME_MAX];
    char output_filename[FILENAME_MAX];
    char blurred_filepath[FILENAME_MAX];
    char psf_filepath[FILENAME_MAX];
    char output_filepath[FILENAME_MAX];

    // Construct filenames using sigma with 1 decimal place
    snprintf(blurred_filename, FILENAME_MAX, "%s_blurred_sigma%.1f.png", base_name, sigma);
    snprintf(psf_filename, FILENAME_MAX, "%s_psf_sigma%.1f.png", base_name, sigma);
    snprintf(output_filename, FILENAME_MAX, "%s_deblurred_sigma%.1f.png", base_name, sigma);

    snprintf(blurred_filepath, FILENAME_MAX, "%s/%s", input_dir, blurred_filename);
    snprintf(psf_filepath, FILENAME_MAX, "%s/%s", input_dir, psf_filename);
    snprintf(output_filepath, FILENAME_MAX, "%s/%s", output_dir, output_filename);

    // --- Load Images ---
    int img_w = 0, img_h = 0;
    int psf_w = 0, psf_h = 0;
    float *img_in_rgb = read_image_rgb(blurred_filepath, &img_w, &img_h);
    if (!img_in_rgb) {
        fprintf(stderr, "Worker (Rank %d): Failed to load blurred image '%s'.\n", rank, blurred_filepath);
        return 0; // Failure
    }
    float *psf_gray = read_image_gray(psf_filepath, &psf_w, &psf_h);
    if (!psf_gray) {
        fprintf(stderr, "Worker (Rank %d): Failed to load PSF image '%s'.\n", rank, psf_filepath);
        free(img_in_rgb);
        return 0; // Failure
    }
    // printf("Worker (Rank %d): Loaded images for sigma %.1f (%dx%d)\n", rank, sigma, img_w, img_h);

    // --- Sanity Checks ---
     if (img_w <= 0 || img_h <= 0) {
         fprintf(stderr, "Worker (Rank %d): Invalid image dimensions loaded (%dx%d) for sigma %.1f.\n", rank, img_w, img_h, sigma);
         free(img_in_rgb); free(psf_gray);
         return 0;
     }
    if (psf_w != img_w || psf_h != img_h) {
        fprintf(stderr, "Worker (Rank %d): Error! PSF dimensions (%dx%d) must match image dimensions (%dx%d) for sigma %.1f.\n", rank, psf_w, psf_h, img_w, img_h, sigma);
        free(img_in_rgb); free(psf_gray);
        return 0; // Failure
    }

    // --- Prepare PSF ---
    center_psf(psf_gray, psf_w, psf_h);
    normalize_psf(psf_gray, psf_w, psf_h);

    // --- Allocate Memory ---
    int N = img_w * img_h;
    float *img_out_rgb = (float *)malloc(sizeof(float) * N * 3);
    float *channel_r_in = (float *)malloc(sizeof(float) * N);
    float *channel_g_in = (float *)malloc(sizeof(float) * N);
    float *channel_b_in = (float *)malloc(sizeof(float) * N);
    float *channel_r_out = (float *)malloc(sizeof(float) * N);
    float *channel_g_out = (float *)malloc(sizeof(float) * N);
    float *channel_b_out = (float *)malloc(sizeof(float) * N);

    if (!img_out_rgb || !channel_r_in || !channel_g_in || !channel_b_in ||
        !channel_r_out || !channel_g_out || !channel_b_out) {
        fprintf(stderr, "Worker (Rank %d): Failed to allocate memory for channel processing (sigma %.1f).\n", rank, sigma);
        free(img_in_rgb); free(psf_gray); free(img_out_rgb);
        free(channel_r_in); free(channel_g_in); free(channel_b_in);
        free(channel_r_out); free(channel_g_out); free(channel_b_out);
        return 0; // Failure
    }

    // --- Separate Color Channels ---
    int i, pixel_idx;
    for (i = 0; i < N; ++i) {
        pixel_idx = i * 3;
        channel_r_in[i] = img_in_rgb[pixel_idx + 0];
        channel_g_in[i] = img_in_rgb[pixel_idx + 1];
        channel_b_in[i] = img_in_rgb[pixel_idx + 2];
    }

    // --- Perform Wiener Deconvolution on each channel ---
    // printf("Worker (Rank %d): Starting deconvolution for sigma %.1f...\n", rank, sigma);
    int success_r = wiener_deconv_channel(channel_r_in, psf_gray, channel_r_out, img_w, img_h, K);
    int success_g = wiener_deconv_channel(channel_g_in, psf_gray, channel_g_out, img_w, img_h, K);
    int success_b = wiener_deconv_channel(channel_b_in, psf_gray, channel_b_out, img_w, img_h, K);

    if (!success_r || !success_g || !success_b) {
         fprintf(stderr, "Worker (Rank %d): Deconvolution failed for one or more channels (sigma %.1f).\n", rank, sigma);
         // Cleanup and return failure
         free(img_in_rgb); free(psf_gray); free(img_out_rgb);
         free(channel_r_in); free(channel_g_in); free(channel_b_in);
         free(channel_r_out); free(channel_g_out); free(channel_b_out);
         return 0; // Failure
    }
    // printf("Worker (Rank %d): Deconvolution finished for sigma %.1f.\n", rank, sigma);


    // --- Combine Channels into Output Image ---
    for (i = 0; i < N; ++i) {
        pixel_idx = i * 3;
        img_out_rgb[pixel_idx + 0] = channel_r_out[i];
        img_out_rgb[pixel_idx + 1] = channel_g_out[i];
        img_out_rgb[pixel_idx + 2] = channel_b_out[i];
    }

    // --- Save Output Image ---
    if (!write_image_rgb(output_filepath, img_out_rgb, img_w, img_h)) {
        fprintf(stderr, "Worker (Rank %d): Failed to write output image '%s'.\n", rank, output_filepath);
        // Continue to cleanup even if writing fails, but report failure
         free(img_in_rgb); free(psf_gray); free(img_out_rgb);
         free(channel_r_in); free(channel_g_in); free(channel_b_in);
         free(channel_r_out); free(channel_g_out); free(channel_b_out);
        return 0; // Failure
    } else {
        // printf("Worker (Rank %d): Successfully wrote deblurred image '%s'\n", rank, output_filepath);
    }

    // --- Cleanup ---
    free(img_in_rgb);
    free(psf_gray);
    free(img_out_rgb);
    free(channel_r_in);
    free(channel_g_in);
    free(channel_b_in);
    free(channel_r_out);
    free(channel_g_out);
    free(channel_b_out);

    return 1; // Success
}


// ======================== Image/FFT Functions (from serial) ========================

// --- Implementations of read_image_rgb, read_image_gray, write_image_rgb, ---
// --- center_psf, normalize_psf, wiener_deconv_channel ---
// --- (These functions are identical to the ones in wiener_serial.c,      ---
// ---  except wiener_deconv_channel now returns int for success/failure) ---


static float *read_image_rgb(const char *path, int *out_w, int *out_h) {
    int channels_in_file;
    unsigned char *data_u8 = stbi_load(path, out_w, out_h, &channels_in_file, 3); 
    if (!data_u8) {
        // fprintf(stderr, "Error: Could not load image %s\n", path); // Reduced verbosity
        return NULL;
    }
    int w = *out_w;
    int h = *out_h;
    float *data_f = (float *)malloc(sizeof(float) * w * h * 3);
    if (!data_f) {
        fprintf(stderr, "Error: Could not allocate memory for image.\n");
        stbi_image_free(data_u8);
        return NULL;
    }
    for (int i = 0; i < w * h * 3; i++) {
        data_f[i] = (float)(data_u8[i]) / 255.0f;
    }
    stbi_image_free(data_u8);
    return data_f;
}

static float *read_image_gray(const char *path, int *out_w, int *out_h) {
    int channels_in_file;
    unsigned char *data_u8 = stbi_load(path, out_w, out_h, &channels_in_file, 1); 
    if (!data_u8) {
        // fprintf(stderr, "Error: Could not load image %s\n", path); // Reduced verbosity
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
    for (int i = 0; i < w * h; i++) {
        data_f[i] = (float)(data_u8[i]) / 255.0f;
    }
    stbi_image_free(data_u8);
    return data_f;
}

static int write_image_rgb(const char *path, const float *data, int w, int h) {
    unsigned char *out_u8 = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 3);
    if (!out_u8) {
        fprintf(stderr, "Error: cannot allocate memory for output image.\n");
        return 0;
    }
    for (int i = 0; i < w * h * 3; i++) {
        float val = data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        out_u8[i] = (unsigned char)(val * 255.0f + 0.5f);
    }
    int stride_bytes = w * 3;
    int ret = stbi_write_png(path, w, h, 3, out_u8, stride_bytes);
    free(out_u8);
    if (!ret) {
        // fprintf(stderr, "Error: stbi_write_png failed for %s\n", path); // Reduced verbosity
        return 0;
    }
    return 1;
}

static void center_psf(float *psf, int w, int h) {
    float *temp = (float *)malloc(w * h * sizeof(float));
    if (!temp) {
        fprintf(stderr, "Error: Out of memory in center_psf\n");
        return; 
    }
    memcpy(temp, psf, w * h * sizeof(float));
    int half_w = w / 2;
    int half_h = h / 2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int src_i = i;
            int src_j = j;
            int dst_i = (i + half_h) % h;
            int dst_j = (j + half_w) % w;
            psf[dst_i * w + dst_j] = temp[src_i * w + src_j];
        }
    }
    free(temp);
}

static void normalize_psf(float *psf, int w, int h) {
    double sum = 0.0;
    for (int i = 0; i < w * h; i++) {
        sum += psf[i];
    }
    if (fabs(sum) > 1e-10) {
        for (int i = 0; i < w * h; i++) {
            psf[i] /= (float)sum;
        }
    } else {
        fprintf(stderr, "Warning: PSF sum is close to zero (%e). Normalization skipped.\n", sum);
    }
}

static int wiener_deconv_channel(
    const float *channel_in,
    const float *full_psf, 
    float *channel_out,
    int img_w,
    int img_h,
    double K
) {
    int N = img_w * img_h;
    double *in_spatial  = (double *)fftw_malloc(sizeof(double) * N);
    double *psf_spatial = (double *)fftw_malloc(sizeof(double) * N);
    double *out_spatial = (double *)fftw_malloc(sizeof(double) * N); 
    if (!in_spatial || !psf_spatial || !out_spatial) {
        fprintf(stderr, "Error: FFTW malloc failed for spatial buffers.\n");
        fftw_free(in_spatial); fftw_free(psf_spatial); fftw_free(out_spatial);
        return 0; // Failure
    }
    for (int i = 0; i < N; i++) {
        in_spatial[i]  = (double)channel_in[i];
        psf_spatial[i] = (double)full_psf[i];
    }
    int n_complex_out = img_h * (img_w / 2 + 1); 
    fftw_complex *freq_in  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);
    fftw_complex *freq_psf = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);
    fftw_complex *freq_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_complex_out);
    if (!freq_in || !freq_psf || !freq_out) {
        fprintf(stderr, "Error: FFTW malloc failed for complex buffers.\n");
        fftw_free(in_spatial); fftw_free(psf_spatial); fftw_free(out_spatial);
        fftw_free(freq_in); fftw_free(freq_psf); fftw_free(freq_out);
        return 0; // Failure
    }

    fftw_plan p_fwd_in = fftw_plan_dft_r2c_2d(img_h, img_w, in_spatial, freq_in, FFTW_ESTIMATE);
    fftw_plan p_fwd_psf = fftw_plan_dft_r2c_2d(img_h, img_w, psf_spatial, freq_psf, FFTW_ESTIMATE);
    fftw_plan p_inv_out = fftw_plan_dft_c2r_2d(img_h, img_w, freq_out, out_spatial, FFTW_ESTIMATE);

     if (!p_fwd_in || !p_fwd_psf || !p_inv_out) {
         fprintf(stderr, "Error: Failed to create FFTW plans.\n");
         fftw_destroy_plan(p_fwd_in); fftw_destroy_plan(p_fwd_psf); fftw_destroy_plan(p_inv_out);
         fftw_free(in_spatial); fftw_free(psf_spatial); fftw_free(out_spatial);
         fftw_free(freq_in); fftw_free(freq_psf); fftw_free(freq_out);
         return 0; // Failure
    }

    fftw_execute(p_fwd_in);
    fftw_execute(p_fwd_psf);

    for (int i = 0; i < n_complex_out; i++) {
        double B_r = freq_in[i][0]; double B_i = freq_in[i][1];
        double H_r = freq_psf[i][0]; double H_i = freq_psf[i][1];
        double H_mag_sq = H_r * H_r + H_i * H_i; 
        double denom = H_mag_sq + K;
        double num_r = B_r * H_r + B_i * H_i;
        double num_i = B_i * H_r - B_r * H_i;
        if (fabs(denom) < DBL_MIN) { // Use DBL_MIN for safety
            freq_out[i][0] = 0.0; freq_out[i][1] = 0.0;
        } else {
            freq_out[i][0] = num_r / denom; freq_out[i][1] = num_i / denom;
        }
    }

    fftw_execute(p_inv_out);

    double scale = 1.0 / (double)N;
    for (int i = 0; i < N; i++) {
        channel_out[i] = (float)(out_spatial[i] * scale);
    }

    fftw_destroy_plan(p_fwd_in);
    fftw_destroy_plan(p_fwd_psf);
    fftw_destroy_plan(p_inv_out);
    fftw_free(freq_in); fftw_free(freq_psf); fftw_free(freq_out);
    fftw_free(in_spatial); fftw_free(psf_spatial); fftw_free(out_spatial);
    
    return 1; // Success
}


