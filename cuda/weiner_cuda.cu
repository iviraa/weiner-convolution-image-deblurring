/*
 * Example: CUDA-based Wiener deconvolution for batches of images.
 * - This program performs image deblurring using the Wiener algorithm on a GPU.
 * - It processes multiple image/PSF pairs sequentially.
 * - GPU memory for primary data (images, PSF, frequency domain) uses cudaMallocManaged
 *   for potentially simplified memory management (behavior depends on system & GPU).
 * - Uses STB libraries for loading/saving images.
 * - Uses POSIX dirent.h for directory scanning (Linux/macOS).
 *
 * Requirements:
 * - NVIDIA GPU with CUDA support (Compute Capability 3.0+ for full unified memory benefits)
 * - CUDA Toolkit (including cuFFT library)
 * - STB single-file header libraries (stb_image.h, stb_image_write.h)
 * - A C/C++ compiler (nvcc will be used)
 *
 * Compilation (adjust sm_XX to your GPU's compute capability, e.g., sm_37, sm_50, sm_60, sm_75):
 * nvcc -arch=sm_37 wiener_cuda_batch.cu -o wiener_cuda_batch -lcufft -lm
 *
 * Execution:
 * ./wiener_cuda_batch <input_dir> <output_dir> <base_name> [k_value]
 * e.g., ./wiener_cuda_batch output_data output_images_cuda pokhara 0.001
 *
 * <input_dir> : Directory containing blurred_*_sigmaX.Y.png and psf_*_sigmaX.Y.png files.
 * <output_dir>: Directory where deblurred_*_sigmaX.Y.png files will be saved.
 * <base_name> : The base name used in the filenames (e.g., 'pokhara').
 * [k_value]   : Optional Wiener K value (default: 0.001).
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>    // For directory scanning (POSIX)
#include <sys/stat.h>  // For mkdir (POSIX)
#include <sys/types.h> // For mkdir (POSIX)
#include <errno.h>     // For errno
#include <float.h>     // For DBL_MAX / FLT_EPSILON
#include <sys/time.h>  // For gettimeofday

/* STB libraries */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// --- CUDA Error Checking Macros ---
#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t err = call;                                                        \
        if (err != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                          \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)

static const char *_cudaGetCufftErrorString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "Unknown cuFFT error";
    }
}

#define CUFFT_CHECK(call)                                                              \
    do {                                                                               \
        cufftResult_t err = call;                                                      \
        if (err != CUFFT_SUCCESS) {                                                    \
            fprintf(stderr, "cuFFT Error in %s at line %d: %s\n", __FILE__, __LINE__,  \
                    _cudaGetCufftErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)

// Thread block dimensions
#define THREADS_PER_BLOCK_2D_X 16
#define THREADS_PER_BLOCK_2D_Y 16
#define THREADS_PER_BLOCK_1D 256


// --- Function Prototypes for Image/CPU Operations (mostly from serial) ---
static float *read_image_rgb(const char *path, int *out_w, int *out_h);
static float *read_image_gray(const char *path, int *out_w, int *out_h);
static int write_image_rgb(const char *path, const float *data, int w, int h);
static void center_psf(float *psf, int w, int h);
static void normalize_psf(float *psf, int w, int h);
int create_directory(const char *path); // Forward declaration

// --- CUDA Kernels (unchanged) ---
__global__ void wiener_filter_kernel(
    cufftComplex *freq_in,
    cufftComplex *freq_psf,
    cufftComplex *freq_out,
    int img_w_complex,
    int img_h,
    float K_val)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_w_complex && y < img_h) {
        int idx = y * img_w_complex + x;
        cufftComplex B = freq_in[idx];
        cufftComplex H = freq_psf[idx];
        float H_mag_sq = H.x * H.x + H.y * H.y;
        float denom = H_mag_sq + K_val;
        cufftComplex num;
        num.x = H.x * B.x + H.y * B.y;
        num.y = H.x * B.y - H.y * B.x;
        if (fabsf(denom) < 1e-9f) {
            freq_out[idx].x = 0.0f;
            freq_out[idx].y = 0.0f;
        } else {
            freq_out[idx].x = num.x / denom;
            freq_out[idx].y = num.y / denom;
        }
    }
}

__global__ void scale_kernel(float *data, int N, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= scale_factor;
    }
}

// --- Function Prototypes for CUDA Operations ---
static int wiener_deconv_channel_cuda(
    float *d_channel_spatial_managed, // Managed device pointer for spatial in/out
    float *d_psf_spatial_managed,     // Managed device pointer to PSF (float)
    int img_w,
    int img_h,
    float K_val);

static int perform_deconvolution_for_sigma_cuda(
    double sigma,
    const char* input_dir,
    const char* output_dir,
    const char* base_name,
    double K_param,
    int task_idx);


// ======================== MAIN ========================
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_dir> <output_dir> <base_name> [k_value]\n", argv[0]);
        return 1;
    }

    char input_dir[FILENAME_MAX];
    char output_dir[FILENAME_MAX];
    char base_name_str[FILENAME_MAX];
    double K = 0.001;

    strncpy(input_dir, argv[1], FILENAME_MAX - 1);
    strncpy(output_dir, argv[2], FILENAME_MAX - 1);
    strncpy(base_name_str, argv[3], FILENAME_MAX - 1);
    input_dir[FILENAME_MAX - 1] = '\0';
    output_dir[FILENAME_MAX - 1] = '\0';
    base_name_str[FILENAME_MAX - 1] = '\0';

    if (argc > 4) {
        K = atof(argv[4]);
        if (K < 0) {
            fprintf(stderr, "Warning: K value cannot be negative. Using default K = %f\n", 0.001);
            K = 0.001;
        }
    }
    printf("CUDA Wiener Deconvolution: Input='%s', Output='%s', Base='%s', K=%.4f\n",
           input_dir, output_dir, base_name_str, K);

    if (!create_directory(output_dir)) {
         fprintf(stderr, "Failed to create or access output directory '%s'. Aborting.\n", output_dir);
         return 1;
    }

    DIR *dir_handle;
    struct dirent *entry;
    double *sigma_tasks = NULL;
    int task_count = 0;
    int task_capacity = 0;
    char pattern_blur[FILENAME_MAX];

    snprintf(pattern_blur, FILENAME_MAX, "%s_blurred_sigma%%lf.png", base_name_str);

    printf("Scanning input directory '%s' for tasks...\n", input_dir);
    dir_handle = opendir(input_dir);
    if (!dir_handle) {
        perror("Error opening input directory");
        return 1;
    }

    while ((entry = readdir(dir_handle)) != NULL) {
        double current_sigma;
        if (sscanf(entry->d_name, pattern_blur, &current_sigma) == 1) {
            char psf_filename[FILENAME_MAX];
            char psf_filepath[FILENAME_MAX];
            snprintf(psf_filename, FILENAME_MAX, "%s_psf_sigma%.1f.png", base_name_str, current_sigma);
            snprintf(psf_filepath, FILENAME_MAX, "%s/%s", input_dir, psf_filename);
            struct stat st_psf;
            if (stat(psf_filepath, &st_psf) == 0 && S_ISREG(st_psf.st_mode)) {
                int found = 0;
                for(int i = 0; i < task_count; ++i) {
                     if (fabs(sigma_tasks[i] - current_sigma) < 1e-5) {
                          found = 1;
                          break;
                     }
                }
                if (!found) {
                     if (task_count >= task_capacity) {
                          task_capacity = (task_capacity == 0) ? 10 : task_capacity * 2;
                          double *new_tasks = (double*)realloc(sigma_tasks, task_capacity * sizeof(double));
                          if (!new_tasks) {
                               fprintf(stderr, "Failed to allocate memory for tasks.\n");
                               free(sigma_tasks); closedir(dir_handle); return 1;
                          }
                          sigma_tasks = new_tasks;
                     }
                     sigma_tasks[task_count++] = current_sigma;
                     printf("Found valid task pair for sigma %.1f\n", current_sigma);
                }
            }
        }
    }
    closedir(dir_handle);
    printf("Found %d tasks.\n", task_count);

    if (task_count == 0) {
         printf("No valid blurred/PSF pairs found. Exiting.\n");
         free(sigma_tasks);
         return 0;
    }

    struct timeval start_time, end_time;
    double elapsed_time;

    gettimeofday(&start_time, NULL); // Start timing for all tasks

    int tasks_succeeded = 0;
    for (int i = 0; i < task_count; ++i) {
        printf("\nProcessing task %d/%d: sigma %.1f\n", i + 1, task_count, sigma_tasks[i]);
        if (perform_deconvolution_for_sigma_cuda(sigma_tasks[i], input_dir, output_dir, base_name_str, K, i)) {
            tasks_succeeded++;
            printf("Successfully processed sigma %.1f\n", sigma_tasks[i]);
        } else {
            fprintf(stderr, "Failed to process sigma %.1f\n", sigma_tasks[i]);
        }
    }

    gettimeofday(&end_time, NULL); // End timing
    elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                   (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    printf("\n--------------------------------------------------------\n");
    printf(" CUDA Wiener Deconvolution Batch Processing Summary \n");
    printf("--------------------------------------------------------\n");
    printf(" Total tasks processed:   %8d\n", task_count);
    printf(" Tasks succeeded:         %8d\n", tasks_succeeded);
    printf(" K value used:            %10.4f\n", K);
    printf(" Total GPU processing time: %.6f seconds\n", elapsed_time);
    printf("--------------------------------------------------------\n\n");

    free(sigma_tasks);
    cudaDeviceReset();
    return 0;
}

// ======================== Deconvolution Task Function (CUDA Orchestrator) ========================
static int perform_deconvolution_for_sigma_cuda(
    double sigma,
    const char* input_dir,
    const char* output_dir,
    const char* base_name,
    double K_param,
    int task_idx)
{
    char blurred_filename[FILENAME_MAX];
    char psf_filename[FILENAME_MAX];
    char output_filename[FILENAME_MAX];
    char blurred_filepath[FILENAME_MAX];
    char psf_filepath[FILENAME_MAX];
    char output_filepath[FILENAME_MAX];

    snprintf(blurred_filename, FILENAME_MAX, "%s_blurred_sigma%.1f.png", base_name, sigma);
    snprintf(psf_filename, FILENAME_MAX, "%s_psf_sigma%.1f.png", base_name, sigma);
    snprintf(output_filename, FILENAME_MAX, "%s_deblurred_sigma%.1f_cuda.png", base_name, sigma); // Suffix to distinguish output

    snprintf(blurred_filepath, FILENAME_MAX, "%s/%s", input_dir, blurred_filename);
    snprintf(psf_filepath, FILENAME_MAX, "%s/%s", input_dir, psf_filename);
    snprintf(output_filepath, FILENAME_MAX, "%s/%s", output_dir, output_filename);

    int img_w = 0, img_h = 0;
    int psf_w = 0, psf_h = 0;
    float *h_img_in_rgb = read_image_rgb(blurred_filepath, &img_w, &img_h);
    if (!h_img_in_rgb) {
        fprintf(stderr, "Task %d (sigma %.1f): Failed to load blurred image '%s'.\n", task_idx, sigma, blurred_filepath);
        return 0;
    }
    float *h_psf_gray = read_image_gray(psf_filepath, &psf_w, &psf_h);
    if (!h_psf_gray) {
        fprintf(stderr, "Task %d (sigma %.1f): Failed to load PSF image '%s'.\n", task_idx, sigma, psf_filepath);
        free(h_img_in_rgb); return 0;
    }
    printf("Task %d (sigma %.1f): Loaded images (%dx%d), PSF (%dx%d)\n", task_idx, sigma, img_w, img_h, psf_w, psf_h);

    if (img_w <= 0 || img_h <= 0 || psf_w != img_w || psf_h != img_h) {
         fprintf(stderr, "Task %d (sigma %.1f): Invalid/mismatched image/PSF dimensions.\n", task_idx, sigma);
         free(h_img_in_rgb); free(h_psf_gray); return 0;
    }

    center_psf(h_psf_gray, psf_w, psf_h);
    normalize_psf(h_psf_gray, psf_w, psf_h);

    int N_spatial = img_w * img_h;
    float *h_img_out_rgb = (float *)malloc(sizeof(float) * N_spatial * 3);
    float *h_channel_r_in = (float *)malloc(sizeof(float) * N_spatial);
    float *h_channel_g_in = (float *)malloc(sizeof(float) * N_spatial);
    float *h_channel_b_in = (float *)malloc(sizeof(float) * N_spatial);
    float *h_channel_r_out = (float *)malloc(sizeof(float) * N_spatial);
    float *h_channel_g_out = (float *)malloc(sizeof(float) * N_spatial);
    float *h_channel_b_out = (float *)malloc(sizeof(float) * N_spatial);

    if (!h_img_out_rgb || !h_channel_r_in || !h_channel_g_in || !h_channel_b_in ||
        !h_channel_r_out || !h_channel_g_out || !h_channel_b_out) {
        fprintf(stderr, "Task %d (sigma %.1f): Failed to allocate host memory for channels.\n", task_idx, sigma);
        free(h_img_in_rgb); free(h_psf_gray); free(h_img_out_rgb);
        free(h_channel_r_in); free(h_channel_g_in); free(h_channel_b_in);
        free(h_channel_r_out); free(h_channel_g_out); free(h_channel_b_out);
        return 0;
    }

    for (int i = 0; i < N_spatial; ++i) {
        h_channel_r_in[i] = h_img_in_rgb[i * 3 + 0];
        h_channel_g_in[i] = h_img_in_rgb[i * 3 + 1];
        h_channel_b_in[i] = h_img_in_rgb[i * 3 + 2];
    }
    free(h_img_in_rgb);

    // --- Allocate GPU Managed Memory ---
    float *d_channel_data_managed, *d_psf_managed;
    CUDA_CHECK(cudaMallocManaged((void **)&d_channel_data_managed, N_spatial * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged((void **)&d_psf_managed, N_spatial * sizeof(float)));

    memcpy(d_psf_managed, h_psf_gray, N_spatial * sizeof(float));
    CUDA_CHECK(cudaDeviceSynchronize());
    free(h_psf_gray);

    float K_float = (float)K_param;
    int success_r = 0, success_g = 0, success_b = 0;

    // --- Process Red Channel ---
    printf("Task %d (sigma %.1f): Processing R channel...\n", task_idx, sigma);
    memcpy(d_channel_data_managed, h_channel_r_in, N_spatial * sizeof(float));
    CUDA_CHECK(cudaDeviceSynchronize());
    success_r = wiener_deconv_channel_cuda(d_channel_data_managed, d_psf_managed, img_w, img_h, K_float);
    if (success_r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        memcpy(h_channel_r_out, d_channel_data_managed, N_spatial * sizeof(float));
    }

    // --- Process Green Channel ---
    if (success_r) {
        printf("Task %d (sigma %.1f): Processing G channel...\n", task_idx, sigma);
        memcpy(d_channel_data_managed, h_channel_g_in, N_spatial * sizeof(float));
        CUDA_CHECK(cudaDeviceSynchronize());
        success_g = wiener_deconv_channel_cuda(d_channel_data_managed, d_psf_managed, img_w, img_h, K_float);
        if (success_g) {
            CUDA_CHECK(cudaDeviceSynchronize());
            memcpy(h_channel_g_out, d_channel_data_managed, N_spatial * sizeof(float));
        }
    }

    // --- Process Blue Channel ---
    if (success_g) {
        printf("Task %d (sigma %.1f): Processing B channel...\n", task_idx, sigma);
        memcpy(d_channel_data_managed, h_channel_b_in, N_spatial * sizeof(float));
        CUDA_CHECK(cudaDeviceSynchronize());
        success_b = wiener_deconv_channel_cuda(d_channel_data_managed, d_psf_managed, img_w, img_h, K_float);
        if (success_b) {
            CUDA_CHECK(cudaDeviceSynchronize());
            memcpy(h_channel_b_out, d_channel_data_managed, N_spatial * sizeof(float));
        }
    }

    CUDA_CHECK(cudaFree(d_channel_data_managed));
    CUDA_CHECK(cudaFree(d_psf_managed));

    free(h_channel_r_in); free(h_channel_g_in); free(h_channel_b_in);

    if (!success_r || !success_g || !success_b) {
        fprintf(stderr, "Task %d (sigma %.1f): Deconvolution failed for one or more channels.\n", task_idx, sigma);
        free(h_img_out_rgb);
        free(h_channel_r_out); free(h_channel_g_out); free(h_channel_b_out);
        return 0;
    }

    for (int i = 0; i < N_spatial; ++i) {
        h_img_out_rgb[i * 3 + 0] = h_channel_r_out[i];
        h_img_out_rgb[i * 3 + 1] = h_channel_g_out[i];
        h_img_out_rgb[i * 3 + 2] = h_channel_b_out[i];
    }

    printf("Task %d (sigma %.1f): Saving deblurred image to '%s'\n", task_idx, sigma, output_filepath);
    if (!write_image_rgb(output_filepath, h_img_out_rgb, img_w, img_h)) {
        fprintf(stderr, "Task %d (sigma %.1f): Failed to write output image '%s'.\n", task_idx, sigma, output_filepath);
        free(h_img_out_rgb);
        free(h_channel_r_out); free(h_channel_g_out); free(h_channel_b_out);
        return 0;
    }

    free(h_img_out_rgb);
    free(h_channel_r_out); free(h_channel_g_out); free(h_channel_b_out);
    return 1;
}


// ======================== CUDA Wiener Deconvolution for a Single Channel ========================
static int wiener_deconv_channel_cuda(
    float *d_channel_spatial_managed, // Managed memory for spatial input & output
    float *d_psf_spatial_managed,     // Managed memory for PSF
    int img_w,
    int img_h,
    float K_val)
{
    int N_spatial = img_w * img_h;
    int img_w_complex = img_w / 2 + 1;
    int N_complex = img_w_complex * img_h;

    cufftComplex *d_freq_in_managed, *d_freq_psf_managed, *d_freq_out_managed;
    CUDA_CHECK(cudaMallocManaged((void **)&d_freq_in_managed, N_complex * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocManaged((void **)&d_freq_psf_managed, N_complex * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocManaged((void **)&d_freq_out_managed, N_complex * sizeof(cufftComplex)));

    cufftHandle plan_r2c_in, plan_r2c_psf, plan_c2r_out;
    CUFFT_CHECK(cufftCreate(&plan_r2c_in));
    CUFFT_CHECK(cufftPlan2d(&plan_r2c_in, img_h, img_w, CUFFT_R2C));
    CUFFT_CHECK(cufftCreate(&plan_r2c_psf));
    CUFFT_CHECK(cufftPlan2d(&plan_r2c_psf, img_h, img_w, CUFFT_R2C));
    CUFFT_CHECK(cufftCreate(&plan_c2r_out));
    CUFFT_CHECK(cufftPlan2d(&plan_c2r_out, img_h, img_w, CUFFT_C2R));

    CUFFT_CHECK(cufftExecR2C(plan_r2c_in, d_channel_spatial_managed, d_freq_in_managed));
    CUFFT_CHECK(cufftExecR2C(plan_r2c_psf, d_psf_spatial_managed, d_freq_psf_managed));

    dim3 threadsPerBlock2D(THREADS_PER_BLOCK_2D_X, THREADS_PER_BLOCK_2D_Y);
    dim3 numBlocks2D(
        (img_w_complex + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
        (img_h + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
    wiener_filter_kernel<<<numBlocks2D, threadsPerBlock2D>>>(
        d_freq_in_managed, d_freq_psf_managed, d_freq_out_managed, img_w_complex, img_h, K_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUFFT_CHECK(cufftExecC2R(plan_c2r_out, d_freq_out_managed, d_channel_spatial_managed));

    float scale_factor = 1.0f / (float)N_spatial;
    dim3 threadsPerBlock1D(THREADS_PER_BLOCK_1D);
    dim3 numBlocks1D((N_spatial + threadsPerBlock1D.x - 1) / threadsPerBlock1D.x);
    scale_kernel<<<numBlocks1D, threadsPerBlock1D>>>(d_channel_spatial_managed, N_spatial, scale_factor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUFFT_CHECK(cufftDestroy(plan_r2c_in));
    CUFFT_CHECK(cufftDestroy(plan_r2c_psf));
    CUFFT_CHECK(cufftDestroy(plan_c2r_out));

    CUDA_CHECK(cudaFree(d_freq_in_managed));
    CUDA_CHECK(cudaFree(d_freq_psf_managed));
    CUDA_CHECK(cudaFree(d_freq_out_managed));

    return 1; // Success
}

// ======================== Image/CPU Utility Functions (unchanged) ========================
static float *read_image_rgb(const char *path, int *out_w, int *out_h) {
    int channels_in_file;
    unsigned char *data_u8 = stbi_load(path, out_w, out_h, &channels_in_file, 3);
    if (!data_u8) { return NULL; }
    int w = *out_w; int h = *out_h;
    float *data_f = (float *)malloc(sizeof(float) * w * h * 3);
    if (!data_f) { stbi_image_free(data_u8); return NULL; }
    for (int i = 0; i < w * h * 3; i++) data_f[i] = (float)(data_u8[i]) / 255.0f;
    stbi_image_free(data_u8); return data_f;
}

static float *read_image_gray(const char *path, int *out_w, int *out_h) {
    int channels_in_file;
    unsigned char *data_u8 = stbi_load(path, out_w, out_h, &channels_in_file, 1);
    if (!data_u8) { return NULL; }
    int w = *out_w; int h = *out_h;
    float *data_f = (float *)malloc(sizeof(float) * w * h);
    if (!data_f) { stbi_image_free(data_u8); return NULL; }
    for (int i = 0; i < w * h; i++) data_f[i] = (float)(data_u8[i]) / 255.0f;
    stbi_image_free(data_u8); return data_f;
}

static int write_image_rgb(const char *path, const float *data, int w, int h) {
    unsigned char *out_u8 = (unsigned char *)malloc(sizeof(unsigned char) * w * h * 3);
    if (!out_u8) { return 0; }
    for (int i = 0; i < w * h * 3; i++) {
        float val = data[i];
        if (val < 0.0f) val = 0.0f; if (val > 1.0f) val = 1.0f;
        out_u8[i] = (unsigned char)(val * 255.0f + 0.5f);
    }
    int ret = stbi_write_png(path, w, h, 3, out_u8, w * 3);
    free(out_u8); return ret;
}

static void center_psf(float *psf, int w, int h) {
    float *temp = (float *)malloc(w * h * sizeof(float));
    if (!temp) { fprintf(stderr, "Error: Out of memory in center_psf.\n"); return; }
    memcpy(temp, psf, w * h * sizeof(float));
    int half_w = w / 2; int half_h = h / 2;
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            psf[((r + half_h) % h) * w + ((c + half_w) % w)] = temp[r * w + c];
        }
    }
    free(temp);
}

static void normalize_psf(float *psf, int w, int h) {
    double sum = 0.0;
    for (int i = 0; i < w * h; i++) sum += psf[i];
    if (fabs(sum) > 1e-10) {
        for (int i = 0; i < w * h; i++) psf[i] /= (float)sum;
    } else {
        fprintf(stderr, "Warning: PSF sum is close to zero (%e). Normalization skipped.\n", sum);
    }
}

int create_directory(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        if (mkdir(path, 0755) != 0) {
             if (errno != EEXIST) { perror("Error creating directory"); return 0; }
        }
    } else if (!S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Error: %s exists but is not a directory\n", path); return 0;
    }
    return 1;
}