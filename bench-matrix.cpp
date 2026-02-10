#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // For gettimeofday
#include <pthread.h>   // For pthreads
#include <unistd.h>

// Structure to represent a matrix
typedef struct {
    double* data; // Pointer to the continuous block of memory
    int nrows;
    int ncols;
} Matrix;

// Function to create a matrix with a continuous block of memory
Matrix create_matrix(int rows, int cols) {
    Matrix matrix;
    matrix.nrows = rows;
    matrix.ncols = cols;
    matrix.data = (double*)malloc(rows * cols * sizeof(double));
    if (matrix.data == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix.\n");
        exit(1);
    }
    return matrix;
}

// Function to free the matrix memory
void free_matrix(Matrix matrix) {
    free(matrix.data);
}

// Function to generate a random double matrix
Matrix generate_random_matrix(int size) {
    Matrix matrix = create_matrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Generate a random double between 0.0 and 1.0
            matrix.data[i * size + j] = (double)rand() / RAND_MAX;
        }
    }
    return matrix;
}

// Function to multiply two matrices
__attribute__((noinline))
Matrix multiply_matrices(Matrix result_matrix, Matrix matrix1, Matrix matrix2) {
    
    //printf("Multiply %d %d\n", matrix1.ncols,matrix1.nrows);

    if (matrix1.ncols != matrix2.nrows) {
        fprintf(stderr, "Matrix dimensions are incompatible for multiplication.\n");
        exit(1);
    }


    for (int i = 0; i < matrix1.nrows; i++) {
        for (int j = 0; j < matrix2.ncols; j++) {
            double sum = 0.0;
            for (int k = 0; k < matrix1.ncols; k++) {
                sum += matrix1.data[i * matrix1.ncols + k] * matrix2.data[k * matrix2.ncols + j];
            }
            result_matrix.data[i * matrix2.ncols + j] = sum;
        }
    }

    return result_matrix;
}

// Function to print a matrix (for demonstration)
void print_matrix(Matrix matrix) {
    for (int i = 0; i < matrix.nrows; i++) {
        for (int j = 0; j < matrix.ncols; j++) {
            printf("%.2f ", matrix.data[i * matrix.ncols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to benchmark matrix multiplication (single-threaded)
void benchmark_1cpu(int matrix_size, int num_multiplications) {
    // Generate two random matrices
    Matrix matrix1 = generate_random_matrix(matrix_size);
    Matrix matrix2 = generate_random_matrix(matrix_size);

    // Start time
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    Matrix result_matrix = create_matrix(matrix1.nrows, matrix2.ncols);

    // Perform multiple matrix multiplications
    for (int i = 0; i < num_multiplications; i++) {
        Matrix r = multiply_matrices(result_matrix, matrix1, matrix2);
    }

    // End time
    gettimeofday(&end_time, NULL);

    // Calculate elapsed time in milliseconds
    long elapsed_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000 +
                           (end_time.tv_usec - start_time.tv_usec) / 1000;

    // Calculate multiplications per millisecond
    double multiplications_per_ms = (double)num_multiplications / elapsed_time_ms;

    // Print the results
    printf("Benchmark Results (1 CPU):\n");
    printf("  Matrix Size: %dx%d\n", matrix_size, matrix_size);
    printf("  Number of Multiplications: %d\n", num_multiplications);
    printf("  Elapsed Time: %ld ms\n", elapsed_time_ms);
    printf("  Multiplications per Millisecond: %.2f\n", multiplications_per_ms);

    // Free the memory
    free_matrix(matrix1);
    free_matrix(matrix2);
}


// Structure to pass data to each thread
typedef struct {
    Matrix matrix1;
    Matrix matrix2;
    int num_multiplications;
    long* elapsed_time_ms; // Pointer to store elapsed time
} ThreadData;


int matrix_size_global = 1000;



// Function to be executed by each thread
void* thread_function(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    //printf("Thread starting num=%d\n", data->num_multiplications);

    // Start time
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    Matrix matrix1 = generate_random_matrix(matrix_size_global);
    Matrix matrix2 = generate_random_matrix(matrix_size_global);
    Matrix result_matrix = create_matrix(matrix1.nrows, matrix2.ncols);

    //printf("thread nummp=%d size=%d\n", data->num_multiplications, matrix_size_global);

    // Perform multiple matrix multiplications
    for (int i = 0; i < data->num_multiplications; i++) {
        Matrix r = multiply_matrices(result_matrix, matrix1, matrix2);
       // free_matrix(result_matrix); // Free memory after each multiplication
    }

    //printf("thread done\n");

    // End time
    gettimeofday(&end_time, NULL);

    pthread_exit(NULL);
}


// Function to benchmark matrix multiplication (multi-threaded)
void benchmark_n_cpu(int matrix_size, int num_multiplications, int num_threads, int quiet = 0) {
    matrix_size_global = matrix_size;
    // Generate two random matrices (shared by all threads)
    Matrix matrix1 = generate_random_matrix(matrix_size);
    Matrix matrix2 = generate_random_matrix(matrix_size);

    // Create thread data
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    pthread_attr_t attr;

    // Initialize thread attributes
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].matrix1 = matrix1;
        thread_data[i].matrix2 = matrix2;
        thread_data[i].num_multiplications = num_multiplications; // Divide work
        pthread_create(&threads[i], &attr, thread_function, &thread_data[i]);
    }

    // Destroy the attribute object
    pthread_attr_destroy(&attr);

    // Start time
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // Wait for threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // End time
    gettimeofday(&end_time, NULL);

    // Calculate elapsed time in milliseconds
    long elapsed_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000 +
                           (end_time.tv_usec - start_time.tv_usec) / 1000;

    // Calculate multiplications per millisecond
    double multiplications_per_ms = (double)num_multiplications * num_threads / elapsed_time_ms;

    // Print the results
    if(! quiet) {
        printf(" %3d CPUs: ", num_threads);
        printf(" %dx%d ", matrix_size, matrix_size);
        printf("count:%d ", num_multiplications);
        printf("    %.3f mult/s\n", multiplications_per_ms*1000.0);
    }

    // Free the memory
    free_matrix(matrix1);
    free_matrix(matrix2);
}


int main(int argc, char *argv[]) {
    // Seed the random number generator
    srand(time(NULL));

    int matrix_size = 100; // Example size, you can change this
    int num_multiplications = 500;
    int num_threads = 1; // Example number of threads

    //num_threads = atoi(argv[1]);

    // Run the benchmark

    // Warm up
    for(int i=0 ; i < 11; i++)
        benchmark_n_cpu(matrix_size, num_multiplications, 2, 1);


    for(int num_threads = 1; num_threads < 30; num_threads++) {
    benchmark_n_cpu(matrix_size, num_multiplications, num_threads);
    usleep(10000);
    }

    return 0;
}
