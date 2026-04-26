#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =========================================================
// FUNCTION PROTOTYPE
// =========================================================
void sparse__multiply(
    int rows,               // number of rows in matrix A
    int cols,               // number of columns in matrix A
    const double* A,        // dense input matrix, row-major layout: element (i,j) = A[i * cols + j]
    const double* x,        // input vector to multiply, length = cols.
    int* out_nnz,           // output: total number of non-zero elements found in A
    double* values,         // output buffer: non-zero values of A stored sequentially (size >= rows*cols)
    int* col_indices,       // output buffer: column index of each non-zero value   (size >= rows*cols)
    int* row_ptrs,          // output buffer: row_ptrs[i] = index in values where row i starts (size >= rows+1)
    double* y               // output vector: result of y = A * x, length = rows
);

// =========================================================
// TODO: USER IMPLEMENTATION
// =========================================================
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    double val;   // current element at row i, column j of the dense matrix A

    //Build CSR from the dense matrix A given in input:
    int nz_value = 0;     //Non-zero values counter.
    for (int i = 0; i < rows; i++) {
        row_ptrs[i] = nz_value;
        for (int j = 0; j < cols; j++) {
            val = A[i * cols + j];      // A is stored as a 1D array in row-major order: element (i,j) = A[i * cols + j]
            if (val != 0.0) {
                values[nz_value] = val;
                col_indices[nz_value] = j;
                nz_value++;
            }
        }
    }
    row_ptrs[rows] = nz_value;   //Last value of row_ptrs to count the total number of non-zero values
    *out_nnz = nz_value;        //Total number of non-zero elements in A (returned via pointer)

    //Compute y = A * x using CSR
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; k++) {       // iterate over non-zero elements of row i
            sum += values[k] * x[col_indices[k]];       // multiply non-zero value by the corresponding x component
        }
        y[i] = sum;     // write dot product result to output vector
    }
}
// =========================================================
// TEST HARNESS
// =========================================================
int main(void) {
    srand(time(NULL));
    
    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double) RAND_MAX) * 0.35;
        
        size_t mat_sz = (size_t) rows * cols;

        double* A = calloc(mat_sz, sizeof(double));
        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double) rand() / RAND_MAX) < density) {
                A[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        double* values = malloc(mat_sz * sizeof(double));
        int* col_indices = malloc(mat_sz * sizeof(int));
        int* row_ptrs = malloc((rows + 1) * sizeof(int));
        double* x = malloc(cols * sizeof(double));
        double* y_user = malloc(rows * sizeof(double));
        double* y_ref = calloc(rows, sizeof(double));
        int out_nnz = 0;

        for (int i = 0; i < cols; ++i) {
            x[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
        }

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                sum += A[i * cols + j] * x[j];
            }
            y_ref[i] = sum;
        }

        sparse_multiply(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y_user);

        double max_err = 0.0;
        int passed = 1;
        for (int i = 0; i < rows; ++i) {
            double diff = fabs(y_user[i] - y_ref[i]);
            double tol = 1e-7 + 1e-7 * fabs(y_ref[i]); // Mixed absolute/relative tolerance
            if (diff > tol) {
                max_err = fmax(max_err, diff);
                passed = 0;
            }
        }

        if (passed) {
            passed_count++;
        }

        printf(
            "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: %s (Max error: %.2e)\n",
            iter, rows, cols, density, out_nnz, passed ? "PASS" : "FAIL", max_err
        );

        free(A);
        free(values);
        free(col_indices);
        free(row_ptrs);
        free(x);
        free(y_user);
        free(y_ref);
    }

    printf(
        "\n%s (%d/%d iterations passed)\n",
        passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
        passed_count, num_iterations
    );
           
    return passed_count == num_iterations ? 0 : 1;
}
