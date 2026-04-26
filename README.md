# sparse-matvec-c
Implementation of sparse matrix-vector multiplication in C using the **Compressed Sparse Row (CSR)** format.  
Given a dense matrix A and a vector x, the function computes **y = A · x** by first extracting only the non-zero elements of A into CSR format, then performing the multiplication exclusively on those elements.
 
**Critical constraint:** the function performs zero dynamic memory allocation. All buffers are pre-allocated by the caller and passed as parameters.
 
---
 
## Why CSR?
 
A dense matrix-vector product touches every element of A, including zeros, which contribute nothing to the result. For sparse matrices (matrices where most elements are zero), this is wasteful.
 
CSR eliminates this waste by storing only the non-zero elements. The gain becomes significant as sparsity increases:
 
| Matrix size | Density | Elements in A | Non-zeros touched | Speedup |
|---|---|---|---|---|
| 100 × 100 | 10% | 10,000 | 1,000 | ~10x |
| 100 × 100 | 5% | 10,000 | 500 | ~20x |
| 1000 × 1000 | 1% | 1,000,000 | 10,000 | ~100x |
 
---
 
## CSR Format
 
The CSR format represents a sparse matrix using three arrays:
 
- **`values[]`** — the non-zero elements of A, stored row by row
- **`col_indices[]`** — the column index of each element in `values[]`
- **`row_ptrs[]`** — `row_ptrs[i]` is the index in `values[]` where row `i` begins. Length is `rows + 1`, where the last element acts as a sentinel marking the end of the last row.
### Example
 
```
A = [ 1  0  2 ]
    [ 0  3  0 ]
    [ 4  0  5 ]
```
 
```
values      = [ 1,  2,  3,  4,  5 ]
col_indices = [ 0,  2,  1,  0,  2 ]
row_ptrs    = [ 0,  2,  3,  5 ]
```
 
Reading `row_ptrs`:
- Row 0 spans `values[0..1]` → elements 1 (col 0) and 2 (col 2)
- Row 1 spans `values[2..2]` → element 3 (col 1)
- Row 2 spans `values[3..4]` → elements 4 (col 0) and 5 (col 2)
The length of row `i` is always `row_ptrs[i+1] - row_ptrs[i]`.
 
---
 
## Function Signature
 
```c
void sparse_multiply(
    int rows,            // number of rows in matrix A
    int cols,            // number of columns in matrix A
    const double* A,     // dense input matrix, row-major order: element (i,j) = A[i * cols + j]
    const double* x,     // input vector, length = cols
    int* out_nnz,        // output: total number of non-zero elements found in A (returned via pointer)
    double* values,      // output buffer: non-zero values of A stored sequentially
    int* col_indices,    // output buffer: column index of each non-zero value
    int* row_ptrs,       // output buffer: row_ptrs[i] = index in values where row i starts
    double* y            // output vector: result of y = A * x, length = rows
);
```
 
### Caller responsibilities
 
The caller must pre-allocate all output buffers before calling the function:
 
| Buffer | Type | Minimum size |
|---|---|---|
| `values` | `double*` | `rows × cols` |
| `col_indices` | `int*` | `rows × cols` |
| `row_ptrs` | `int*` | `rows + 1` |
| `y` | `double*` | `rows` |
 
The worst-case size for `values` and `col_indices` is `rows × cols` (fully dense matrix). Allocating this size is always safe regardless of actual sparsity.
 
---
 
## Algorithm
 
The function operates in two sequential passes:
 
**Pass 1 — CSR extraction** `O(rows × cols)`  
Iterates over A in row-major order. At the start of each row `i`, records `row_ptrs[i] = nz` (current non-zero count). For each non-zero element, writes the value to `values[nz]` and the column index to `col_indices[nz]`, then increments `nz`. After all rows, writes the sentinel `row_ptrs[rows] = nz` and stores the final count in `*out_nnz`.
 
**Pass 2 — Sparse multiply** `O(nnz)`  
For each row `i`, iterates over the slice of `values[]` from `row_ptrs[i]` to `row_ptrs[i+1]`. For each non-zero element at position `k`, adds `values[k] * x[col_indices[k]]` to the accumulator. Writes the result to `y[i]`.
 
 
## Build & Run
 
```bash
gcc -o run challenge.c -lm
./run
```
 
---
 
## Test Harness
 
The included test harness runs **100 randomized iterations**. Each iteration:
 
1. Generates a random matrix A with dimensions between 5×5 and 45×45 and density between 5% and 40%
2. Generates a random input vector x
3. Computes the reference result using a standard dense multiply
4. Calls `sparse_multiply()` and compares the output against the reference
5. Validates each element with mixed absolute/relative tolerance of `1e-7`
```bash
Iter  0 [ 23x 18, density=0.28, nnz=  97]: PASS (Max error: 0.00e+00)
Iter  1 [ 38x 41, density=0.12, nnz= 178]: PASS (Max error: 0.00e+00)
...
All tests passed! (100/100 iterations passed)
```
 
---
 
## File Structure
 
```
sparse-multiply/
├── challenge.c    # function implementation and test harness
└── README.md      # this file
```
