// Include necessary headers
#include <iostream>
#include <cuda_runtime.h>

// Define constant for array size
const int N = 512;

// GPU kernel function to add two arrays
__global__ void add(int *a, int *b, int *c) {
    // Get the current block's index
    int tid = blockIdx.x;
    
    // Ensure we don't go out of bounds
    if (tid < N)
        c[tid] = a[tid] + b[tid];  // Add corresponding elements of arrays a and b
}

int main() {
    // Declare arrays of size N
    int a[N], b[N], c[N];
    
    // Declare pointers for device memory
    int *dev_a, *dev_b, *dev_c;

    // Allocate memory on the GPU for the arrays
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // Initialize arrays a and b with values
    for (int i = 0; i < N; i++) {
        a[i] = -i;          // Initialize array a with negative values
        b[i] = i * i;       // Initialize array b with squares of indices
    }

    // Copy initialized arrays a and b from host to device memory
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the add kernel on the GPU with N blocks and 1 thread per block
    add<<<N,1>>>(dev_a, dev_b, dev_c);

    // Copy the result from device to host memory
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the results of the addition
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Free the allocated device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Return 0 indicating successful execution
    return 0;
}
