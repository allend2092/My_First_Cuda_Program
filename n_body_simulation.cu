#include <iostream>
#include <cuda_runtime.h>

const int N = 1000;  // Number of particles
const float G = 1.0f;  // Gravitational constant
const float dt = 0.01f;  // Time step

__global__ void update(float2 *positions, float2 *velocities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float2 pos_i = positions[i];
        float2 acc = {0.0f, 0.0f};
        for (int j = 0; j < N; j++) {
            if (i != j) {
                float2 pos_j = positions[j];
                float dx = pos_j.x - pos_i.x;
                float dy = pos_j.y - pos_i.y;
                float distance = sqrtf(dx*dx + dy*dy);
                float force = G / (distance + 1e-9f) / (distance + 1e-9f);
                acc.x += force * dx / (distance + 1e-9f);
                acc.y += force * dy / (distance + 1e-9f);
            }
        }
        velocities[i].x += acc.x * dt;
        velocities[i].y += acc.y * dt;
        positions[i].x += velocities[i].x * dt;
        positions[i].y += velocities[i].y * dt;
    }
}

int main() {
    float2 *positions, *velocities;
    float2 *d_positions, *d_velocities;

    // Allocate host memory
    positions = new float2[N];
    velocities = new float2[N];

    // Initialize positions and velocities
    for (int i = 0; i < N; i++) {
        positions[i] = {rand() / (float)RAND_MAX, rand() / (float)RAND_MAX};
        velocities[i] = {rand() / (float)RAND_MAX, rand() / (float)RAND_MAX};
    }

    // Allocate device memory
    cudaMalloc((void**)&d_positions, N * sizeof(float2));
    cudaMalloc((void**)&d_velocities, N * sizeof(float2));

    // Copy data to device
    cudaMemcpy(d_positions, positions, N * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities, N * sizeof(float2), cudaMemcpyHostToDevice);

    // Launch update kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < 100; i++) {
        update<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_velocities);
        cudaDeviceSynchronize();
    }

    // Cleanup
    delete[] positions;
    delete[] velocities;
    cudaFree(d_positions);
    cudaFree(d_velocities);

    return 0;
}
