#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#define BLOCK_SIZE 64
#define HALF_BLOCK_FACTOR BLOCK_SIZE/2
using namespace std;
std::chrono::steady_clock::time_point total_start, total_end;

const int INF = ((1 << 30) - 1);
void input(char* infile);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
__global__ void Phase1(int *dist, int Round, int n);
__global__ void Phase2(int *dist, int Round, int n);
__global__ void Phase3(int *dist, int Round, int n);

int N, n, m;
int* Dist = NULL;

int main(int argc, char* argv[]) {
    total_start = std::chrono::steady_clock::now();
	input(argv[1]);
	block_FW(BLOCK_SIZE);
	output(argv[2]);
    cudaFreeHost(Dist);
    total_end = std::chrono::steady_clock::now();
    std::cout << "[TOTAL_TIME] " << std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() << "\n";
	return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&N, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);


    n = N + (BLOCK_SIZE - (N%BLOCK_SIZE));
    Dist = (int*) malloc(sizeof(int)*n*n);
    cout << "N: " << N << " | n: " << n << endl;
    for (int block_i = 0; block_i < n; ++ block_i) {
        int IN = block_i * n;
        #pragma GCC ivdep
        for (int j = 0; j < block_i; ++j) {
            Dist[IN + j] = INF;
        }
        #pragma GCC ivdep
        for (int j = block_i + 1; j < n; ++j) {
            Dist[IN + j] = INF;
        }
    }

    int pair[3];
    for (int block_i = 0; block_i < m; ++block_i) { 
        fread(pair, sizeof(int), 3, file); 
        Dist[pair[0] * n + pair[1]] = pair[2]; 
    } 
    fclose(file);
}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");
    for (int block_i = 0; block_i < N; ++block_i) {
        fwrite(&Dist[block_i * n], sizeof(int), N, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { 
    return (a + b - 1) / b; 
}

void block_FW(int B) {
    int* dst = NULL;


    cudaHostRegister(Dist, n*n*sizeof(int), cudaHostRegisterDefault);
    cudaMalloc(&dst, n*n*sizeof(int));
	cudaMemcpy(dst, Dist, n*n*sizeof(int), cudaMemcpyHostToDevice);

    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_dim(HALF_BLOCK_FACTOR, HALF_BLOCK_FACTOR, 1);
    dim3 grid_dim(blocks, blocks, 1);

    int round = ceil(n, B);
    
    for (int r = 0; r < round; ++r) {
        Phase1<<<1, block_dim>>>(dst, r, n);
        Phase2<<<blocks, block_dim>>>(dst, r, n);
        Phase3<<<grid_dim, block_dim>>>(dst, r, n);
    }
   
    cudaMemcpy(Dist, dst, n*n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dst);
}

__global__ void Phase1(int *dist, int Round, int n) {
    __shared__ int shared_memory[BLOCK_SIZE][BLOCK_SIZE];
    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int offset = BLOCK_SIZE * Round;

    shared_memory[i   ][j   ] = dist[offset*(n+1) + i*n + j];
    shared_memory[i+HALF_BLOCK_FACTOR][j   ] = dist[offset*(n+1) + (i+HALF_BLOCK_FACTOR)*n + j];
    shared_memory[i   ][j+HALF_BLOCK_FACTOR] = dist[offset*(n+1) + i*n + j + HALF_BLOCK_FACTOR];
    shared_memory[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = dist[offset*(n+1) + (i+HALF_BLOCK_FACTOR)*n + j + HALF_BLOCK_FACTOR];
    __syncthreads();

    // #pragma unroll 16
    for (int k = 0; k < BLOCK_SIZE; k++) {
        shared_memory[i   ][j   ] = min(shared_memory[i   ][j   ], shared_memory[i   ][k] + shared_memory[k][j   ]);
        shared_memory[i+HALF_BLOCK_FACTOR][j   ] = min(shared_memory[i+HALF_BLOCK_FACTOR][j   ], shared_memory[i+HALF_BLOCK_FACTOR][k] + shared_memory[k][j   ]);
        shared_memory[i   ][j+HALF_BLOCK_FACTOR] = min(shared_memory[i   ][j+HALF_BLOCK_FACTOR], shared_memory[i   ][k] + shared_memory[k][j+HALF_BLOCK_FACTOR]);
        shared_memory[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = min(shared_memory[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR], shared_memory[i+HALF_BLOCK_FACTOR][k] + shared_memory[k][j+HALF_BLOCK_FACTOR]);
        __syncthreads();
    }

    dist[offset*(n+1) + i*n + j]           = shared_memory[i   ][j   ];
    dist[offset*(n+1) + (i+HALF_BLOCK_FACTOR)*n + j]      = shared_memory[i+HALF_BLOCK_FACTOR][j   ];
    dist[offset*(n+1) + i*n + j + HALF_BLOCK_FACTOR]      = shared_memory[i   ][j+HALF_BLOCK_FACTOR];
    dist[offset*(n+1) + (i+HALF_BLOCK_FACTOR)*n + j + HALF_BLOCK_FACTOR] = shared_memory[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR];
}

__global__ void Phase2(int *dist, int Round, int n) {
    __shared__ int both[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int row_blocks[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int col_blocks[BLOCK_SIZE][BLOCK_SIZE];

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int offset = BLOCK_SIZE * Round;
    const int block_i = blockIdx.x;

    if (block_i == Round) 
        return;

    row_blocks[i   ][j   ] = dist[block_i*BLOCK_SIZE*n + offset + i*n + j];
    row_blocks[i+HALF_BLOCK_FACTOR][j   ] = dist[block_i*BLOCK_SIZE*n + offset + (i+HALF_BLOCK_FACTOR)*n + j];
    row_blocks[i   ][j+HALF_BLOCK_FACTOR] = dist[block_i*BLOCK_SIZE*n + offset + i*n + j + HALF_BLOCK_FACTOR];
    row_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = dist[block_i*BLOCK_SIZE*n + offset + (i+HALF_BLOCK_FACTOR)*n + j+HALF_BLOCK_FACTOR];

    col_blocks[i   ][j   ] = dist[offset*n + block_i*BLOCK_SIZE + i*n + j];
    col_blocks[i+HALF_BLOCK_FACTOR][j   ] = dist[offset*n + block_i*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j];
    col_blocks[i   ][j+HALF_BLOCK_FACTOR] = dist[offset*n + block_i*BLOCK_SIZE + i*n + j+HALF_BLOCK_FACTOR];
    col_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = dist[offset*n + block_i*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j+HALF_BLOCK_FACTOR];

    both[i   ][j   ] = dist[offset*(n+1) + i*n + j];
    both[i+HALF_BLOCK_FACTOR][j   ] = dist[offset*(n+1) + (i+HALF_BLOCK_FACTOR)*n + j];
    both[i   ][j+HALF_BLOCK_FACTOR] = dist[offset*(n+1) + i*n + j+HALF_BLOCK_FACTOR];
    both[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = dist[offset*(n+1) + (i+HALF_BLOCK_FACTOR)*n + j+HALF_BLOCK_FACTOR];
  
    __syncthreads();

    // #pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {

        row_blocks[i   ][j   ] = min(row_blocks[i][j], row_blocks[i][k] + both[k][j]);
        row_blocks[i+HALF_BLOCK_FACTOR][j   ] = min(row_blocks[i+HALF_BLOCK_FACTOR][j], row_blocks[i+HALF_BLOCK_FACTOR][k] + both[k][j]);
        row_blocks[i   ][j+HALF_BLOCK_FACTOR] = min(row_blocks[i][j+HALF_BLOCK_FACTOR], row_blocks[i][k] + both[k][j+HALF_BLOCK_FACTOR]);
        row_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = min(row_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR], row_blocks[i+HALF_BLOCK_FACTOR][k] + both[k][j+HALF_BLOCK_FACTOR]);

        col_blocks[i   ][j   ] = min(col_blocks[i][j], both[i][k] + col_blocks[k][j]);
        col_blocks[i+HALF_BLOCK_FACTOR][j   ] = min(col_blocks[i+HALF_BLOCK_FACTOR][j], both[i+HALF_BLOCK_FACTOR][k] + col_blocks[k][j]);
        col_blocks[i   ][j+HALF_BLOCK_FACTOR] = min(col_blocks[i][j+HALF_BLOCK_FACTOR], both[i][k] + col_blocks[k][j+HALF_BLOCK_FACTOR]);
        col_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = min(col_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR], both[i+HALF_BLOCK_FACTOR][k] + col_blocks[k][j+HALF_BLOCK_FACTOR]);
    }

    dist[block_i*BLOCK_SIZE*n + offset + i*n + j]         = row_blocks[i   ][j   ];
    dist[block_i*BLOCK_SIZE*n + offset + (i+HALF_BLOCK_FACTOR)*n + j]    = row_blocks[i+HALF_BLOCK_FACTOR][j   ];
    dist[block_i*BLOCK_SIZE*n + offset + i*n + j + HALF_BLOCK_FACTOR]    = row_blocks[i   ][j+HALF_BLOCK_FACTOR];
    dist[block_i*BLOCK_SIZE*n + offset + (i+HALF_BLOCK_FACTOR)*n + j+HALF_BLOCK_FACTOR] = row_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR];

    dist[offset*n + block_i*BLOCK_SIZE + i*n + j]         = col_blocks[i   ][j   ];
    dist[offset*n + block_i*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j]    = col_blocks[i+HALF_BLOCK_FACTOR][j   ];
    dist[offset*n + block_i*BLOCK_SIZE + i*n + j+HALF_BLOCK_FACTOR]      = col_blocks[i   ][j+HALF_BLOCK_FACTOR];
    dist[offset*n + block_i*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j+HALF_BLOCK_FACTOR] = col_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR];
}

__global__ void Phase3(int *dist, int Round, int n) {
    __shared__ int row_blocks[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int col_blocks[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_memory[BLOCK_SIZE][BLOCK_SIZE];
    
    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int block_i = blockIdx.y;
    const int block_j = blockIdx.x;
    const int offset = BLOCK_SIZE * Round;

    if (block_i == Round && block_j == Round) 
        return;

    shared_memory[i   ][j   ] = dist[block_i*BLOCK_SIZE*n + block_j*BLOCK_SIZE + i*n + j];
    shared_memory[i+HALF_BLOCK_FACTOR][j   ] = dist[block_i*BLOCK_SIZE*n + block_j*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j];
    shared_memory[i   ][j+HALF_BLOCK_FACTOR] = dist[block_i*BLOCK_SIZE*n + block_j*BLOCK_SIZE + i*n + j+HALF_BLOCK_FACTOR];
    shared_memory[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = dist[block_i*BLOCK_SIZE*n + block_j*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j+HALF_BLOCK_FACTOR];

    row_blocks[i   ][j   ] = dist[block_i*BLOCK_SIZE*n + offset + i*n + j];
    row_blocks[i+HALF_BLOCK_FACTOR][j   ] = dist[block_i*BLOCK_SIZE*n + offset + (i+HALF_BLOCK_FACTOR)*n + j];
    row_blocks[i   ][j+HALF_BLOCK_FACTOR] = dist[block_i*BLOCK_SIZE*n + offset + i*n + j + HALF_BLOCK_FACTOR];
    row_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = dist[block_i*BLOCK_SIZE*n + offset + (i+HALF_BLOCK_FACTOR)*n + j + HALF_BLOCK_FACTOR];

    col_blocks[i   ][j   ] = dist[offset*n + block_j*BLOCK_SIZE + i*n + j];
    col_blocks[i+HALF_BLOCK_FACTOR][j   ] = dist[offset*n + block_j*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j];
    col_blocks[i   ][j+HALF_BLOCK_FACTOR] = dist[offset*n + block_j*BLOCK_SIZE + i*n + j+HALF_BLOCK_FACTOR];
    col_blocks[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = dist[offset*n + block_j*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j+HALF_BLOCK_FACTOR];
  
    __syncthreads();

    // #pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        shared_memory[i   ][   j] = min(shared_memory[i][j], row_blocks[i][k] + col_blocks[k][j]);
        shared_memory[i+HALF_BLOCK_FACTOR][j   ] = min(shared_memory[i+HALF_BLOCK_FACTOR][j], row_blocks[i+HALF_BLOCK_FACTOR][k] + col_blocks[k][j]);
        shared_memory[i   ][j+HALF_BLOCK_FACTOR] = min(shared_memory[i][j+HALF_BLOCK_FACTOR], row_blocks[i][k] + col_blocks[k][j+HALF_BLOCK_FACTOR]);
        shared_memory[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR] = min(shared_memory[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR], row_blocks[i+HALF_BLOCK_FACTOR][k] + col_blocks[k][j+HALF_BLOCK_FACTOR]);
    }

    dist[block_i*BLOCK_SIZE*n + block_j*BLOCK_SIZE + i*n + j]         = shared_memory[i   ][j   ];
    dist[block_i*BLOCK_SIZE*n + block_j*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j]    = shared_memory[i+HALF_BLOCK_FACTOR][j   ];
    dist[block_i*BLOCK_SIZE*n + block_j*BLOCK_SIZE + i*n + j+HALF_BLOCK_FACTOR]      = shared_memory[i   ][j+HALF_BLOCK_FACTOR];
    dist[block_i*BLOCK_SIZE*n + block_j*BLOCK_SIZE + (i+HALF_BLOCK_FACTOR)*n + j+HALF_BLOCK_FACTOR] = shared_memory[i+HALF_BLOCK_FACTOR][j+HALF_BLOCK_FACTOR];
}