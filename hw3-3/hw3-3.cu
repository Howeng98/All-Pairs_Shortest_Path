#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#define BLOCK_SIZE 64
#define HALF_BLOCK_SIZE 32
using namespace std;
std::chrono::steady_clock::time_point total_start, total_end;

const int INF = ((1 << 30) - 1);
void input(char* infile);
void output(char *outFileName);

void block_FW();
int ceil(int a, int b);
__global__ void Phase1(int *dist, int Round, int n);
__global__ void Phase2(int *dist, int Round, int n);
__global__ void Phase3(int *dist, int Round, int n, int yoffset);

int N, n, m;
int* Dist = NULL;

int main(int argc, char *argv[]){
    total_start = std::chrono::steady_clock::now();
    input(argv[1]);
	block_FW();
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

void block_FW() {
    const int num_threads = 2;
    int* dst_threads[num_threads];
    int round = ceil(n, BLOCK_SIZE);

    cudaHostRegister(Dist, n*n*sizeof(int), cudaHostRegisterDefault);

	#pragma omp parallel num_threads(num_threads)
	{
        const int cpu_threadID = omp_get_thread_num();
		cudaSetDevice(cpu_threadID);
        cudaMalloc(&dst_threads[cpu_threadID], n*n*sizeof(int));

        const int blocks = (n+BLOCK_SIZE-1) / BLOCK_SIZE;
        int offset = BLOCK_SIZE*n;
	
		int round_per_thread = round / 2;
        const int yoffset = cpu_threadID == 0 ? 0 : round_per_thread;

        if(cpu_threadID == 1 && (round % 2) == 1) 
        round_per_thread += 1;

        dim3 block_dim(32, 32);
		dim3 grid_dim(round, round_per_thread);

        const size_t row_size = offset*sizeof(int);
        const size_t halfBlockSize = row_size * round_per_thread;
        const size_t yOffsetSize = yoffset*offset;
        
        cudaMemcpy(dst_threads[cpu_threadID] + yOffsetSize, Dist + yOffsetSize, halfBlockSize, cudaMemcpyHostToDevice);

		for(int r = 0; r < round; r++) {
            const size_t roundBlockOffset = r * BLOCK_SIZE * n;
            const int isInSelfRange = (r >= yoffset) && (r < (yoffset + round_per_thread));
            if (isInSelfRange) {
                cudaMemcpy(Dist + roundBlockOffset, dst_threads[cpu_threadID] + roundBlockOffset, row_size, cudaMemcpyDeviceToHost);
            }
            #pragma omp barrier

            cudaMemcpy(dst_threads[cpu_threadID] + roundBlockOffset, Dist + roundBlockOffset, row_size, cudaMemcpyHostToDevice);

            Phase1 <<<1, block_dim>>>(dst_threads[cpu_threadID], r, n);
            Phase2 <<<blocks, block_dim>>>(dst_threads[cpu_threadID], r, n);
            Phase3 <<<grid_dim, block_dim>>>(dst_threads[cpu_threadID], r, n, yoffset);
        }

		cudaMemcpy(Dist + yOffsetSize, dst_threads[cpu_threadID] + yOffsetSize, halfBlockSize, cudaMemcpyDeviceToHost);
		#pragma omp barrier
    }
    cudaFree(dst_threads);
}

__global__ void Phase1(int *dist, int Round, int n) {
    __shared__ int shared_memory[64][64];
    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int offset = 64 * Round;

    shared_memory[i   ][j   ] = dist[offset*(n+1) + i*n + j];
    shared_memory[i+32][j   ] = dist[offset*(n+1) + (i+32)*n + j];
    shared_memory[i   ][j+32] = dist[offset*(n+1) + i*n + j + 32];
    shared_memory[i+32][j+32] = dist[offset*(n+1) + (i+32)*n + j + 32];
    __syncthreads();

    #pragma unroll 16
    for (int k = 0; k < 64; k++) {
        shared_memory[i   ][j   ] = min(shared_memory[i   ][j   ], shared_memory[i   ][k] + shared_memory[k][j   ]);
        shared_memory[i+32][j   ] = min(shared_memory[i+32][j   ], shared_memory[i+32][k] + shared_memory[k][j   ]);
        shared_memory[i   ][j+32] = min(shared_memory[i   ][j+32], shared_memory[i   ][k] + shared_memory[k][j+32]);
        shared_memory[i+32][j+32] = min(shared_memory[i+32][j+32], shared_memory[i+32][k] + shared_memory[k][j+32]);
        __syncthreads();
    }

    dist[offset*(n+1) + i*n + j]           = shared_memory[i   ][j   ];
    dist[offset*(n+1) + (i+32)*n + j]      = shared_memory[i+32][j   ];
    dist[offset*(n+1) + i*n + j + 32]      = shared_memory[i   ][j+32];
    dist[offset*(n+1) + (i+32)*n + j + 32] = shared_memory[i+32][j+32];
}

__global__ void Phase2(int *dist, int Round, int n) {
    __shared__ int both[64][64];
    __shared__ int row_blocks[64][64];
    __shared__ int col_blocks[64][64];

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int offset = 64 * Round;
    const int block_i = blockIdx.x;

    if (block_i == Round) 
        return;

    row_blocks[i   ][j   ] = dist[block_i*64*n + offset + i*n + j];
    row_blocks[i+32][j   ] = dist[block_i*64*n + offset + (i+32)*n + j];
    row_blocks[i   ][j+32] = dist[block_i*64*n + offset + i*n + j + 32];
    row_blocks[i+32][j+32] = dist[block_i*64*n + offset + (i+32)*n + j+32];

    col_blocks[i   ][j   ] = dist[offset*n + block_i*64 + i*n + j];
    col_blocks[i+32][j   ] = dist[offset*n + block_i*64 + (i+32)*n + j];
    col_blocks[i   ][j+32] = dist[offset*n + block_i*64 + i*n + j+32];
    col_blocks[i+32][j+32] = dist[offset*n + block_i*64 + (i+32)*n + j+32];

    both[i   ][j   ] = dist[offset*(n+1) + i*n + j];
    both[i+32][j   ] = dist[offset*(n+1) + (i+32)*n + j];
    both[i   ][j+32] = dist[offset*(n+1) + i*n + j+32];
    both[i+32][j+32] = dist[offset*(n+1) + (i+32)*n + j+32];
  
    __syncthreads();

    #pragma unroll 32
    for (int k = 0; k < 64; k++) {

        row_blocks[i   ][j   ] = min(row_blocks[i][j], row_blocks[i][k] + both[k][j]);
        row_blocks[i+32][j   ] = min(row_blocks[i+32][j], row_blocks[i+32][k] + both[k][j]);
        row_blocks[i   ][j+32] = min(row_blocks[i][j+32], row_blocks[i][k] + both[k][j+32]);
        row_blocks[i+32][j+32] = min(row_blocks[i+32][j+32], row_blocks[i+32][k] + both[k][j+32]);

        col_blocks[i   ][j   ] = min(col_blocks[i][j], both[i][k] + col_blocks[k][j]);
        col_blocks[i+32][j   ] = min(col_blocks[i+32][j], both[i+32][k] + col_blocks[k][j]);
        col_blocks[i   ][j+32] = min(col_blocks[i][j+32], both[i][k] + col_blocks[k][j+32]);
        col_blocks[i+32][j+32] = min(col_blocks[i+32][j+32], both[i+32][k] + col_blocks[k][j+32]);
    }

    dist[block_i*64*n + offset + i*n + j]         = row_blocks[i   ][j   ];
    dist[block_i*64*n + offset + (i+32)*n + j]    = row_blocks[i+32][j   ];
    dist[block_i*64*n + offset + i*n + j + 32]    = row_blocks[i   ][j+32];
    dist[block_i*64*n + offset + (i+32)*n + j+32] = row_blocks[i+32][j+32];

    dist[offset*n + block_i*64 + i*n + j]         = col_blocks[i   ][j   ];
    dist[offset*n + block_i*64 + (i+32)*n + j]    = col_blocks[i+32][j   ];
    dist[offset*n + block_i*64 + i*n + j+32]      = col_blocks[i   ][j+32];
    dist[offset*n + block_i*64 + (i+32)*n + j+32] = col_blocks[i+32][j+32];
}

__global__ void Phase3(int *dist, int Round, int n, int yoffset) {
    __shared__ int row_blocks[64][64];
    __shared__ int col_blocks[64][64];
    __shared__ int shared_memory[64][64];
    
    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int block_i = blockIdx.y + yoffset;
    const int block_j = blockIdx.x;
    const int offset = 64 * Round;

    if (block_i == Round && block_j == Round) 
        return;

    shared_memory[i   ][j   ] = dist[block_i*64*n + block_j*64 + i*n + j];
    shared_memory[i+32][j   ] = dist[block_i*64*n + block_j*64 + (i+32)*n + j];
    shared_memory[i   ][j+32] = dist[block_i*64*n + block_j*64 + i*n + j+32];
    shared_memory[i+32][j+32] = dist[block_i*64*n + block_j*64 + (i+32)*n + j+32];

    row_blocks[i   ][j   ] = dist[block_i*64*n + offset + i*n + j];
    row_blocks[i+32][j   ] = dist[block_i*64*n + offset + (i+32)*n + j];
    row_blocks[i   ][j+32] = dist[block_i*64*n + offset + i*n + j + 32];
    row_blocks[i+32][j+32] = dist[block_i*64*n + offset + (i+32)*n + j + 32];

    col_blocks[i   ][j   ] = dist[offset*n + block_j*64 + i*n + j];
    col_blocks[i+32][j   ] = dist[offset*n + block_j*64 + (i+32)*n + j];
    col_blocks[i   ][j+32] = dist[offset*n + block_j*64 + i*n + j+32];
    col_blocks[i+32][j+32] = dist[offset*n + block_j*64 + (i+32)*n + j+32];
  
    __syncthreads();

    #pragma unroll 32
    for (int k = 0; k < 64; k++) {
        shared_memory[i   ][   j] = min(shared_memory[i][j], row_blocks[i][k] + col_blocks[k][j]);
        shared_memory[i+32][j   ] = min(shared_memory[i+32][j], row_blocks[i+32][k] + col_blocks[k][j]);
        shared_memory[i   ][j+32] = min(shared_memory[i][j+32], row_blocks[i][k] + col_blocks[k][j+32]);
        shared_memory[i+32][j+32] = min(shared_memory[i+32][j+32], row_blocks[i+32][k] + col_blocks[k][j+32]);
    }

    dist[block_i*64*n + block_j*64 + i*n + j]         = shared_memory[i   ][j   ];
    dist[block_i*64*n + block_j*64 + (i+32)*n + j]    = shared_memory[i+32][j   ];
    dist[block_i*64*n + block_j*64 + i*n + j+32]      = shared_memory[i   ][j+32];
    dist[block_i*64*n + block_j*64 + (i+32)*n + j+32] = shared_memory[i+32][j+32];
}