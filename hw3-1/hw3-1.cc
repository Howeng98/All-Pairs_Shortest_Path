#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <pthread.h>
#include <emmintrin.h>
#include <time.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;
inline void input(char* inFileName);
inline void output(char* outFileName);

int n, m; //vertex number, edges number
static int Dist[V][V];

int current_i;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t thread_barrier;

typedef struct{
    int value;
} Args;

void* worker(void* args){
    int k,i,j;
    for(k=0; k<n; k++){
        current_i = 0;
        while(current_i < n){
            pthread_mutex_lock(&mutex);
            // do something
            if(current_i < n){
                i = current_i;
                current_i++;
            }
            pthread_mutex_unlock(&mutex);
            for(j=0; j<n; j++){
                if(Dist[i][j] > Dist[i][k]+Dist[k][j] && (Dist[i][k]!=INF && Dist[k][j]!=INF))
                    Dist[i][j] = Dist[i][k]+Dist[k][j];
            }
        }
        pthread_barrier_wait(&thread_barrier);
    }
    pthread_exit(NULL);
    return NULL;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file); //vertex number
    fread(&m, sizeof(int), 1, file); //edges number

    //initialize Dist
    for (int i = 0; i < n; ++i) {
        // #pragma unroll 3
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // #pragma unroll 3
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int main(int argc, char* argv[]) {

    // CPU number
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_threads = CPU_COUNT(&cpu_set);
    // printf("%d\n", num_threads);

    // Pthread initialize
    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&thread_barrier, NULL, num_threads);
    pthread_t threads[num_threads];
    Args args[num_threads];

    // Read data
    input(argv[1]);

    int i,j,k,idx;
    current_i = 0;
    for(idx=0;idx<num_threads;idx++){
        args[idx].value = 0;
    }

    // Create pthread
    for(idx=0;idx<num_threads;idx++){
        pthread_create(&threads[idx], NULL, worker, &args[idx]);
    }

    // Join pthread
    for(idx=0;idx<num_threads;idx++){
        pthread_join(threads[idx], NULL);
    }

    output(argv[2]);
    return 0;
}
