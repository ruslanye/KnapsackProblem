#include <bits/stdc++.h>
#include <curand_kernel.h>

using namespace std;

constexpr int POPULATION_SIZE = 128;  

constexpr int GENERATIONS = 100;

constexpr double MUTATION_RATE = 0.1;

//constexpr int MAX_SIZE = 1000;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// __constant__ int d_weights[MAX_SIZE];
// __constant__ int d_values[MAX_SIZE];

__global__ void curand_init(curandState_t* states, unsigned int seed){
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    curand_init(seed+id, id, 0, &states[id]);
}

__global__ void init(int* pop, int N, int i, curandState_t* states) {
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t state = states[id];
    pop[i*N+id] = ceilf(curand_uniform(&state) * 2)-1;
    
}

__global__ void zero(int* values){
    values[threadIdx.x] = 0;
}

__global__ void evaluate(int* pop, int* fitness, int N, int* weights, int* values, int* weight, int i) {
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    if (id < N) {
        int v = values[id]*pop[i*N+id];
        int w = weights[id]*pop[i*N+id];
        atomicAdd(fitness+i, v);
        atomicAdd(weight, w);
    }
}

__global__ void check(int* fitness, int* weight, int W, int i){
    if ((*weight)>W)
        fitness[i] = 0;
}

__global__ void get_best(int* values, int* best, int n){
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ int arr[POPULATION_SIZE];
    __shared__ int pos[POPULATION_SIZE];
    arr[id] = values[id];
    pos[id] = id;
    __syncthreads();
    int x, p;
    for(int i = 1; i < n; i*=2){
        if(id>=i){
            x = arr[id-i];
            p = pos[id-i];
        }
        __syncthreads();
        if(id>=i&&x>arr[id]){
            arr[id] = x;
            pos[id] = p;
            
        }
        __syncthreads();    
    }
    if(id == POPULATION_SIZE-1)
        *best = pos[id];
}

__global__ void prefixsum(int* values, int n){
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ int arr[POPULATION_SIZE];
    arr[id] = values[id];
    __syncthreads();
    int x;
    for(int i = 1; i < n; i*=2){
        if(id>=i){
            x = arr[id-i];
        }
        __syncthreads();
        if(id>=i&&x>arr[id]){
            arr[id] = x + arr[id];
        }
        __syncthreads();    
    }
    values[id] = arr[id];
}

__global__ void copy_elite(int* pop, int* new_pop, int* best, int N){
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    if(id < N){
        int pos = *best;
        new_pop[id] = pop[N*pos+id];
    }
}

__device__ int lower_bound(int* arr, int n, int val){
    int l = 0;
    int h = n;
    while (l < h) {
        int mid = (l + h) / 2;
        if (val <= arr[mid]) {
            h = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

__device__ int select(int* values, curandState_t* state) {
    int m = values[POPULATION_SIZE-1];
    if (m == 0)
        return ceilf(curand_uniform(state) * POPULATION_SIZE)-1;
    return lower_bound(values, POPULATION_SIZE, ceilf(curand_uniform(state) * m) - 1);
}

__global__ void prepare(int* fathers, int* mothers, int* positions, int* fitness, curandState_t* states, int N){
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    if(id > 0){
        curandState_t state = states[id];
        fathers[id] = select(fitness, &state);
        mothers[id] = select(fitness, &state);
        positions[id] = ceilf(curand_uniform(&state) * N)-1;
    }
}

__global__ void crossover(int* pop, int* father, int* mother, int* child, int* position, int N) {
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < *position){
        child[id] = pop[(*father)*N+id];
    } else if(id < N){
        child[id] = pop[(*mother)*N+id];
    }
}

__global__ void mutate(int* genom, int N, curandState_t* state) {
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if(id < N){
        if (curand_uniform(state)<=MUTATION_RATE)
            genom[id] = 1 - genom[id];
    }
}

int main() {
    
    vector<int> weights, values, genom;
    int *pop, *new_pop, *fitness, *temp, *d_weights, *d_values, *fathers, *mothers, *positions;
    curandState_t* states;
    int W, N;
    unsigned int blocksize = 1024;
    unsigned int gridsize;

    cin >> N >> W;

    gridsize = (N + blocksize - 1)/blocksize;

    values.resize(N, 0);
    weights.resize(N, 0);
    genom.resize(N, 0);

    for (int i = 0; i < N; i++) {
        cin >> weights[i] >> values[i];
    }

    // gpuErrchk(cudaMemcpyToSymbol(d_weights, weights.data(), N*sizeof(int)));
    // gpuErrchk(cudaMemcpyToSymbol(d_values, values.data(), N*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_weights, N*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_values, N*sizeof(int)));
    gpuErrchk(cudaMalloc(&pop, POPULATION_SIZE*N*sizeof(int)));
    gpuErrchk(cudaMalloc(&new_pop, POPULATION_SIZE*N*sizeof(int)));
    gpuErrchk(cudaMalloc(&fitness, POPULATION_SIZE*sizeof(int)));
    gpuErrchk(cudaMalloc(&fathers, POPULATION_SIZE*sizeof(int)));
    gpuErrchk(cudaMalloc(&mothers, POPULATION_SIZE*sizeof(int)));
    gpuErrchk(cudaMalloc(&positions, POPULATION_SIZE*sizeof(int)));
    gpuErrchk(cudaMalloc(&temp, sizeof(int)));
    gpuErrchk(cudaMalloc(&states, POPULATION_SIZE*N*sizeof(curandState_t)));

    gpuErrchk(cudaMemcpy(d_weights, weights.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_values, values.data(), N*sizeof(int), cudaMemcpyHostToDevice));

    curand_init<<<gridsize, blocksize>>>(states, time(0));

    int gen = 0;

    auto start = chrono::steady_clock::now();
    for(int i = 0; i < POPULATION_SIZE; i++){
        init<<<gridsize, blocksize>>>(pop, N, i, states);
        gpuErrchk(cudaDeviceSynchronize());
    }
    while (gen < GENERATIONS) {
        zero<<<1, POPULATION_SIZE>>>(fitness);
        for(int i = 0; i < POPULATION_SIZE; i++){
            int z = 0;
            gpuErrchk(cudaMemcpy(temp, &z, sizeof(int), cudaMemcpyHostToDevice));
            evaluate<<<gridsize, blocksize>>>(pop, fitness, N, d_weights, d_values, temp, i);
            gpuErrchk(cudaDeviceSynchronize());
            check<<<1, 1>>>(fitness, temp, W, i);
            gpuErrchk(cudaDeviceSynchronize());
        }
        get_best<<<1, POPULATION_SIZE>>>(fitness, temp, N);
        gpuErrchk(cudaDeviceSynchronize());
        prefixsum<<<1, POPULATION_SIZE>>>(fitness, N);
        gpuErrchk(cudaDeviceSynchronize());
        copy_elite<<<gridsize, blocksize>>>(pop, new_pop, temp, N);
        gpuErrchk(cudaDeviceSynchronize());
        prepare<<<1, POPULATION_SIZE>>>(fathers, mothers, positions, fitness, states, N);
        for(int i = 0; i < POPULATION_SIZE; i++){
            crossover<<<gridsize, blocksize>>>(pop, fathers+i, mothers+i, new_pop+i, positions+i, N);
            gpuErrchk(cudaDeviceSynchronize());
            mutate<<<gridsize, blocksize>>>(new_pop+i, N, states);
        }
        swap(pop, new_pop);
        gen++;
    }
    auto stop = chrono::steady_clock::now();
    zero<<<1, POPULATION_SIZE>>>(fitness);
    for(int i = 0; i < POPULATION_SIZE; i++){
        int z= 0;
        gpuErrchk(cudaMemcpy(temp, &z, sizeof(int), cudaMemcpyHostToDevice));
        evaluate<<<gridsize, blocksize>>>(pop, fitness, N, d_weights, d_values, temp, i);
        gpuErrchk(cudaDeviceSynchronize());
        check<<<1, 1>>>(fitness, temp, W, i);
        gpuErrchk(cudaDeviceSynchronize());
    }
    get_best<<<1, POPULATION_SIZE>>>(fitness, temp, N);
    int pos, result;
    gpuErrchk(cudaMemcpy(&pos, temp, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(genom.data(), pop+pos*N, N*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&result, fitness+pos, sizeof(int), cudaMemcpyDeviceToHost));
    cout << "Profit: " << result << "\n";
    // for (int i = 0; i < N; i++) {
    //     cout << genom[i] << " ";
    // }
    // cout << "\n";

    cerr << "Elapsed time: "
         << chrono::duration_cast<chrono::microseconds>(stop - start).count()
         << "μs\n";
    return 0;
}