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

__global__ void init(int* pop, int N, curandState_t* states) {
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t state = states[id];
    for (int i = 0; i < N; i++) {
        pop[id*N+i] = ceilf(curand_uniform(&state) * 2)-1;
    }
}

__global__ void evaluate(int* pop, int* fitness, int N, int W, int* weights, int* values) {
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    fitness[id] = 0;
    int weight = 0;
    for (int i = 0; i < N; i++) {
        fitness[id] += pop[id*N+i] * values[i];
        weight += pop[id*N+i] * weights[i];
    }
    if (weight > W)
        fitness[id] = 0;
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

__device__ void crossover(int* father, int* mother, int* child, int N, curandState_t* state) {
    int pos = ceilf(curand_uniform(state) * N)-1;
    for (int i = 0; i < pos; i++)
        child[i] = father[i];
    for (int i = pos; i < N; i++)
        child[i] = mother[i];
}

__device__ void mutate(int* genom, int N, curandState_t* state) {
    for (int i = 0; i < N; i++) {
        if (curand_uniform(state)<=MUTATION_RATE)
            genom[i] = 1 - genom[i];
    }
}

__global__ void generate(int* pop, int* new_pop, int* fitness, int N, curandState_t* states, int* best){
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    if(id == 0){
        int pos = *best;
        for(int i = 0; i < N; i++)
            new_pop[i] = pop[pos*N+i];
    } else{
        curandState_t state = states[id];
        int father = select(fitness, &state);
        int mother = select(fitness, &state);
        crossover(pop+N*father, pop+N*mother, new_pop+N*id, N, &state);
        mutate(new_pop+N*id, N, &state);
    }
}

int main() {
    
    vector<int> weights, values, genom;
    int *pop, *new_pop, *fitness, *best, *d_weights, *d_values;
    curandState_t* states;
    int W, N;

    cin >> N >> W;

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
    gpuErrchk(cudaMalloc(&best, sizeof(int)));
    gpuErrchk(cudaMalloc(&states, POPULATION_SIZE*sizeof(curandState_t)));

    gpuErrchk(cudaMemcpy(d_weights, weights.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_values, values.data(), N*sizeof(int), cudaMemcpyHostToDevice));

    curand_init<<<1, POPULATION_SIZE>>>(states, time(0));

    int gen = 0;

    auto start = chrono::steady_clock::now();
    init<<<1, POPULATION_SIZE>>>(pop, N, states);
    gpuErrchk(cudaDeviceSynchronize());
    while (gen < GENERATIONS) {
        evaluate<<<1, POPULATION_SIZE>>>(pop, fitness, N, W, d_weights, d_values);
        gpuErrchk(cudaDeviceSynchronize());
        get_best<<<1, POPULATION_SIZE>>>(fitness, best, N);
        gpuErrchk(cudaDeviceSynchronize());
        prefixsum<<<1, POPULATION_SIZE>>>(fitness, N);
        gpuErrchk(cudaDeviceSynchronize());
        generate<<<1, POPULATION_SIZE>>>(pop, new_pop, fitness, N, states, best);
        gpuErrchk(cudaDeviceSynchronize());
        swap(pop, new_pop);
        gen++;
    }
    auto stop = chrono::steady_clock::now();
    evaluate<<<1, POPULATION_SIZE>>>(pop, fitness, N, W, d_weights, d_values);
    gpuErrchk(cudaDeviceSynchronize());
    get_best<<<1, POPULATION_SIZE>>>(fitness, best, N);
    int pos, result;
    gpuErrchk(cudaMemcpy(&pos, best, sizeof(int), cudaMemcpyDeviceToHost));
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