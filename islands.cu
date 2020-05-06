#include <iostream>
#include <chrono>
#include <vector>
#include<curand_kernel.h>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

constexpr int MAX_SIZE = 300;
constexpr int GENERATIONS = 100;
constexpr int ISLANDS = 32;
constexpr int POPULATION = 64;

__device__ __forceinline__ void swap(int& a, int& b){
    int temp = a;
    a = b;
    b = temp;
}

__device__ __forceinline__ int cal_fitness(int* weights, int* values, char* genom, int W, int n){
    int fitness = 0, weight = 0;
    for(int i = 0; i < n; i++){
        fitness+=values[i]*(int)genom[i];
        weight+=weights[i]*(int)genom[i];
    }
    if(weight>W)
        fitness = 0;
    return fitness;
}

__device__ __forceinline__ void selection(char* a, char* b, int n){
    for(int i = 0; i < n; i++){
        a[i] = b[i];
    }
}

__device__ __forceinline__ void crossover(char* mother, char* father, char* child, curandState_t* state, int n){
    for(int i = 0; i < n; i++){
        double p = curand_uniform(state);
        if(p < 0.45)
            child[i] = father[i];
        else if(p < 0.90)
            child[i] = mother[i];
        else
            child[i] = curand(state)%2;
    }
}

__device__ __forceinline__ void sort(int* fitness, int* pos, int n){
    int tid = threadIdx.x;
    for(unsigned int k = 2; k <= POPULATION; k*=2){
        for(unsigned int j = k/2; j > 0; j/=2){
            unsigned int ixj = tid^j;
            if(ixj>tid){
                if((tid & k) == 0)
                    if(fitness[tid] < fitness[ixj]){
                        swap(fitness[tid], fitness[ixj]);
                        swap(pos[tid], pos[ixj]);
                    }
                else if(fitness[tid] > fitness[ixj]){
                    swap(fitness[tid], fitness[ixj]);
                    swap(pos[tid], pos[ixj]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void prefixmax(int* arr, int* pos, int n){
    int x, p;
    int tid = threadIdx.x;
    for(int i = 1; i < n; i*=2){
        if(tid>=i){
            x = arr[tid-i];
            p = pos[tid-i];
        }
        __syncthreads();
        if(tid>=i&&x>arr[tid]){
            arr[tid] = x;
            pos[tid] = p;
            
        }
        __syncthreads();    
    }
}

__global__ void kernel(int* w, int* v, int n, int W, char* result, int* profit, curandState_t* states) {
    __shared__ int weights[MAX_SIZE];
    __shared__ int values[MAX_SIZE];
    __shared__ char population[POPULATION][MAX_SIZE];
    __shared__ char new_population[POPULATION][MAX_SIZE];
    __shared__ int fitness[POPULATION];
    __shared__ int pos[POPULATION];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = blockDim.x*bid + tid;
    int frac = POPULATION/10;
    int p1, p2;

    curandState_t state = states[id];

    for(int i = tid; i < n; i+=POPULATION){
        weights[i] = w[i];
        values[i] = v[i];
    }
    __syncthreads();
    for(int i = 0; i < n; i++){
        population[tid][i] = curand(&state)%2;
    }
    int not_changed = 0;
    int prev = 0;
    int iter = 0;
    for(int g = 0; g < GENERATIONS+1; g++)
    //while(not_changed<GENERATIONS)
    {
        iter++;
        fitness[tid] = cal_fitness(weights, values, population[tid], W, n);
        pos[tid] = tid;
        sort(fitness, pos, n);
        __syncthreads();
        // if(prev == fitness[0])
        //     not_changed++;
        // else
        //     not_changed = 0;
        // prev = fitness[0];
        // __syncthreads();
        if(tid < frac){
            selection(new_population[tid], population[pos[tid]], n);
        }
        if(tid >= frac){
            p1 = ceilf(curand_uniform(&state) * (POPULATION/2));
            p2 = ceilf(curand_uniform(&state) * (POPULATION/2));
            crossover(population[pos[p1]], population[pos[p2]], new_population[tid], &state, n);
        }
        __syncthreads();
        for(int i = 0; i < n; i++)
            population[tid][i] = new_population[tid][i];
        __syncthreads();
    }
    fitness[tid] = cal_fitness(weights, values, population[tid], W, n);
    pos[tid] = tid;
    __syncthreads();
    prefixmax(fitness, pos, n);
    if(tid == 0){
        profit[bid] = fitness[POPULATION-1];
        // stats[bid] = iter;
    }
    __syncthreads();
    for(int i = tid; i < n; i+=POPULATION)
        result[bid*n+i] = population[pos[POPULATION-1]][i];
}

__global__ void init(curandState_t* states, unsigned int seed){
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

int main(){
    cudaSetDevice(0);

    int *d_weights, *d_values, *d_profit;
    // int* d_stats;
    char* d_result;
    curandState_t* states;
    int n, W;
    cin>>n>>W;
    vector<int> weights(n), values(n), profit(ISLANDS);
    vector<char> result(ISLANDS*n);
    // vector<int> stats(ISLANDS);
    for(int i = 0; i < n; i++){
        cin>>weights[i]>>values[i];
    }
    
    gpuErrchk(cudaMalloc(&d_weights, n*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_values, n*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_result, ISLANDS*n*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_profit, ISLANDS*sizeof(int)));
    // gpuErrchk(cudaMalloc(&d_stats, ISLANDS*sizeof(int)));
    gpuErrchk(cudaMalloc(&states, ISLANDS*POPULATION*sizeof(curandState_t)));
    gpuErrchk(cudaMemcpy(d_weights, weights.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_values, values.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    init<<<ISLANDS, POPULATION>>>(states, time(0));

    gpuErrchk(cudaDeviceSynchronize());

    auto start = chrono::steady_clock::now();

    kernel<<<ISLANDS, POPULATION>>>(d_weights, d_values, n, W, d_result, d_profit, states);

    gpuErrchk(cudaDeviceSynchronize());

    auto stop = chrono::steady_clock::now();

    gpuErrchk(cudaMemcpy(profit.data(), d_profit, ISLANDS*sizeof(int), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(stats.data(), d_stats, ISLANDS*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(result.data(), d_result, n*ISLANDS, cudaMemcpyDeviceToHost));

    int best = 0;
    // int worst = 0;
    for(int i = 0; i < ISLANDS; i++){
        if(profit[i]>profit[best])
        best = i;
        // if(stats[i]>stats[worst])
        // worst = i;
    }
    cout<<"Best island: "<<best<<endl;
    cout<<"Profit: "<<profit[best]<<endl;
    // cout<<"Max generations:"<<stats[worst]<<endl;
    for(int i = 0; i < n; i++)
        cout<<+result[best*n+i]<<" ";
    cout<<endl;
    cerr << "Elapsed time: " << chrono::duration_cast<chrono::microseconds>(stop - start).count() << "μs\n";
    gpuErrchk(cudaFree(states));
    gpuErrchk(cudaFree(d_weights));
    gpuErrchk(cudaFree(d_values));
    gpuErrchk(cudaFree(d_profit));
    gpuErrchk(cudaFree(d_result));
    // gpuErrchk(cudaFree(d_stats));
    return 0;
}