#include <bits/stdc++.h>
using namespace std;

#define POPULATION_SIZE 128  

#define GENERATIONS 100

#define MUTATION_RATE 100

void init(vector<vector<int>> &pop, int N) {
    for (int g = 0; g < POPULATION_SIZE; g++) {
        for (int i = 0; i < N; i++) {
            pop[g].emplace_back(rand() % 2);
        }
    }
}

int evaluate(vector<vector<int>> &pop, vector<int> &fitness,
             vector<int> &values, vector<int> &weights, int N, int W) {
    int best = 0;
    for (int g = 0; g < POPULATION_SIZE; g++) {
        fitness[g] = 0;
        int weight = 0;
        for (int i = 0; i < N; i++) {
            fitness[g] += pop[g][i] * values[i];
            weight += pop[g][i] * weights[i];
        }
        if (weight > W)
            fitness[g] = 0;
        if (fitness[g] > fitness[best])
            best = g;
    }
    return best;
}

void prefixsum(vector<int> &values) {
    for (int i = 1; i < values.size(); i++)
        values[i] = values[i - 1] + values[i];
}

int select(vector<int> &values) {
    if (values.back() == 0)
        return rand() % values.size();
    return lower_bound(values.begin(), values.end(), rand() % values.back()) -
           values.begin();
}

void crossover(vector<int> &father, vector<int> &mother, vector<int> &child) {
    int pos = rand() % father.size();
    for (int i = 0; i < pos; i++)
        child[i] = father[i];
    for (int i = pos; i < father.size(); i++)
        child[i] = mother[i];
}

void mutate(vector<int> &genom) {
    for (int i = 0; i < genom.size(); i++) {
        if (rand() % MUTATION_RATE == 1)
            genom[i] = 1 - genom[i];
    }
}

int main() {
    srand((unsigned)(time(0)));
    vector<int> weights, values;
    vector<vector<int>> pop, new_pop;
    vector<int> fitness;
    int W, N;

    cin >> N >> W;

    values.resize(N, 0);
    weights.resize(N, 0);
    pop.resize(POPULATION_SIZE);
    fitness.resize(POPULATION_SIZE);

    for (int i = 0; i < N; i++) {
        cin >> weights[i] >> values[i];
    }
    int gen = 0;
    auto start = chrono::steady_clock::now();
    init(pop, N);
    while (gen < GENERATIONS) {
        int id = evaluate(pop, fitness, values, weights, N, W);
        prefixsum(fitness);
        new_pop.resize(POPULATION_SIZE);
        new_pop[0] = pop[id];
        for (int i = 1; i < POPULATION_SIZE; i++) {
            new_pop[i].resize(N);
            int father = select(fitness);
            int mother = select(fitness);
            crossover(pop[father], pop[mother], new_pop[i]);
            mutate(new_pop[i]);
        }
        swap(pop, new_pop);
        gen++;
    }
    auto stop = chrono::steady_clock::now();
    int id = evaluate(pop, fitness, values, weights, N, W);
    cout << "Profit: " << fitness[id] << "\n";
    // for (int i = 0; i < N; i++) {
    //     cout << pop[id][i] << " ";
    // }
    // cout << "\n";

    cerr << "Elapsed time: "
         << chrono::duration_cast<chrono::microseconds>(stop - start).count()
         << "μs\n";
    return 0;
}