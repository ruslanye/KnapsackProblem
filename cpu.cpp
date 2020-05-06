// C++ program to create target string, starting from
// random string using Genetic Algorithm

#include <bits/stdc++.h>
#include <chrono>
using namespace std;

// Number of individuals in each generation
#define POPULATION_SIZE 64

#define GENERATIONS 100

vector<int> weights, values;
int W, N;

// Function to generate random numbers in given range
int random_num(int start, int end) {
    int range = (end - start) + 1;
    int random_int = start + (rand() % range);
    return random_int;
}

// Create random genes for mutation
char mutated_genes() {
    int r = random_num(0, N - 2);
    return r % 2;
}

// create chromosome or string of genes
void create_gnome(vector<int> &gnome) {
    for (int i = 0; i < N; i++)
        gnome[i] = mutated_genes();
}

// Class representing individual in population
class Individual {
  public:
    vector<int> chromosome;
    int fitness;
    Individual(vector<int> &chromosome);
    Individual mate(Individual &parent2);
    int cal_fitness();
};

Individual::Individual(vector<int> &chromosome) {
    this->chromosome = vector<int>(chromosome);
    fitness = cal_fitness();
};

// Perform mating and produce new offspring
Individual Individual::mate(Individual &par2) {
    // chromosome for offspring
    vector<int> child_chromosome(N, 0);

    int len = chromosome.size();
    for (int i = 0; i < len; i++) {
        // random probability
        float p = random_num(0, 100) / 100;

        // if prob is less than 0.45, insert gene
        // from parent 1
        if (p < 0.45)
            child_chromosome[i] = chromosome[i];

        // if prob is between 0.45 and 0.90, insert
        // gene from parent 2
        else if (p < 0.90)
            child_chromosome[i] = par2.chromosome[i];

        // otherwise insert random gene(mutate),
        // for maintaining diversity
        else
            child_chromosome[i] = mutated_genes();
    }

    // create new Individual(offspring) using
    // generated chromosome for offspring
    return Individual(child_chromosome);
};

// Calculate fittness score, it is the number of
// characters in string which differ from target
// string.
int Individual::cal_fitness() {
    int fitness = 0, weight = 0;
    for (int i = 0; i < N; i++) {
        fitness += chromosome[i] * values[i];
        weight += chromosome[i] * weights[i];
    }
    if (weight > W)
        fitness = 0;
    return fitness;
};

// Overloading < operator
bool operator<(const Individual &ind1, const Individual &ind2) {
    return ind1.fitness > ind2.fitness;
}

// Driver code
int main() {
    srand((unsigned)(time(0)));

    cin >> N >> W;

    values.resize(N, 0);
    weights.resize(N, 0);

    for (int i = 0; i < N; i++) {
        cin >> weights[i] >> values[i];
    }

    // current generation
    int generation = 0;

    vector<Individual> population;
    bool found = false;

    // create initial population
    for (int i = 0; i < POPULATION_SIZE; i++) {
        vector<int> gnome(N);
        create_gnome(gnome);
        population.push_back(Individual(gnome));
    }

    int not_changed = 0;
    int prev = 0;
    auto start = chrono::steady_clock::now();
    for (int g = 0; g < GENERATIONS; g++)
    // while(not_changed<GENERATIONS)
    {
        // sort the population in increasing order of fitness score
        sort(population.begin(), population.end());

        // if the individual having lowest fitness score ie.
        // 0 then we know that we have reached to the target
        // and break the loop
        // if (population[0].fitness <= 0) {
        //     found = true;
        //     break;
        // }

        // Otherwise generate new offsprings for new generation
        vector<Individual> new_generation;

        // Perform Elitism, that mean 10% of fittest population
        // goes to the next generation
        int s = (10 * POPULATION_SIZE) / 100;
        for (int i = 0; i < s; i++)
            new_generation.push_back(population[i]);

        // From 50% of fittest population, Individuals
        // will mate to produce offspring
        s = (90 * POPULATION_SIZE) / 100;
        for (int i = 0; i < s; i++) {
            int len = population.size();
            int r = random_num(0, 50);
            Individual parent1 = population[r];
            r = random_num(0, 50);
            Individual parent2 = population[r];
            Individual offspring = parent1.mate(parent2);
            new_generation.push_back(offspring);
        }
        population = new_generation;
        if (population[0].fitness == prev)
            not_changed++;
        else
            not_changed = 0;
        prev = population[0].fitness;
        // cout << "Generation: " << generation << "\t";
        // cout << "Items: ";
        // for (int i = 0; i < N; i++){
        //     cout << population[0].chromosome[i] << " ";
        // }
        // cout << "\t";
        // cout << "Fitness: " << population[0].fitness << "\n";

        generation++;
    }
    auto stop = chrono::steady_clock::now();
    // cout << "Generation: " << generation << "\n";
    cout << "Profit: " << population[0].fitness << "\n";
    for (int i = 0; i < N; i++) {
        cout << population[0].chromosome[i] << " ";
    }
    cout << "\n";

    cerr << "Elapsed time: "
         << chrono::duration_cast<chrono::microseconds>(stop - start).count()
         << "μs\n";
}