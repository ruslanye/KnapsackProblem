#include <chrono>
#include <iostream>

using namespace std;

#define MAX_SIZE 1073741824

int w[MAX_SIZE], c[MAX_SIZE], B, d[2][MAX_SIZE], z, n;

int main() {
    cin >> n;
    cin >> B;
    for (int i = 0; i < n; i++) {
        cin >> w[i + 1] >> c[i + 1];
    }
    for (int i = 0; i <= B; i++) {
        d[0][i] = 0;
    }
    auto start = chrono::steady_clock::now();
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= B; j++) {
            int a = d[(i - 1) % 2][j];
            int b = (j - w[i] >= 0) ? (d[(i - 1) % 2][j - w[i]] + c[i]) : 0;
            d[i % 2][j] = a > b ? a : b;
        }
    }
    cout << d[n % 2][B] << endl;

    auto stop = chrono::steady_clock::now();
    cerr << "Elapsed time: "
         << chrono::duration_cast<chrono::microseconds>(stop - start).count()
         << "μs\n";
    return 0;
}