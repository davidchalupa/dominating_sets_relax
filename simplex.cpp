#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <numeric>

#include "common.h"
#include "simplex.h"

class RelaxedLPRelaxationSolver {
public:
    double compute_lower_bound(graph G, int max_iterations = 5000) {
        int n = G->n;
        if (n <= 0) return 0.0;

        // Primal variables (x), Dual variables (y), and the Extrapolated Primal (x_bar)
        std::vector<double> x(n, 0.0);
        std::vector<double> y(n, 0.0);
        std::vector<double> x_old(n, 0.0);

        // Heuristic step sizes based on the max degree to ensure stability
        double max_deg = 0;
        for (int i = 0; i < n; i++)
            if (G->V[i].edgecount > max_deg) max_deg = G->V[i].edgecount;

        // Step sizes: tau * sigma * ||A||^2 < 1.
        // For this graph, ||A|| is bounded by (max_deg + 1)
        double tau = 1.0 / (max_deg + 1.0);
        double sigma = 1.0 / (max_deg + 1.0);

        for (int iter = 0; iter < max_iterations; iter++) {
            // 1. Update Dual Variables (y)
            // y = clip(y + sigma * (A * x_bar - 1), 0, inf)
            // Note: x_bar = 2x_curr - x_old
            for (int i = 0; i < n; i++) {
                double x_bar_i = 2.0 * x[i] - x_old[i];
                double Ax_i = x_bar_i; // self-loop (vertex dominates itself)
                for (int k = 0; k < G->V[i].edgecount; k++) {
                    refer neighbor = G->V[i].sibl[k];
                    if (neighbor >= 0 && neighbor < n) {
                        double x_bar_neigh = 2.0 * x[neighbor] - x_old[neighbor];
                        Ax_i += x_bar_neigh;
                    }
                }

                // We want Ax >= 1, so the residual is (1 - Ax)
                y[i] += sigma * (1.0 - Ax_i);
                if (y[i] < 0) y[i] = 0;
            }

            // Store current x to calculate x_bar in the next round
            x_old = x;

            // 2. Update Primal Variables (x)
            // x = clip(x - tau * (1 - A^T * y), 0, 1)
            for (int i = 0; i < n; i++) {
                double ATy_i = y[i]; // self-loop
                for (int k = 0; k < G->V[i].edgecount; k++) {
                    refer neighbor = G->V[i].sibl[k];
                    if (neighbor >= 0 && neighbor < n) {
                        ATy_i += y[neighbor];
                    }
                }

                // Objective is to minimize sum(x), so gradient is 1.
                // Constraint is Ax >= 1, and A is symmetric for undirected graphs.
                x[i] -= tau * (1.0 - ATy_i);

                // Projection onto [0, 1]
                if (x[i] < 0) x[i] = 0;
                if (x[i] > 1) x[i] = 1;
            }

            // 3. Early Exit (Convergence check)
            if (iter % 200 == 0 && iter > 0) {
                double primal_obj = 0;
                for (double val : x) primal_obj += val;
                // If the change is negligible, we are done
                if (primal_obj > 0.1) { /* Optional: compare to previous and break */ }
            }
        }

        double final_lb = 0;
        for (double val : x) final_lb += val;
        return final_lb;
    }
};

class StrictMDSLowerBound {
public:
    double compute(graph G) {
        int n = G->n;
        if (n <= 0) return 0.0;

        // Dual variables y_i, initialized to 0.0
        std::vector<double> y(n, 0.0);

        // current_load[i] = sum of y_j for all j in N[i]
        // We must maintain the invariant: current_load[i] <= 1.0 for all i
        std::vector<double> current_load(n, 0.0);

        // Safety epsilon to account for floating point precision
        // This ensures we stay strictly below or equal to the optimum
        const double epsilon = 1e-12;
        const double limit = 1.0 - epsilon;

        // One pass is often enough for a decent bound,
        // but multiple passes allow y_i to "fill in the gaps"
        for (int pass = 0; pass < 10; pass++) {
            bool changed = false;
            for (int i = 0; i < n; i++) {
                // How much can we increase y[i] without breaking
                // the constraint for i OR any of its neighbors?

                double max_increase = limit - current_load[i];
                if (max_increase <= 0) continue;

                for (int k = 0; k < G->V[i].edgecount; k++) {
                    refer neighbor = G->V[i].sibl[k];
                    if (neighbor >= 0 && neighbor < n) {
                        double slack = limit - current_load[neighbor];
                        if (slack < max_increase) max_increase = slack;
                    }
                }

                // If we found room to grow, increase y[i]
                if (max_increase > epsilon) {
                    y[i] += max_increase;

                    // Update the loads
                    current_load[i] += max_increase;
                    for (int k = 0; k < G->V[i].edgecount; k++) {
                        refer neighbor = G->V[i].sibl[k];
                        if (neighbor >= 0 && neighbor < n) {
                            current_load[neighbor] += max_increase;
                        }
                    }
                    changed = true;
                }
            }
            if (!changed) break;
        }

        // Sum of y_i is a mathematically guaranteed lower bound
        double lower_bound = 0.0;
        for (int i = 0; i < n; i++) {
            lower_bound += y[i];
        }
        return lower_bound;
    }
};

class StrictMDSLowerBound2 {
public:
    double compute(graph G) {
        int n = G->n;
        if (n <= 0) return 0.0;

        std::vector<double> y(n, 0.0);
        std::vector<double> current_load(n, 0.0);

        const double epsilon = 1e-12;
        const double limit = 1.0 - epsilon;

        // 1. Create an ordering of vertices based on degree (ascending)
        // Heuristic: Small neighborhoods are easier to "fill" without
        // blocking too many neighboring constraints.
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return G->V[a].edgecount < G->V[b].edgecount;
        });

        // 2. Multi-pass greedy packing
        // We use the degree-ascending order to maximize the packing.
        for (int pass = 0; pass < 5; pass++) {
            bool changed = false;

            for (int i : order) {
                // Calculate max allowable increase for y[i]
                // It must satisfy: y[i] + load[j] <= 1.0 for all j in N[i]
                double max_increase = limit - current_load[i];
                if (max_increase <= epsilon) continue;

                for (int k = 0; k < G->V[i].edgecount; k++) {
                    int neighbor = G->V[i].sibl[k];
                    double slack = limit - current_load[neighbor];
                    if (slack < max_increase) max_increase = slack;
                    if (max_increase <= epsilon) break;
                }

                if (max_increase > epsilon) {
                    y[i] += max_increase;
                    current_load[i] += max_increase;
                    for (int k = 0; k < G->V[i].edgecount; k++) {
                        int neighbor = G->V[i].sibl[k];
                        current_load[neighbor] += max_increase;
                    }
                    changed = true;
                }
            }

            // On subsequent passes, we can shuffle the order slightly
            // to find small pockets of slack missed by the degree heuristic.
            if (!changed) break;
            std::random_shuffle(order.begin(), order.end());
        }

        // 3. Final summation
        double lower_bound = 0.0;
        for (double val : y) {
            lower_bound += val;
        }

        return lower_bound;
    }
};

double get_mds_lower_bound(graph G) {
//    StrictMDSLowerBound solver;
//    return solver.compute(G);
    StrictMDSLowerBound2 solver;
    return solver.compute(G);

//    RelaxedLPRelaxationSolver solver;
//    return solver.compute_lower_bound(G);
}
