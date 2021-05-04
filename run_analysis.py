import random
from random import randint
import time

from goldberg_rao.algorithm import goldberg_rao

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms.flow import dinitz
from networkx.generators.classic import balanced_tree, barbell_graph, binomial_tree, \
                                        complete_graph, cycle_graph, path_graph, star_graph
from networkx.drawing.nx_pylab import draw_networkx

verbose = True
    
if verbose:
    def vprint(*args):
        for arg in args:
            print(arg,)
else:   
    vprint = lambda *a: None
            
def run_analysis_n_nodes(n, unit_cap, n_runs=5):
    print(f"Running analysis for {n} nodes...")
    graphs = [lambda: complete_graph(n), lambda: balanced_tree(2, int(round(np.log2(n)))), lambda: barbell_graph(int(round(n/2)), 1),
              lambda: binomial_tree(int(round(np.log2(n)))), lambda: cycle_graph(n),
              lambda: path_graph(n), lambda: star_graph(n)]
    results_gr = {}
    results_dinitz = {}
    for graph, name in zip(graphs, names):
        # Initialize both graphs
        G_dinitz = graph()
        G_gr = graph()

        total_time_gr = 0
        total_time_dinitz = 0
        for _ in range(n_runs):
            # Set random capacities of graph edges
            for u, v in G_dinitz.edges:
                cap = randint(1, 100) if not unit_cap else 1
                G_dinitz.edges[u, v]["capacity"] = cap
                G_gr.edges[u, v]["capacity"] = cap

            # Pick random start and end node
            start_node = randint(0, len(G_dinitz.nodes)-1)
            end_node = randint(0, len(G_dinitz.nodes)-1)
            while start_node == end_node: end_node = randint(0, len(G_dinitz.nodes)-1)

            # Run max-flow
            init_time = time.time()
            R_dinitz = dinitz(G_dinitz, start_node, end_node)
            total_time_dinitz += time.time() - init_time
        
            init_time = time.time()
            R_gr = goldberg_rao(G_gr, start_node, end_node)
            total_time_gr += time.time() - init_time
            
            # Check correctness
            d_mf = R_dinitz.graph["flow_value"]
            gr_mf = R_gr.graph["flow_value"]
            if d_mf != gr_mf:
                vprint(f"\t\t\tComputed max flow in {name} graph is {d_mf}, but goldberg_rao function computed {gr_mf}".upper())

        vprint(f"{name} with {n} nodes took {total_time_gr / n_runs} seconds with goldberg_rao")
        vprint(f"{name} with {n} nodes took {total_time_dinitz / n_runs} seconds with dinitz")
        results_gr[name] = total_time_gr / n_runs
        results_dinitz[name] = total_time_dinitz / n_runs

    return results_gr, results_dinitz

if __name__ == "__main__":
    names = ["complete_graph", "binary_tree", "balanced_barbell_graph", "binomial_tree", "cycle_graph",
             "path_graph", "star_graph"]
    
    output_filename = "analysis_results"
    n_nodes = [int(n) for n in np.logspace(1, 5, num=10)]
    unit_results_gr = {}
    unit_results_dinitz = {}
    #rand_cap_results_gr = {}
    #rand_cap_results_dinitz = {}
    
    for n in n_nodes:
        unit_results_gr[n], unit_results_dinitz[n] = run_analysis_n_nodes(n, unit_cap=True, n_runs=1)
        #rand_cap_results_gr[n], rand_cap_results_dinitz[n] = run_analysis_n_nodes(n, unit_cap=False, n_runs=1)

    for name in names:
        plt.plot(n_nodes, [unit_results_gr[n][name] for n in n_nodes], label=f"{name}, goldberg-rao")
        plt.plot(n_nodes, [unit_results_dinitz[n][name] for n in n_nodes], label=f"{name}, dinitz")
    plt.title("Runtime of goldberg rao vs. dinitz in seconds with unit capacities")
    plt.xlabel("Number of nodes")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.plot()

    """
    for name in names:
        plt.plot(n_nodes, [rand_cap_results_gr[n][name] for n in n_nodes], label=f"{name}, goldberg-rao")
        plt.plot(n_nodes, [rand_cap_results_dinitz[n][name] for n in n_nodes], label=f"{name}, dinitz")
    plt.title("Runtime of goldberg rao vs. dinitz in seconds with random capacities from 1-100")
    plt.xlabel("Number of nodes")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.plot()
    """  

