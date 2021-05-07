from collections import defaultdict
import random
import math
from random import randint
import time
import pickle

from goldberg_rao.algorithm import goldberg_rao

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms.flow import dinitz
from networkx.generators.classic import balanced_tree, barbell_graph, binomial_tree, \
                                        complete_graph, cycle_graph, path_graph, star_graph
from networkx.generators.random_graphs import random_regular_graph
from networkx.drawing.nx_pylab import draw_networkx

verbose = True
    
if verbose:
    def vprint(*args):
        for arg in args:
            print(arg,)
else:   
    vprint = lambda *a: None

names = ["d-regular graph"]

def run_analysis_n_nodes(n, d, cap, n_runs=3):
    print(f"Running analysis for {n} nodes and {d} regularity...")
    graphs = [lambda: random_regular_graph(d, n)]

    results_gr = {}
    results_dinitz = {}
    for graph, name in zip(graphs, names):
        # Initialize both graphs
        G_dinitz = graph()
        G_gr = G_dinitz.copy()

        total_time_gr = 0
        total_time_dinitz = 0
        for _ in range(n_runs):
            # Set random capacities of graph edges
            for u, v in G_dinitz.edges:
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

        vprint(f"{d}-regular graph with {len(G_gr.nodes)} nodes, cap {cap} and {len(G_gr.edges)} edges took {total_time_gr / n_runs} seconds with goldberg_rao")
        vprint(f"{d}-regular graph with {len(G_dinitz.nodes)} nodes, cap {cap} and {len(G_dinitz.edges)} edges took {total_time_dinitz / n_runs} seconds with dinitz")
        results_gr = total_time_gr / n_runs
        results_dinitz = total_time_dinitz / n_runs

    return results_gr, results_dinitz

def main():
    output_filename_d = "vary_nodes_dinitz"
    output_filename_gr = "vary_nodes_goldberg_rao"
    n_edges = [1000, 2000, 3000]
    cap = 1
    unit_results_gr = {n: {} for n in n_edges}
    unit_results_dinitz = {n: {} for n in n_edges}
    
    for n in n_edges:
        for d in range(1, n):
            n_nodes = 2*n/d
            if n_nodes == int(n_nodes) and d <= n_nodes:
                unit_results_gr[n][n_nodes], unit_results_dinitz[n][n_nodes] = run_analysis_n_nodes(int(n_nodes), d, 1, n_runs=5)

    with open(output_filename_d, "wb") as f:
        pickle.dump(unit_results_dinitz, f)

    with open(output_filename_gr, "wb") as f:
        pickle.dump(unit_results_gr, f)

    for n in n_edges:
        plt.plot(unit_results_gr[n].keys(), unit_results_gr[n].values(), label=f"d-regular graph with {n} edges")
    plt.title("Runtime of Goldberg-Rao on d-regular graphs in seconds vs. number of nodes")
    plt.xlabel("Number of edges")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.plot()
    plt.show()

    for n in n_edges:
        plt.plot(unit_results_dinitz[n].keys(), unit_results_dinitz[n].values(), label=f"d-regular graph with {n} edges")
    plt.title("Runtime of Dinitz on d-regular graphs in seconds vs. number of nodes")
    plt.xlabel("Number of edges")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.plot()
    plt.show()


if __name__ == "__main__":
    main()
