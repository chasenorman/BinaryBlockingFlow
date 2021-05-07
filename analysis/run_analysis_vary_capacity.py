import random
import math
from random import randint
import time
import pickle

from goldberg_rao.algorithm import goldberg_rao
from goldberg_rao.visualize import visualize_graph

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms.flow import dinitz
from networkx.generators.classic import balanced_tree, barbell_graph, binomial_tree, \
                                        complete_graph, cycle_graph, path_graph, star_graph
from networkx.generators.random_graphs import random_regular_graph
from networkx.drawing.nx_pylab import draw_networkx

verbose = True
random.seed(2)
    
if verbose:
    def vprint(*args):
        for arg in args:
            print(arg,)
else:   
    vprint = lambda *a: None

names = ["5-regular graph"]

def run_analysis_n_nodes(n, cap=1, n_runs=3):
    print(f"Running analysis for {n} nodes with {cap} capacities...")
    graphs = [lambda: random_regular_graph(5, n)]

    results_gr = {}
    results_dinitz = {}
    for graph, name in zip(graphs, names):
        # Initialize both graphs
        G_dinitz = graph()
        G_gr = G_dinitz.copy()

        total_time_gr = 0
        total_time_dinitz = 0
        for _ in range(n_runs):
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


        vprint(f"{name} with {len(G_gr.nodes)} nodes and capacity {cap} took {total_time_gr / n_runs} seconds with goldberg_rao")
        vprint(f"{name} with {len(G_dinitz.nodes)} nodes and capacity {cap} took {total_time_dinitz / n_runs} seconds with dinitz")
        results_gr[name] = total_time_gr / n_runs
        results_dinitz[name] = total_time_dinitz / n_runs

    return results_gr, results_dinitz

def main():
    output_filename_d = "vary_capacity_dinitz"
    output_filename_gr = "vary_capacity_gr"
    capacities = [int(n) for n in range(1, 201)]
    rand_cap_results_gr = {}
    rand_cap_results_dinitz = {}

    n = 100
    for cap in capacities:
        rand_cap_results_gr[cap], rand_cap_results_dinitz[cap] = run_analysis_n_nodes(n, cap=cap, n_runs=5)

    with open(output_filename_d, "wb") as f:
        pickle.dump(rand_cap_results_dinitz, f)

    with open(output_filename_gr, "wb") as f:
        pickle.dump(rand_cap_results_gr, f)

    for name in names:
        plt.plot(capacities, [rand_cap_results_gr[cap][name] for cap in capacities], label=f"{name}")
    plt.title("Runtime of Goldberg-Rao in seconds on a 5-regular graph of 100 nodes with varying capacities")
    plt.xlabel("Number of nodes")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.plot()
    plt.show()

    for name in names:
        plt.plot(capacities, [rand_cap_results_dinitz[cap][name] for cap in capacities], label=f"{name}")
    plt.title("Runtime of Dinitz in seconds on a 5-regular graph of 100 nodes with varying capacities")
    plt.xlabel("Number of nodes")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.plot()
    plt.show()

if __name__ == "__main__":
    main()
