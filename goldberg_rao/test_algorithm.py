from random import randint

from .algorithm import goldberg_rao

import networkx as nx
from networkx.algorithms.flow import dinitz
from networkx.generators.classic import balanced_tree, barbell_graph, binomial_tree, \
                                        complete_graph, cycle_graph, path_graph, star_graph
from networkx.drawing.nx_pylab import draw_networkx

def test_correctness():
    # note: in the future can restrict all graphs to have the same number of nodes, for analysis
    graphs = [lambda: basic_graph(), lambda: balanced_tree(3, 5), lambda: barbell_graph(6, 7),
              lambda: binomial_tree(10), lambda: complete_graph(30), lambda: cycle_graph(50),
              lambda: path_graph(200), lambda: star_graph(200)]
    names = ["basic_graph", "balanced_tree", "barbell_graph", "binomial_tree", "complete_graph", "cycle_graph",
             "path_graph", "star_graph"]
    for graph, name in zip(graphs, names):
        print(f"Testing graph {name}...")
        
        # Initialize both graphs
        G_dinitz = graph()
        G_gr = graph()

        # Set random capacities of graph edges
        for u, v in G_dinitz.edges:
            cap = randint(1, 5)
            G_dinitz.edges[u, v]["capacity"] = cap
            G_gr.edges[u, v]["capacity"] = cap

        # Pick random start and end node
        start_node = randint(0, len(G_dinitz.nodes)-1)
        end_node = randint(0, len(G_dinitz.nodes)-1)
        while start_node == end_node: end_node = randint(0, len(G_dinitz.nodes)-1)

        # Run max-flow
        R_dinitz = dinitz(G_dinitz, start_node, end_node)
        R_gr = goldberg_rao(G_gr, "s", "t")

        # Check correctness
        d_mf = R_dinitz.graph["flow_value"]
        gr_mf = R_gr.graph["flow_value"]
        assert d_mf == gr_mf, f"Computed max flow in {name} graph is {d_mf}, but goldberg_rao function computed {gr_mf}"
    
def basic_graph():
    G = nx.DiGraph()
    G.add_edge(0, 1, capacity=3.0)
    G.add_edge(0, 2, capacity=1.0)
    G.add_edge(1, 3, capacity=3.0)
    G.add_edge(2, 3, capacity=5.0)
    G.add_edge(2, 4, capacity=4.0)
    G.add_edge(4, 5, capacity=2.0)
    G.add_edge(3, 6, capacity=2.0)
    G.add_edge(5, 6, capacity=3.0)
    
    return G



    
