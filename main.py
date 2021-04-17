import networkx

def III_2(U):
    pass

def III_6(U):
    pass

def st_mincut(G):
    pass

def sparsification_algorithm():
    pass

G = networkx.DiGraph()

def goldberg_rao(G, s, t):
    U = max(G.edges, key=lambda e: e['weight'])
    m = G.number_of_edges()
    n = G.number_of_nodes()
    F = U*m
    f = G.create_empty_copy() # says f=0 in the paper but proceeds to use as flow.
    L = min(m**0.5, n**1.5)
    while F >= 1:
        D = F/(2*L)
        for _ in range(1, 5*L):
            l = G.create_empty_copy()
            for u, v, a in f.edges:
                if a['weight'] <= D:
                    l.add_edge(u, v) # presuming 0 is unconnected
            dl = networkx.single_target_shortest_path_length(l, t)

            GA = ... #construct admissible graph
            GprimeA = ... # contact strongly-connected components induced by 0-length arcs in GA into G′A
            f_tilde = ... # run BLOCKING-FLOW or MAX-FLOW for |f|=∆/4 to find flow f ̃
            f_hat = ... # let fˆ be f ̃ routed through contracted SCCs
            f = ... # augment f along fˆ

        F = F/2

