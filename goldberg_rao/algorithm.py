import networkx
from abc import ABC
from networkx.algorithms.flow.utils import build_residual_network
import math

"""
Implementation of the Goldberg Rao Algorithm: 

See the following resources to learn more about it: 

- https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/beyond%20the%20flow.pdf
- https://www.cs.princeton.edu/courses/archive/fall06/cos528/handouts/Goldberg-Rao.pdf
- http://cs.ucls.uchicago.edu/~rahulmehta/papers/GoldbergRao-Notes.pdf

"""


def goldberg_rao(G, s, t, capacity="capacity", residual=None, value_only=False, cutoff=None):

    """Find a maximum single-commodity flow using Dinitz' algorithm.

    This function returns the residual network resulting after computing
    the maximum flow. See below for details about the conventions
    NetworkX uses for defining residual networks.

    This algorithm has a running time of $O(n^2 m)$ for $n$ nodes and $m$
    edges [1]_.


    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute called
        'capacity'. If this attribute is not present, the edge is
        considered to have infinite capacity.

    s : node
        Source node for the flow.

    t : node
        Sink node for the flow.

    capacity : string
        Edges of the graph G are expected to have an attribute capacity
        that indicates how much flow the edge can support. If this
        attribute is not present, the edge is considered to have
        infinite capacity. Default value: 'capacity'.

    residual : NetworkX graph
        Residual network on which the algorithm is to be executed. If None, a
        new residual network is created. Default value: None.

    value_only : bool
        If True compute only the value of the maximum flow. This parameter
        will be ignored by this algorithm because it is not applicable.

    cutoff : integer, float
        If specified, the algorithm will terminate when the flow value reaches
        or exceeds the cutoff. In this case, it may be unable to immediately
        determine a minimum cut. Default value: None.

    Returns
    -------
    R : NetworkX DiGraph
        Residual network after computing the maximum flow.

    Raises
    ------
    NetworkXError
        The algorithm does not support MultiGraph and MultiDiGraph. If
        the input graph is an instance of one of these two classes, a
        NetworkXError is raised.

    NetworkXUnbounded
        If the graph has a path of infinite capacity, the value of a
        feasible flow on the graph is unbounded above and the function
        raises a NetworkXUnbounded.

    See also
    --------
    :meth:`maximum_flow`
    :meth:`minimum_cut`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    Notes
    -----
    The residual network :samp:`R` from an input graph :samp:`G` has the
    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair
    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a
    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists
    in :samp:`G`.

    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v]['capacity']`
    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists
    in :samp:`G` or zero otherwise. If the capacity is infinite,
    :samp:`R[u][v]['capacity']` will have a high arbitrary finite value
    that does not affect the solution of the problem. This value is stored in
    :samp:`R.graph['inf']`. For each edge :samp:`(u, v)` in :samp:`R`,
    :samp:`R[u][v]['flow']` represents the flow function of :samp:`(u, v)` and
    satisfies :samp:`R[u][v]['flow'] == -R[v][u]['flow']`.

    The flow value, defined as the total flow into :samp:`t`, the sink, is
    stored in :samp:`R.graph['flow_value']`. If :samp:`cutoff` is not
    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such
    that :samp:`R[u][v]['flow'] < R[u][v]['capacity']` induces a minimum
    :samp:`s`-:samp:`t` cut.

    Examples
    --------
    >>> from networkx.algorithms.flow import dinitz

    The functions that implement flow algorithms and output a residual
    network, such as this one, are not imported to the base NetworkX
    namespace, so you have to explicitly import them from the flow package.

    >>> G = nx.DiGraph()
    >>> G.add_edge("x", "a", capacity=3.0)
    >>> G.add_edge("x", "b", capacity=1.0)
    >>> G.add_edge("a", "c", capacity=3.0)
    >>> G.add_edge("b", "c", capacity=5.0)
    >>> G.add_edge("b", "d", capacity=4.0)
    >>> G.add_edge("d", "e", capacity=2.0)
    >>> G.add_edge("c", "y", capacity=2.0)
    >>> G.add_edge("e", "y", capacity=3.0)
    >>> R = dinitz(G, "x", "y")
    >>> flow_value = nx.maximum_flow_value(G, "x", "y")
    >>> flow_value
    3.0
    >>> flow_value == R.graph["flow_value"]
    True

    References
    ----------
    .. [1] Dinitz' Algorithm: The Original Version and Even's Version.
           2006. Yefim Dinitz. In Theoretical Computer Science. Lecture
           Notes in Computer Science. Volume 3895. pp 218-240.
           http://www.cs.bgu.ac.il/~dinitz/Papers/Dinitz_alg.pdf

    """
    residual, =  goldberg_rao_impl(G, s, t, capacity, residual, cutoff)

    if value_only:
        return residual.graph['flow_value']
    residual.graph["algorithm"] = "goldberg-rao"
    return residual


def goldberg_rao_impl(G, s, t, capacity="capacity", residual=None, cutoff=None):
    if s not in G:
        raise nx.NetworkXError(f"node {str(s)} not in graph")
    if t not in G:
        raise nx.NetworkXError(f"node {str(t)} not in graph")
    if s == t:
        raise nx.NetworkXError("source and sink are the same node")

    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual

    # Zero-out
    for u in R:
        for e in R[u].values():
            e["flow"] = 0

    start_node = s
    end_node = t
    graph = R
    
    max_capacity =  max(G.edges, key=lambda e: e[capacity])
    m = G.number_of_edges()
    n = G.number_of_nodes()

    # F in the algorithm
    error_bound = n * max_capacity

    # Lambda in the algorithm
    phases = int(math.ceil(min(math.sqrt(m), math.pow(n, 2 / 3))))



    while error_bound >= 1:
        # Delta in the paper
        flow_to_route = math.ceil(error_bound / phases) # ceil?

        for u, v, attr in graph.edges: # TODO should we be iterating over reverse edges?
            attr["length"] = 0 if (get_residual_cap(graph, u, v)) >= 3 * flow_to_route else 1

        dials_algorithm(graph, end_node)
        
        for _ in range(phases):
            
            for u, v, attr in graph.edges: 
                resid_cap = get_residual_cap(graph, u, v)
                resid_cap_reverse = get_residual_cap(graph, v, u)
                if resid_cap >= 2 * flow_to_route and resid_cap < 3 * flow_to_route and resid_cap_reverse >= 3 * flow_to_route and u['distance'] == v['distance']:
                    attr['length_bar'] = 0
                else:
                    attr['length_bar'] = attr['length']

            A, S = contraction(graph) 
            


            if min_canonical_cut(graph) <= error_bound // 2:
                break 
            

        
        error_bound //= 2
            

        
        
        
    return 1


def get_residual_cap(graph, u, v):
    attr = graph.edges[u, v]
    return attr["capacity"] - attr["flow"] + graph.edges[v, u]["flow"]

def goldberg_rao_phase(graph, start_node, end_node, error_bound, num_phase_iterations):
    pass

def min_canonical_cut(graph, distance="distance"):
    # TODO why is dl(s) a bound on distance?
    # TODO there are nodes of distance 0, why are they not included?
    INF = graph.graph["inf"]
    max_distance = max(G.nodes, key=lambda v: v[distance] if v[distance] != INF else -1)
    if max_distance == -1:
        return ...
        

# adds "distance" to each vertex based on "length"
# sets disconnected edges to infinity distance TODO
def dial_algorithm(graph, end_node):

    INF = graph.graph["inf"]
    for node in graph:
        node["distance"] = INF

    end_node["distance"] = 0
    n =  graph.number_of_nodes()
    buckets = [set() for _ in range(n)]

    bucket_idx = 0
    while True:
        while len(buckets[bucket_idx]) == 0 and bucket_idx < n:
            idx += 1
        if idx == n:
            break

        vertex = buckets[bucket_idx][0]
        buckets[bucket_idx].remove(vertex)
        for neighbor in graph.predecessors(vertex):
            length = graph[neighbor][vertex]['length']
            
            dist_vertex = vertex.get("distance")
            dist_neighbor = neighbor.get("distance", None)
            
            if dist_neighbor == INF or dist_neighbor > dist_vertex + length:
                if dist_neighbor != -1:
                    buckets[dist_neighbor].remove(neighbor)
                dist_neighbor = dist_vertex + length
                neighbor["distance"] = dist_neighbor
                buckets[dist_neighbor].add(neighbor)
    return graph   
            
                






                    







        


    
    

    