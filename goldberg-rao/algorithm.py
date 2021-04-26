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
    
    max_capacity =  max(graph.edges, key=lambda e: e[capacity])
    m = graph.number_of_edges()
    n = graph.number_of_nodes()

    # F in the algorithm
    error_bound = n * max_capacity

    # Lambda in the algorithm
    phases = int(math.ceil(min(math.sqrt(m), math.pow(n, 2 / 3))))
    total_routed_flow = 0

    while error_bound >= 1:
        # Delta in the paper
        flow_to_route = math.ceil(error_bound / phases) # ceil?

        for u, v, attr in graph.edges: # TODO should we be iterating over reverse edges?
            attr["length"] = 0 if (get_residual_cap(graph, u, v)) >= 3 * flow_to_route else 1

        construct_distance_metric(graph, end_node)
        
        for _ in range(phases):
            
            for u, v, attr in graph.edges: 
                resid_cap = get_residual_cap(graph, u, v)
                resid_cap_reverse = get_residual_cap(graph, v, u)
                if resid_cap >= 2 * flow_to_route and resid_cap < 3 * flow_to_route and resid_cap_reverse >= 3 * flow_to_route and u['distance'] == v['distance']:
                    attr['length_bar'] = 0
                else:
                    attr['length_bar'] = attr['length']

            contracted_graph = construct_graph_contraction(graph)
            total_routed_flow += compute_blocking_flow(contracted_graph, start_node, end_node, flow_to_route) 
            translate_flow_from_contraction_to_original(contracted_graph, graph)
            
            if min_canonical_cut(graph) <= error_bound // 2:
                break 
            
        error_bound //= 2
    return 1


def get_residual_cap(graph, u, v, capacity="capacity"):
    attr = graph.edges[u, v]
    return attr[capacity] - attr["flow"] + graph.edges[v, u]["flow"]

def update_flow(graph, u, v, flow_val, capacity="capacity"):
    if flow_val < 0:
        update_flow(graph, v, u, -flow_val)
        return
    attr = graph.edges[u, v]

    # residual edge
    if attr[capacity] == 0:
        graph.edges[v, u]["flow"] -= flow_val
    
    # forward edge
    else if flow_val <= attr[capacity] - attr["flow"]:
        attr["flow"] += flow_val
    
    else: 
        raise AssertionError("Cannot update flow value here")
    



def goldberg_rao_phase(graph, start_node, end_node, error_bound, num_phase_iterations):
    pass

def min_canonical_cut(graph, start_node, end_node, distance="distance"):
    INF = graph.graph["inf"]
    max_distance = start_node[distance]
    distance_arr = [0 for _ in range(max_distance)]

    for u, v, attr in G.edges: 

        if is_admissible_edge(u, v): 
            distance_arr[u[distance]] += get_residual_cap(graph, u, v)
    
    return min(distance_arr)

def is_admissible_edge(u, v, distance="distance"):

    return u[distance] == v[distance] - 1


def compute_blocking_flow(graph, start_node, end_node, maximum_flow_to_route):
    
    curr_node = start_node
    curr_path = [start_node]
    total_flow = 0
    while total_flow < maximum_flow_to_route:
        flow_val = blocking_flow_helper(graph, start_node, curr_path, end_node, maximum_flow_to_route - total_flow)
        if not flow_val:
            return
        total_flow += flow_val
    return total_flow
 
    
def blocking_flow_helper(graph, curr_node, curr_path, end_node, max_flow_left):

    if curr_node == end_node: 
        flow_val = max_flow_left
        for idx, node in enumerate(curr_path):
            if idx == 0:
                continue
            flow_val = min(flow_val, get_residual_cap(graph, curr_path[idx-1], node))
            graph[curr_path[idx -1]][node]["on_blocking_flow"] = True

        for idx, node in enumerate(curr_path):
            if idx == 0:
                continue

            graph[curr_path[idx -1]][node]["flow"] = flow_val

        return flow_val


    for neighbor in graph.successors(curr_node):
        if not is_admissible_edge(curr_node, neighbor) or graph[curr_node][neighbor].get("is_visited", False):
            continue
        curr_path.append(neighbor)
            path_found = blocking_flow_helper(graph, neighbor, curr_path, end_node, max_flow_left)
        if path_found >= 0:
            return
    
    curr_path.pop()
    return -1



def construct_graph_contraction(graph, start_node, end_node):

    condensed_graph = condensation(graph)
    
    # construct in-tree and out-tree

    for scc in condensed_graph:

        if len(scc["members"]) < 2:
            continue

        rep_vertex = scc["members"][0]

        if start_node in scc["members"]:
            rep_vertex = start_node
        else if end_node in scc["members"]:
            rep_vertex = end_node
        
        scc["representative"] = rep_vertex
         # out tree construction 

        children_queue = [rep_vertex]
        visited = set()
        while len(children_queue) != 0:
            curr_vertex = children_queue.pop()
            curr_vertex["out_children"] = []
            visited.add(curr_vertex)
            for neighbor in graph.successors(curr_vertex):
                if neighbor not in visited and graph[curr_vertex, neighbor]["length"] == 0:
                    curr_vertex["out_children"].append(neighbor)
        # in tree construction
        children_queue = [rep_vertex]
        visited = set()
        while len(children_queue) != 0:
            curr_vertex = children_queue.pop()
            curr_vertex["in_children"] = []
            visited.add(curr_vertex)
            for neighbor in graph.successors(curr_vertex):
                if neighbor not in visited and graph[neighbor, curr_vertex]["length"] == 0:
                    curr_vertex["in_children"].append(neighbor)


def translate_flow_from_contraction_to_original(contraction, original):

    # assign flows to edges and the flow in - out for the contracted graph
    for contraction_edge in contraction.edges: 
        remaining_edge_flow = contraction_edge["flow"]
        if not contraction_edge["on_blocking_flow"]:
            continue
        for original_edge in contraction_edge["members"]:
            
            start_vert, end_vert = original_edge
            flow_to_route = min(get_residual_cap(original, start_vert, end_vert), remaining_edge_flow)
            update_flow(original, start_vert, end_vert, flow_to_route)
            start_vert["flow"] -= flow_to_route
            end_vert["flow"] += flow_to_route
            if flow_to_route == remaining_edge_flow:
                break
            remaining_edge_flow -= flow_to_route
        
    # assigns flows for each strongly connected component

    for vertex in contraction: 

        if len(vertex["members"]) < 2:
            rep_vertex = vertex["representative"]
            flow_in = route_in_flow_tree(rep_vertex)
            flow_out = route_out_flow_tree(rep_vertex)
            assert flow_to_route == flow_in
    


def route_in_flow_tree(graph, curr_vertex):

    flow = max(curr_vertex['flow'], 0)
    for child in curr_vertex["in_children"]:
        child_flow = route_in_flow_tree(child)
        if child_flow != 0
            update_flow(graph, child, curr_vertex, child_flow)
    del curr_vertex["in_children"]
    return flow

def route_out_flow_tree(graph, curr_vertex):

    flow = max(-curr_vertex["flow"], 0)
    for child in curr_vertex["out_children"]:
        child_flow = route_out_flow_tree(child)
        if child_flow != 0
            update_flow(graph, curr_vertex, child, child_flow)
    del curr_vertex["out_children"]
    del curr_vertex["flow"]
    return flow


# adds "distance" to each vertex based on "length"
# sets disconnected edges to infinity distance TODO
def construct_distance_metric(graph, end_node):

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
            

def length_strongly_connected_components(G):
    """Generate nodes in strongly connected components of graph.

    Parameters
    ----------
    G : NetworkX Graph
        A directed graph.

    Returns
    -------
    comp : generator of sets
        A generator of sets of nodes, one for each strongly connected
        component of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    Generate a sorted list of strongly connected components, largest first.

    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph())
    >>> nx.add_cycle(G, [10, 11, 12])
    >>> [
    ...     len(c)
    ...     for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    ... ]
    [4, 3]

    If you only want the largest component, it's more efficient to
    use max instead of sort.

    >>> largest = max(nx.strongly_connected_components(G), key=len)

    See Also
    --------
    connected_components
    weakly_connected_components
    kosaraju_strongly_connected_components

    Notes
    -----
    Uses Tarjan's algorithm[1]_ with Nuutila's modifications[2]_.
    Nonrecursive version of algorithm.

    References
    ----------
    .. [1] Depth-first search and linear graph algorithms, R. Tarjan
       SIAM Journal of Computing 1(2):146-160, (1972).

    .. [2] On finding the strongly connected components in a directed graph.
       E. Nuutila and E. Soisalon-Soinen
       Information Processing Letters 49(1): 9-14, (1994)..

    """
    preorder = {}
    lowlink = {}
    scc_found = set()
    scc_queue = []
    i = 0  # Preorder counter
    for source in G:
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                done = True
                for w in G[v]:
                    if G[v, w]["length"] == 0 and w not in preorder:
                        queue.append(w)
                        done = False
                        break
                if done:
                    lowlink[v] = preorder[v]
                    for w in G[v]:
                        if G[v, w]["length"] == 0 and w not in scc_found:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
                            k["flow"] = 0
                            scc.add(k)
                        scc_found.update(scc)
                        yield scc
                    else:
                        scc_queue.append(v)

                

def condensation(G, scc=None):

    if scc is None:
        scc = length_strongly_connected_components(G)
    mapping = {}
    members = {}
    edge_members = {}
    C = nx.DiGraph()
    # Add mapping dict as graph attribute
    C.graph["mapping"] = mapping
    if len(G) == 0:
        return C
    for i, component in enumerate(scc):
        members[i] = component
        mapping.update((n, i) for n in component)

    number_of_components = i + 1
    C.add_nodes_from(range(number_of_components))
    C.add_edges_from(
        (mapping[u], mapping[v]) for u, v in G.edges() if mapping[u] != mapping[v]
    )
    for u, v, attr in G.edges:
        if mapping[u] != mapping[v]:
            if (mapping[u], mapping[v]) in edge_members.keys():
               edge_members[(mapping[u], mapping[v])]["members"].add((u, v, attr))
               edge_members[(mapping[u], mapping[v])]["capacity"] += get_residual_cap(G, u, v)
            else:
                edge_members[(mapping[u], mapping[v])] = {}
                edge_members[(mapping[u], mapping[v])]["flow"] = 0
                edge_members[(mapping[u], mapping[v])]["on_blocking_flow"] = 0
                edge_members[(mapping[u], mapping[v])]["members"] = set((u, v, attr))
                edge_members[(mapping[u], mapping[v])]["capacity"] = get_residual_cap(G, u, v)
        
        
    # Add a list of members (ie original nodes) to each node (ie scc) in C.
    nx.set_node_attributes(C, members, "members")

    # Add edge attributes to each node 
    nx.set_edge_attributes(C, edge_members)
    return C




                    







        


    
    

    