import time
import math

from .dinitz_blocking_flow import dinitz_blocking_flow
from .blocking_flow import compute_blocking_flow
from .visualize import visualize_graph

import networkx as nx
from abc import ABC
from networkx.algorithms.flow.utils import build_residual_network

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
    residual =  goldberg_rao_impl(G, s, t, capacity, residual, cutoff)

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
    sum_capacity = sum([attr[capacity] for u, v, attr in graph.edges(data=True)])
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    # F in the algorithm
    error_bound = sum_capacity
    INF = graph.graph.get("inf", float("inf"))

    # Lambda in the algorithm
    num_iterations_in_phase = int(math.ceil(min(math.sqrt(m), math.pow(n, 2 / 3))))
    total_routed_flow = 0
    prev_error_bound = float('inf')
    while error_bound >= 1:
        # Delta in the paper
        assert prev_error_bound == float("inf") or error_bound <= prev_error_bound // 2
        flow_to_route = math.ceil(error_bound / num_iterations_in_phase)

        for _ in range(num_iterations_in_phase):
            for u, v, attr in graph.edges(data=True):
                if is_at_capacity(graph, u, v):
                    # edges that are at capacity disappear from this algorithm on edges with strictly positive resid
                    # capacity are considered
                    if "_length" in attr:
                        del attr["_length"]
                    if "length" in attr:
                        del attr["length"]
                    continue
                attr["_length"] = 0 if (get_residual_cap(graph, u, v)) >= flow_to_route * 3 else 1
            construct_distance_metric(graph, end_node, length="_length")
            if graph.nodes[start_node]["distance"] == INF:
                graph.graph['flow_value'] = total_routed_flow
                return graph

            max_flow_upper_bound = min_canonical_cut(graph, start_node)
            if max_flow_upper_bound <= error_bound // 2:
                prev_error_bound = error_bound
                while max_flow_upper_bound <= error_bound // 2 and error_bound >= 1:
                    error_bound //= 2
                break

            for u, v, attr in graph.edges(data=True):
                if is_at_capacity(graph, u, v):
                    continue
                resid_cap = get_residual_cap(graph, u, v)
                resid_cap_reverse = get_residual_cap(graph, v, u)
                if 2 * flow_to_route <= resid_cap < 3 * flow_to_route <= resid_cap_reverse and graph.nodes[u]['distance'] == graph.nodes[v]['distance']:
                    attr['length'] = 0
                else:
                    attr['length'] = attr['_length']
            contracted_graph = construct_graph_contraction(graph, start_node, end_node)
            flow_routed = flow_to_route
            """
            if the start and end notes already in the same component, we don't need a blocking flow since 
            we can already route delta flow through that component.
            """
            if contracted_graph.graph["start_mapping"] != contracted_graph.graph["end_mapping"]:
                flow_routed = compute_blocking_flow(contracted_graph, contracted_graph.graph["start_mapping"], contracted_graph.graph["end_mapping"], flow_to_route)

            total_routed_flow += flow_routed

            if flow_routed == 0:
                graph.graph['flow_value'] = total_routed_flow
                return graph

            translate_flow_from_contraction_to_original(contracted_graph, graph, start_node, end_node, flow_routed )

    graph.graph['flow_value'] = total_routed_flow

    return graph


def is_at_capacity(graph, start_vert, end_vert, capacity="capacity"):
    """
    Checks to see if the current residual capacity of the edge is 0. For the purposes of this algorithm, this meansthe
    edge is non-existent in the graph.
    :param graph: Networkx X Graph
    :param start_vert: Starting vertex of edge
    :param end_vert: Ending vertex of edge
    :param capacity: Capacity attribute on the edge
    :return:
    """
    return get_residual_cap(graph, start_vert, end_vert, capacity) == 0


def get_residual_cap(graph, u, v, capacity="capacity", include_reverse_flow=True):
    """
    Returns the residual capacity of the graph
    :param graph:
    :param u:
    :param v:
    :param capacity:
    :param include_reverse_flow: Parameter to include additions to the residual capacity in reverse
    :return:
    """
    attr = graph.edges[u, v]
    val = attr[capacity] - attr["flow"]
    if include_reverse_flow:
        val += graph.edges[v, u]["flow"]
    return val


def update_flow(graph, u, v, flow_val, capacity="capacity"):

    attr = graph[u][v]
    attr_r = graph[v][u]
    # residual edge

    new_flow = flow_val + attr["flow"] - attr_r["flow"]

    if new_flow < 0:
        assert -new_flow <= attr_r["capacity"]
        attr["flow"] = 0
        attr_r["flow"] = new_flow

    else:
        assert new_flow <= attr["capacity"]
        attr["flow"] = new_flow
        attr_r["flow"] = 0


def min_canonical_cut(graph, start_node, distance="distance"):
    """
    Given a graph and a start node and a distance metric on the graph. Define a canonical cut as
    cuts between vertices of distance i and i + 1 where the cut-edges are such that one side has distance i
    and the other has i + 1.

    :param graph: Networkx Graph
    :param start_node: Node of max distance to compute canonical cuts for.
    :param distance: The attribute in graph.nodes that specifies the distance metric.
    :return:
    """
    max_distance = graph.nodes[start_node][distance]
    if max_distance == 0:
        return float("inf")
    distance_arr = [0 for _ in range(max_distance)]
    for u, v, attr in graph.edges(data=True):
        if not is_at_capacity(graph, u, v) and graph.nodes[u][distance] == graph.nodes[v][distance] + 1 and graph.nodes[v][distance] < max_distance:
            distance_arr[graph.nodes[v][distance]] += get_residual_cap(graph, u, v)
    return min(distance_arr)


def is_admissible_edge(graph, u, v, distance="distance", length="length"):
    return graph.nodes[u][distance] == graph.nodes[v][distance] + graph[u][v][length]


def construct_graph_contraction(graph, start_node, end_node):

    condensed_graph = condensation(graph)
    # construct in-tree and out-tree

    for scc, scc_attr in condensed_graph.nodes(data=True):

        rep_vertex = next(iter(scc_attr["members"]))
        if start_node in scc_attr["members"]:
            condensed_graph.graph["start_mapping"] = scc
        if end_node in scc_attr["members"]:
            condensed_graph.graph["end_mapping"] = scc

        if len(scc_attr["members"]) < 2:
            continue

        scc_attr["representative"] = rep_vertex
        # out tree construction
        children_queue = [rep_vertex]
        not_visited = set(scc_attr["members"])
        not_visited.remove(rep_vertex)
        while len(children_queue) != 0:
            curr_vertex = children_queue.pop()
            graph.nodes[curr_vertex]["out_children"] = []
            to_remove = set()
            for neighbor in not_visited:
                if not graph.has_edge(curr_vertex, neighbor) or is_at_capacity(graph, curr_vertex, neighbor):
                    continue
                if graph.edges[curr_vertex, neighbor]["length"] == 0:
                    to_remove.add(neighbor)
                    graph.nodes[curr_vertex]["out_children"].append(neighbor)
                    children_queue.append(neighbor)
            not_visited.difference_update(to_remove)
        # in tree construction
        children_queue = [rep_vertex]
        not_visited = set(scc_attr["members"])
        not_visited.remove(rep_vertex)
        while len(children_queue) != 0:
            curr_vertex = children_queue.pop()
            graph.nodes[curr_vertex]["in_children"] = []
            to_remove = set()

            for neighbor in not_visited:
                if not graph.has_edge(neighbor, curr_vertex) or is_at_capacity(graph, neighbor, curr_vertex):
                    continue
                if graph.edges[neighbor, curr_vertex]["length"] == 0:
                    to_remove.add(neighbor)

                    graph.nodes[curr_vertex]["in_children"].append(neighbor)
                    children_queue.append(neighbor)
            not_visited.difference_update(to_remove)

    return condensed_graph


def translate_flow_from_contraction_to_original(contraction, original, start_node, end_node, flow_routed):

    for node in original:
        original.nodes[node]["flow"] = 0
    original.nodes[start_node]["flow"] = flow_routed
    original.nodes[end_node]["flow"] = -flow_routed
    # assign flows to edges and the flow in - out for the contracted graph
    for contract_u, contract_v, contraction_edge in contraction.edges(data=True):
        remaining_edge_flow = contraction_edge["flow"]
        if remaining_edge_flow == 0:
            continue
        for original_edge in contraction_edge["members"]:
            
            start_vert, end_vert = original_edge
            flow_to_route = min(get_residual_cap(original, start_vert, end_vert), remaining_edge_flow)
            update_flow(original, start_vert, end_vert, flow_to_route)
            original.nodes[start_vert]["flow"] = original.nodes[start_vert].get("flow", 0) - flow_to_route
            original.nodes[end_vert]["flow"] = original.nodes[end_vert].get("flow", 0) + flow_to_route
            if flow_to_route == remaining_edge_flow:
                break
            remaining_edge_flow -= flow_to_route

    # assigns flows for each strongly connected component

    for vertex, attr in contraction.nodes(data=True):

        if len(attr["members"]) >= 2:
            rep_vertex = attr["representative"]
            flow_in = route_in_flow_tree(original, rep_vertex)
            flow_out = route_out_flow_tree(original, rep_vertex)
            if flow_out != flow_in:
                raise AssertionError("Flow in does not equal to flow out. If this error is raised, then something is wrong")


def route_in_flow_tree(graph, curr_vertex):
    flow = max(graph.nodes[curr_vertex]['flow'], 0)
    for child in graph.nodes[curr_vertex]["in_children"]:
        child_flow = route_in_flow_tree(graph, child)
        if child_flow != 0:
            update_flow(graph, child, curr_vertex, child_flow)
        flow += child_flow
    del graph.nodes[curr_vertex]["in_children"]
    return flow


def route_out_flow_tree(graph, curr_vertex):

    flow = max(-graph.nodes[curr_vertex]['flow'], 0)
    for child in graph.nodes[curr_vertex]["out_children"]:
        child_flow = route_out_flow_tree(graph, child)
        if child_flow != 0:
            update_flow(graph, curr_vertex, child, child_flow)
        flow += child_flow
    del graph.nodes[curr_vertex]["out_children"]
    del graph.nodes[curr_vertex]["flow"]
    return flow


def construct_distance_metric(graph, end_node, length='length'):
    """
    Given a length we compute a distance metric on the graph using the shortest path metric where the edge weights are
    the distances.

    :param graph: Networkx Graph
    :param end_node: The node which we compute distances from (0 distance node)
    :param length:
    :return:
    """
    INF = graph.graph.get("inf", float("inf"))
    for node in graph:
        graph.nodes[node]["distance"] = INF

    graph.nodes[end_node]["distance"] = 0
    n = graph.number_of_nodes()
    buckets = [set() for _ in range(n)]
    bucket_idx = 0
    buckets[0].add(end_node)
    while True:
        while bucket_idx < n and len(buckets[bucket_idx]) == 0:
            bucket_idx += 1
        if bucket_idx == n:
            break
        vertex = buckets[bucket_idx].pop()
        for neighbor in graph.predecessors(vertex):
            if is_at_capacity(graph, neighbor, vertex):
                continue
            length_neighbor = graph.edges[neighbor, vertex][length]
            dist_vertex = graph.nodes[vertex].get("distance")
            dist_neighbor = graph.nodes[neighbor].get("distance", None)
            
            if dist_neighbor == INF or dist_neighbor > dist_vertex + length_neighbor:
                if dist_neighbor != INF:
                    buckets[dist_neighbor].remove(neighbor)
                dist_neighbor = dist_vertex + length_neighbor
                graph.nodes[neighbor]["distance"] = dist_neighbor
                buckets[dist_neighbor].add(neighbor)
    return graph   


def length_strongly_connected_components(G):
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
                    if not is_at_capacity(G, v, w) and is_admissible_edge(G, v, w) and G[v][w]["length"] == 0 and w not in preorder:
                        queue.append(w)
                        done = False
                        break
                if done:
                    lowlink[v] = preorder[v]
                    for w in G[v]:
                        if not is_at_capacity(G, v, w) and is_admissible_edge(G, v, w) and G[v][w]["length"] == 0 and w not in scc_found:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
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
    distances = {}
    edge_members = {}
    C = nx.DiGraph()
    # Add mapping dict as graph attribute
    C.graph["mapping"] = mapping
    if len(G) == 0:
        return C
    i = 0
    for component in scc:
        members[i] = component
        distances[i] = G.nodes[list(component)[0]]["distance"]
        mapping.update((n, i) for n in component)
        i += 1
    number_of_components = i
    C.add_nodes_from(range(number_of_components))
    C.add_edges_from(
        (mapping[u], mapping[v]) for u, v in G.edges() if mapping[u] != mapping[v] and not is_at_capacity(G, u, v) and is_admissible_edge(G, u, v)
    )
    for u, v, attr in G.edges(data=True):
        if mapping[u] != mapping[v] and not is_at_capacity(G, u, v) and is_admissible_edge(G, u, v):
            if (mapping[u], mapping[v]) in edge_members.keys():
               edge_members[(mapping[u], mapping[v])]["members"].add((u, v))
               edge_members[(mapping[u], mapping[v])]["capacity"] += get_residual_cap(G, u, v)
               assert G[u][v]["length"] == edge_members[(mapping[u], mapping[v])]["length"]
            else:
                edge_members[(mapping[u], mapping[v])] = {}
                edge_members[(mapping[u], mapping[v])]["flow"] = 0
                edge_members[(mapping[u], mapping[v])]["members"] = {(u, v)}
                edge_members[(mapping[u], mapping[v])]["capacity"] = get_residual_cap(G, u, v)
                edge_members[(mapping[u], mapping[v])]["length"] = G[u][v]["length"]

    # Add a list of members (ie original nodes) to each node (ie scc) in C.
    nx.set_node_attributes(C, members, "members")
    nx.set_node_attributes(C, distances, "distance")

    # Add edge attributes to each node 
    nx.set_edge_attributes(C, edge_members)
    return C
