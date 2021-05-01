import networkx


def dinitz_blocking_flow(graph, start_node, end_node, maximum_flow_to_route):
    """
    Runs the dinitz blocking flow.

    :param graph: NetworkX Graph
    :param start_node: Node where we start.
    :param end_node: Node where we end.
    :param maximum_flow_to_route: Maximum amount of flow that we can route.
    :return:
    """
    total_flow = 0
    while total_flow < maximum_flow_to_route:
        curr_path = [start_node]
        flow_val = blocking_flow_helper(graph, start_node, curr_path, end_node, maximum_flow_to_route - total_flow)
        if flow_val < 0:
            return total_flow
        total_flow += flow_val
    return total_flow


def blocking_flow_helper(graph, curr_node, curr_path, end_node, max_flow_left):
    if curr_node == end_node:
        flow_val = max_flow_left
        for idx, node in enumerate(curr_path):
            if idx == 0:
                continue
            flow_val = min(flow_val, get_residual_cap(graph, curr_path[idx - 1], node, include_reverse_flow=False))
            graph[curr_path[idx - 1]][node]["on_blocking_flow"] = True
        for idx, node in enumerate(curr_path):
            if idx == 0:
                continue
            graph[curr_path[idx - 1]][node]["flow"] += flow_val
            if not get_residual_cap(graph, curr_path[idx - 1], node, include_reverse_flow=False) == 0:
                graph[curr_path[idx - 1]][node]["is_visited"] = False
        return flow_val

    for neighbor in graph.successors(curr_node):
        if graph[curr_node][neighbor].get("is_visited", False) or get_residual_cap(graph, curr_node, neighbor,
                                                                                   include_reverse_flow=False) == 0:
            continue
        curr_path.append(neighbor)
        path_found = blocking_flow_helper(graph, neighbor, curr_path, end_node, max_flow_left)
        if path_found >= 0:
            return path_found
    curr_path.pop()
    return -1


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

