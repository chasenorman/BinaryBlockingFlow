import networkx as nx
import visualize

class dynamic_tree():
    def find_root(self, v):
        pass

    def find_size(self, v):
        pass

    def find_value(self, v):
        pass

    def find_min(self, v):
        pass

    def change_value(self, v, x):
        pass

    def link(self, v, w):
        pass

    def cut(self, v):
        pass

class finger_tree():

    def __init__(self, graph, start_node, end_node):
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.L = list(nx.algorithms.dag.topological_sort(graph))

    def first_active(self):
        for v in self.L:
            if v != self.start_node and v != self.end_node and \
                    self.graph.nodes[v]["excess"] > 0:
                return v

    def move_to_front(self, v):
        self.L.remove(v)
        self.L.insert(0, v)


def compute_blocking_flow(graph, start_node, end_node, maximum_flow_to_route):
    for v in graph.nodes:
        graph.nodes[v]["blocked"] = False
        graph.nodes[v]["excess"] = 0
    for u,v in graph.edges:
        if u == start_node:
            # TODO if there are residuals, these are not set properly.
            graph.edges[u, v]["flow"] = graph.edges[u, v]["capacity"]
            graph.nodes[v]["excess"] = graph.edges[u, v]["capacity"]
        else:
            graph.edges[u, v]["flow"] = 0

    def push(u, v):
        # TODO if there are residuals, these are not set properly.
        delta = min(graph.nodes[u]["excess"], graph.edges[u, v]["capacity"] - graph.edges[u, v]["flow"])
        graph.edges[u, v]["flow"] += delta
        #graph[v, u]["on_blocking_flow"] -= delta
        graph.nodes[v]["excess"] += delta
        graph.nodes[u]["excess"] -= delta

    def pull(u, v):
        # TODO if there are residuals, these are not set properly.
        delta = min(graph.nodes[v]["excess"], graph.edges[u, v]["flow"])
        graph.edges[u, v]["flow"] -= delta
        #graph[v, u]["on_blocking_flow"] += delta
        graph.nodes[v]["excess"] -= delta
        graph.nodes[u]["excess"] += delta

    def discharge(v):
        if not graph.nodes[v]["blocked"]:
            for w in graph.successors(v):
                if not graph.nodes[w]["blocked"]:
                    push(v, w)
                    if graph.nodes[v]["excess"] <= 0:
                        return

        # we can assert that either v is already blocked,
        # or all outgoing edges are full or connected to a blocked node
        graph.nodes[v]["blocked"] = True

        for u in graph.predecessors(v):
            pull(u, v)
            if graph.nodes[v]["excess"] <= 0:
                return

        raise AssertionError('How did we get here?')

    L = finger_tree(graph, start_node, end_node)
    v = L.first_active()
    while v is not None:
        discharge(v)
        if graph.nodes[v]["blocked"]:
            L.move_to_front(v)
        v = L.first_active()

    return limit_flow(graph, start_node, end_node, maximum_flow_to_route)


def limit_flow(graph, start_node, end_node, maximum_flow_to_route):
    l = list(graph.edges(data=True))
    for u,v,attr in l:
        if attr["flow"] == 0:
            graph.remove_edge(u,v)
    for u in graph.nodes:
        pass
        #print(graph.nodes[u]["excess"])
    visualize.visualize_graph(graph, weight="flow", filename="2.png")
    pass

if __name__ == "__main__":
    import random
    G = nx.gnp_random_graph(7, 0.5, directed=True)
    DAG = nx.DiGraph([(u,v,{'capacity':random.randint(0,10)}) for (u,v) in G.edges() if u<v])
    visualize.visualize_graph(DAG, weight="capacity", filename="1.png")
    compute_blocking_flow(DAG, 0, 6, 0)