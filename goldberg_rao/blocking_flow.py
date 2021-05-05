import networkx as nx
from link_cut_tree.link_cut_tree import Node


class DynamicTree(Node):
    def update_augmentation(self):
        self.augmentation = self.value
        for c in self.children:
            if c:
                self.augmentation = min(self.augmentation, c)

    def __init__(self, name, value):
        self.name = name
        self.delta_sum = value
        super().__init__(value)

    def display_str(self):
        return f"{self.name}: {self.delta_sum}"

    def _rotate_up(self):
        p = self.parent
        c = self.children[1 - self.child_index()]

        if c:
            c.delta_sum += self.delta_sum

        self.delta_sum, p.delta_sum = self.delta_sum + p.delta_sum, -self.delta_sum

        super()._rotate_up()

    def _lc_replace_right_subtree(self, new_right_child):
        if self.right:
            self.right.delta_sum += self.delta_sum

        if new_right_child:
            new_right_child.delta_sum -= self.delta_sum

        super()._lc_replace_right_subtree(new_right_child)

    def lc_link(self, v):
        super().lc_link(v)

        v.delta_sum += self.delta_sum

    def get_sum(self):
        self.lc_expose()
        return self.delta_sum

    def set_value(self, value):
        self.lc_expose()
        delta = value - self.value
        self.delta_sum += delta
        self.value = value


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
    graph_nodes = graph.nodes

    for v in graph_nodes:
        graph_nodes[v]["blocked"] = False
        graph_nodes[v]["excess"] = 0
    for u,v in graph.edges:
        if u == start_node:
            graph[u][v]["flow"] = graph[u][v]["capacity"]
            graph_nodes[v]["excess"] = graph[u][v]["capacity"]
        else:
            graph[u][v]["flow"] = 0

    def push(u, v):
        delta = min(graph_nodes[u]["excess"], graph[u][v]["capacity"] - graph[u][v]["flow"])
        graph[u][v]["flow"] += delta
        graph_nodes[v]["excess"] += delta
        graph_nodes[u]["excess"] -= delta

    def pull(u, v):
        delta = min(graph_nodes[v]["excess"], graph[u][v]["flow"])
        graph[u][v]["flow"] -= delta
        graph_nodes[v]["excess"] -= delta
        graph_nodes[u]["excess"] += delta

    def discharge(v):
        if not graph_nodes[v]["blocked"]:
            for w in graph.successors(v):
                if not graph_nodes[w]["blocked"]:
                    push(v, w)
                    if graph_nodes[v]["excess"] == 0:
                        return

        # we can assert that either v is already blocked,
        # or all outgoing edges are full or connected to a blocked node
        graph_nodes[v]["blocked"] = True

        for u in graph.predecessors(v):
            pull(u, v)
            if graph_nodes[v]["excess"] == 0:
                return

        raise AssertionError('How did we get here?')

    L = finger_tree(graph, start_node, end_node)
    v = L.first_active()
    while v is not None:
        discharge(v)
        if graph_nodes[v]["blocked"]:
            L.move_to_front(v)
        v = L.first_active()

    return limit_flow(graph, start_node, end_node, maximum_flow_to_route)

def flow_value(graph, start_node, end_node):
    X = 0
    for u in graph.predecessors(end_node):
        X += graph[u][end_node]["flow"]
    for w in graph.successors(end_node):
        X -= graph[end_node][w]["flow"]
    return X

def limit_flow(graph, start_node, end_node, maximum_flow_to_route):
    X = flow_value(graph, start_node, end_node)
    if X <= maximum_flow_to_route:
        return X
    graph_nodes = graph.nodes
    graph_nodes[end_node]["excess"] = X - maximum_flow_to_route
    topo = reversed(list(nx.algorithms.dag.topological_sort(graph)))
    for v in topo:
        for u in graph.predecessors(v):
            if graph_nodes[v]["excess"] == 0:
                break
            delta = min(graph[u][v]["flow"], graph_nodes[v]["excess"])
            graph_nodes[v]["excess"] -= delta
            graph_nodes[u]["excess"] += delta
            graph[u][v]["flow"] -= delta
    return maximum_flow_to_route

def is_flow(graph, start_node, end_node):
    for u,v,attr in graph.edges(data=True):
        if attr["flow"] > attr["capacity"]:
            return False
        if attr["flow"] < 0:
            return False
    for v in graph.nodes:
        if v == start_node or v == end_node:
            continue
        in_flow = 0
        out_flow = 0
        for u in graph.predecessors(v):
            in_flow += graph.edges[u, v]["flow"]
        for w in graph.successors(v):
            out_flow += graph.edges[v, w]["flow"]
        if in_flow != out_flow:
            return False
    return True

if __name__ == "__main__":
    import random
    for i in range(100):
        G = nx.gnp_random_graph(100, 0.5, directed=True)
        DAG = nx.DiGraph([(u,v,{'capacity':random.randint(0,10)}) for (u,v) in G.edges() if u<v])
        #visualize.visualize_graph(DAG, weight="capacity", filename="1.png")

        graph = compute_blocking_flow(DAG, 0, 99, 150)
        # l = list(graph.edges(data=True))
        # for u, v, attr in l:
        #     if attr["flow"] == 0:
        #         graph.remove_edge(u, v)
        # for u in graph.nodes:
        #     pass
        #     # print(graph.nodes[u]["excess"])
        # visualize.visualize_graph(graph, weight="flow", filename="2.png")

        visited = [False] * (graph.number_of_nodes())

        queue = []
        queue.append(0)
        visited[0] = True

        while queue:
            s = queue.pop(0)
            for i in graph.successors(s):
                if graph.edges[s, i]["capacity"] - graph.edges[s, i]["flow"] != 0:
                    if visited[i] == False:
                        queue.append(i)
                        visited[i] = True

        if not visited[99]:
            print("BLOCKING", end=', ')
        else:
            print("NOT BLOCKING", end=', ')

        if is_flow(graph, 0, 99):
            print("FLOW", end = ', ')
        else:
            print("NOT FLOW", end = ', ')

        print(flow_value(graph, 0, 99))
