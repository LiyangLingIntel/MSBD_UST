
import numpy as np
from queue import Queue


def get_input(file_path: str):

    adj_matrix = []
    num_points = 0

    with open(file_path, 'r') as f:
        input_values = f.readlines()
        num_points = int(input_values[0])
        for line in input_values[1:]:
            adj_matrix.append([int(i) for i in line.strip()])

    return num_points, np.array(adj_matrix)


def build_adj_list(n, adj_maxtrix):

    adj_list = []
    for i in range(n):
        adj_list.append([])
        for j in range(n):
            if adj_maxtrix[i, j] == 1:
                adj_list[i].append(j)

    return adj_list


def bfs_betweeness(n, root_node, adj_list) -> dict:
    WHITE = 1
    GRAY = 2
    BLACK = -1

    # init
    paths_from_root = dict.fromkeys(list(range(n)), 0)
    paths_from_root[root_node] = 1
    edges = {}

    node_status = dict.fromkeys(list(range(n)), WHITE)

    # 我们需要知道 u 邻接矩阵中的点是自己的父节点，兄弟节点还是子节点,
    # 以及自节点的数量

    brothers = {root_node: ''}
    childrens = {}

    q = Queue()
    q.put(root_node)
    node_status[root_node] = GRAY

    while not q.empty():
        u = q.get()

        if u not in brothers:
            brothers = childrens.copy()
            childrens = {}

        child_list = []
        for v in adj_list[u]:
            if node_status[v] == BLACK:
                continue
            if v in brothers:
                continue
            child_list.append(v)
            childrens[v] = ''

        for v in child_list:
            path_weight = paths_from_root[u] / len(child_list)
            paths_from_root[v] += path_weight
            if node_status[v] != GRAY:
                q.put(v)
                node_status[v] = GRAY

            edges[tuple(sorted([u, v]))] = path_weight

        node_status[u] = BLACK

    return edges


def get_betweeness_map(n, adj_list):

    # init
    betweeness_map = bfs_betweeness(n, 0, adj_list)
    for i in range(1, n):
        for edge, paths in bfs_betweeness(n, i, adj_list).items():
            if edge not in betweeness_map:
                betweeness_map[edge] = paths
            else:
                betweeness_map[edge] += paths

    for edge, paths in list(betweeness_map.items()):
        betweeness_map[edge] = round(paths / 2, 5)

    return betweeness_map


def get_clusters(n, edges:dict, adj_matrix):

    # edges_list = sorted(list(edges.items()), key=lambda x: x[1], reverse=True)

    adj_m = adj_matrix.copy()
    max_edges = [e for e in edges.keys() if edges[e] == max(list(edges.values()))]
    retain_edges = edges.copy()
    # retain_edges = [e for e in edges.keys() if edges[e] != max(list(edges.values()))]

    for edge in max_edges:
        retain_edges.pop(edge)
        adj_m[edge[0]][edge[1]] = 0
        adj_m[edge[1]][edge[0]] = 0

    clusters = []
    counted_list = []
    for i in range(n):
        if i in counted_list:
            continue
        c = [i] + [j for j in range(n) if adj_m[i][j] == 1]
        clusters.append(c)
        counted_list.extend(c)
    return clusters, adj_m, retain_edges
        


############################################################
# calculate modularity
############################################################

def get_modularity(n, communities, adj_matrix):
    
    m2 = np.sum(adj_matrix)
    mod = 0
    for comm in communities:
        for i in comm:
            for j in comm:
                mod += adj_matrix[i][j] - sum(adj_matrix[i])*sum(adj_matrix[j]) / m2
    mod /= m2
    return round(mod, 4)


if __name__ == '__main__':

    N, adj_matrix = get_input('./asgn1_input.txt')
    adj_list = build_adj_list(N, adj_matrix)
    paths = get_betweeness_map(N, adj_list)
    
    decompositions = []

    # First part:
    print('First Part:')
    print('network decomposition:')
    adj_m = adj_matrix.copy()
    while len(paths) != 0:
        clusters, adj_m, paths = get_clusters(N, paths, adj_m)
        print(clusters)
        decompositions.append(clusters)

    # Second part
    print('Second Part:')
    mod_list = []
    for c_i in range(len(decompositions)):
        cluster = decompositions[c_i]
        modularity = get_modularity(N, cluster, adj_matrix)
        print(f'{len(cluster)} clusters: modularity {modularity}')

        if modularity > 0.3 and modularity < 0.7:
            mod_list.append((c_i, modularity))

    if not mod_list:
        print(f'no optimal structure')
    else:
        best_ci = sorted(mod_list, key=lambda x: x[1])[-1][0]
        print(f'optimal structure: {decompositions[best_ci]}')
