import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

from main import Optimizer


matplotlib.use('TkAgg')


def generate_weighted_graph(num_cities):
    np.random.seed(40)
    graph = np.random.randint(1, 100, size=(num_cities, num_cities))
    np.fill_diagonal(graph, 0)
    graph = (graph + graph.T) // 2
    return graph


def draw_graph(distance_matrix, tour=None):
    G = nx.Graph()
    num_cities = distance_matrix.shape[0]

    for i in range(num_cities):
        G.add_node(i)

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    pos = nx.spring_layout(G, seed=42)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    if tour:
        path_edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
        path_edges.append((tour[-1], tour[0]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)

    plt.axis('off')
    plt.show()


cities = [30, 50, 100, 200]
for city in cities:
    graph = generate_weighted_graph(city)
    # draw_graph(graph)
    o = Optimizer(graph, generations=100)

    res_ant, best_tour = o.optimize(method="ant")
    res_gen = o.optimize(method="gen", population_size=100)

    # draw_graph(graph, tour=best_tour)

    plt.plot(res_ant, label="Мурахи")
    plt.plot(res_gen, label="Генетичний")
    plt.legend()
    plt.title(f"{city} вершин")
    plt.xlabel("Покоління")
    plt.ylabel("Відстань")
    plt.grid(True)
    plt.show()
