import heapq
import networkx as nx
import matplotlib.pyplot as plt

def dijkstra(graph, start):
    # Inicialización
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}

    print(f"Estado inicial:")
    print(f"Cola de prioridad: {queue}")
    print(f"Distancias: {distances}")
    print(f"Predecesores: {predecessors}")
    print("-" * 50)

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

        # Imprimir el estado actual
        print(f"Visitando nodo: {current_node}")
        print(f"Cola de prioridad: {queue}")
        print(f"Distancias: {distances}")
        print(f"Predecesores: {predecessors}")
        print("-" * 50)

    return distances, predecessors

def draw_graph(graph, ax, distances=None, path=None):
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, ax=ax)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)

    if distances:
        distance_labels = {node: f"{node}\n{dist}" for node, dist in distances.items()}
        nx.draw_networkx_labels(G, pos, labels=distance_labels, font_color='red', ax=ax)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2, ax=ax)

def get_shortest_path(predecessors, start, end):
    path = []
    current = end
    while current:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    return path

# Definir el grafo
graph = {
    'A': {'B': 3, 'C': 1},
    'B': {'A': 3, 'D': 2, 'E': 2, 'G': 2, 'H': 3},
    'C': {'A': 1, 'F': 2,},
    'D': {'B': 2, 'E': 3},
    'E': {'B': 2, 'D': 3, 'H': 3},
    'F': {'G': 2, 'C': 2},
    'G': {'B': 2, 'F': 2},
    'H': {'B': 3, 'E': 3}
}

# Ejecutar el algoritmo
distances, predecessors = dijkstra(graph, 'A')
print("Distancias finales:", distances)
print("Predecesores finales:", predecessors)

# Obtener el camino más corto de A a D
shortest_path = get_shortest_path(predecessors, 'A', 'G')
print("Camino más corto de A a D:", shortest_path)

# Crear la figura y los ejes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Dibujar el grafo con distancias
draw_graph(graph, ax1, distances=distances)
ax1.set_title("Grafo con distancias")

# Dibujar el camino más corto
draw_graph(graph, ax2, distances=distances, path=shortest_path)
ax2.set_title("Camino más corto de A a G")

plt.show()
