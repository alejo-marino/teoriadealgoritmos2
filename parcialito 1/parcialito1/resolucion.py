from math import log

from networkx import bridges, local_bridges, diameter, number_connected_components, average_shortest_path_length, \
    k_core, closeness_centrality, harmonic_centrality
from homofilia import *
from metricas import *
from modelos import *
from embeddings import *
import networkx as nx
from csv import reader


def generate_graph(csv):
    graph = nx.Graph()
    with open(csv, "r") as input_data:
        next(input_data, None)  # skip header
        csv_reader = reader(input_data)
        for row in csv_reader:
            graph.add_edge(row[0], row[1])
        input_data.close()
    return graph


def print_nodes_and_edges(graph):
    print("Number of nodes in given network: " + str(graph.number_of_nodes()))
    print("Number of edges in given network: " + str(graph.number_of_edges()))


def print_global_bridges(graph):
    puentes_globales = list(bridges(graph))
    indice_puentes = 0
    for puente in puentes_globales:
        indice_puentes += 1
        print("Puente " + str(indice_puentes) + ": '" + puente[0] + "'-'" + puente[1] + "'")


def print_local_bridges(graph):
    puentes_locales = list(local_bridges(graph))
    indice_puentes = 0
    for puente in puentes_locales:
        indice_puentes += 1
        print("Puente " + str(indice_puentes) + ": '" + puente[0] + "'-'" + puente[1] + "'")


"""
mapping function which returns true if the inputted node has the name of a 'rich'* country as its label
* rich countries are considered countries which have a >200% PPP GDP per capita, for reference refer to:
https://www.worldometers.info/gdp/gdp-per-capita/
"""
def is_rich(node):
    rich_countries = {"Qatar", "Macao", "Luxembourg", "Singapore", "Brunei", "Ireland", "United Arab Emirates", "Kuwait", "Switzerland", "San Marino", "Norway", "Hong Kong", "United States", "Iceland", "Netherlands", "Denmark", "Saudi Arabia", "Austria", "Germany", "Sweden", "Australia", "Belgium", "Bahrain", "Canada", "Finland", "United Kingdom", "France", "Japan", "Oman", "Italy", "Malta", "New Zealand", "Aruba", "Spain", "Israel", "South Korea", "Czech Republic", "Slovenia", "Cyprus"}
    return node in rich_countries


def is_poor(node):
    return not is_rich(node)


def is_red(node):
    red_countries = {"North Korea", "Cuba"}
    return node in red_countries


def main():
    graph = generate_graph('World - 2 columns.csv')
    print_nodes_and_edges(graph)
    print("Average degree of network: " + str(grado_promedio(graph)))
    print("----------Global Bridges----------", "\n")
    print_global_bridges(graph)
    print()
    print("----------Local Bridges----------", "\n")
    print_local_bridges(graph)
    print()
    print("----------Rich vs Poor countries proportions----------", "\n")
    print_nodes_and_edges(graph)
    proportions = proporcion_por_tipo(graph, is_rich)
    print("Proportion of Rich countries in the network: " + str(proportions[1]))
    print("Proportion of Poor countries in the network: " + str(proportions[0]))
    print("Proportion of edges that have a Rich country on one end, and a Poor country on its other: " + str(proporcion_cruzan_campo(graph, is_rich)))
    print("Proportion of Rich countries' edges which connect a Poor and a Rich country: " + str(proporcion_cruzan_campo_de_tipo(graph, True, is_rich)))
    print("Proportion of Poor countries' edges which connect a Poor and a Rich country: " + str(proporcion_cruzan_campo_de_tipo(graph, False, is_rich)))
    print()
    print("----------Red countries vs rest of the world proportions----------", "\n")
    print_nodes_and_edges(graph)
    proportions = proporcion_por_tipo(graph, is_red)
    print("Proportion of red countries in the network: " + str(proportions[1]))
    print("Proportion of not red countries in the network: " + str(proportions[0]))
    print("Proportion of edges that have a red country on one end, and a not red country on its other: " + str(proporcion_cruzan_campo(graph, is_red)))
    print("Proportion of red countries' edges which connect a not red and a red country: " + str(proporcion_cruzan_campo_de_tipo(graph, True, is_red)))
    print("Proportion of not red countries' edges which connect a not red and a red country: " + str(proporcion_cruzan_campo_de_tipo(graph, False, is_red)))
    print()
    print("----------Centrality analysis----------", "\n")
    #k_core_decomp = k_core(graph)
    #nx.draw(k_core_decomp)
    #plt.show()
    #harmonic = harmonic_centrality(graph)
    #nx.draw(harmonoc) #doesnt work since harmonic_centrality returns a dict{value=NAME, value=DEGREEOFCLOSENESS}
    #plt.show()
    print()
    print("----------Erdos-Renyi simulation----------", "\n")
    k = grado_promedio(graph)
    n = graph.number_of_nodes()
    erdos_renyi_graph = erdos_renyi(n, k)
    print("Average degree of original network: " + str(grado_promedio(graph)))
    print("Average degree of Erdos-Renyi simulation: " + str(grado_promedio(erdos_renyi_graph)))
    print("Diameter of original network: " + str(diameter(graph)))
    print("Diameter of Erdos-Renyi simulation: " + str(diameter(erdos_renyi_graph)))
    print("Clustering coefficient of original network: " + str(clustering(graph)[1]))
    print("Clustering coefficient of Erdos-Renyi simulation: " + str(clustering(erdos_renyi_graph)[1]))
    print("Average degree divided by number of nodes (which should be similar to the Erdos-Renyi clustering coefficient): " + str(k/n))
    print("Connected components in original network: " + str(number_connected_components(graph)))
    print("Connected components in Erdos-Renyi simulation: " + str(number_connected_components(erdos_renyi_graph)))
    print("Average path length in original network: " + str(average_shortest_path_length(graph)))
    print("Average path length in Erdos-Renyi simulation: " + str(average_shortest_path_length(erdos_renyi_graph)))
    #print(graficar_distribuciones(distribucion_grados(graph)))
    #print(graficar_distribuciones(distribucion_grados(erdos_renyi_graph)))
    print()
    print("----------Preferential Attachment simulation----------", "\n")
    graficar_distribuciones(distrubucion_ccdf(graph))
    print(alfa_preferential_attachment(graph, 8))
    preferential_graph = preferential_attachment(False, 2.25, n, k)
    graficar_distribuciones(distrubucion_ccdf(preferential_graph))
    print("Average degree of original network: " + str(grado_promedio(graph)))
    print("Average degree of Preferential Attachment simulation: " + str(grado_promedio(preferential_graph)))
    print("Diameter of original network: " + str(diameter(graph)))
    print("Diameter of Preferential Attachment simulation: " + str(diameter(preferential_graph)))
    print("Clustering coefficient of original network: " + str(clustering(graph)[1]))
    print("Clustering coefficient of Preferential Attachment simulation: " + str(clustering(preferential_graph)[1]))
    print("Average path length in original network: " + str(average_shortest_path_length(graph)))
    print("Average path length in Preferential Attachment simulation: " + str(average_shortest_path_length(preferential_graph)))
    print("Connected components in original network: " + str(number_connected_components(graph)))
    print("Connected components in Preferential Attachment simulation: " + str(number_connected_components(preferential_graph)))
    print()
    print("----------Anonymous walk simulation----------", "\n")
    anonymous_original = anoymous_walks(graph, 7)
    anonymous_er = anoymous_walks(erdos_renyi_graph, 7)
    anonymous_preferential = anoymous_walks(preferential_graph, 7)

    print("Cosine distance between original network and  Erdos-Renyi simulation: " + str(distancia_coseno(anonymous_original[1], anonymous_er[1])))
    print("Cosine distance between original network and Preferential Attachment simulation: " + str(distancia_coseno(anonymous_original[1], anonymous_preferential[1])))
    return


main()
