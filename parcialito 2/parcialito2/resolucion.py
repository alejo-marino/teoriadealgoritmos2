from csv import reader
from parcialito2.motifs.calculos import calcular_motifs
from parcialito2.parcialito_utils import *
import networkx as nx
from community import community_louvain



rich_countries = ["Qatar", "Macao", "Luxembourg", "Singapore", "Brunei", "Ireland", "United Arab Emirates",
                  "Kuwait", "Switzerland", "San Marino", "Norway", "Hong Kong", "United States", "Iceland",
                  "Netherlands", "Denmark", "Saudi Arabia", "Austria", "Germany", "Sweden", "Australia", "Belgium",
                  "Bahrain", "Canada", "Finland", "United Kingdom", "France", "Japan", "Oman", "Italy", "Malta",
                  "New Zealand", "Aruba", "Spain", "Israel", "South Korea", "Czech Republic", "Slovenia", "Cyprus"]
n_nodes = 229


def louvain(graph):
    communities = community_louvain.best_partition(graph)
    labels = {node: graph.nodes()[node]['country'] for node in graph.nodes()}
    unique_coms = np.unique(list(communities.values()))
    cmap = {
        0: 'maroon',
        1: 'teal',
        2: 'black',
        3: 'orange',
        4: 'green',
        5: 'yellow'
    }
    countries = nx.get_node_attributes(graph, "country")

    node_cmap = [cmap[v] for _, v in communities.items()]
    pos = nx.spring_layout(graph)
    #nx.draw(graph, pos, node_size=75, alpha=0.8, node_color=node_cmap)
    available_colors = {0: '#E9D758', 1: '#297373', 2: '#ff8552', 3: '#888888', 4: '#000000'}
    colors = [available_colors[communities[node]] for node in graph.nodes()]

    plt.figure(figsize=(15, 15))
    plt.title("Louvain communities")
    nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes(), node_color=colors, alpha=0.7, node_size=700, linewidths=2)
    nx.draw_networkx_edges(graph, pos, width=0.3, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, labels=labels)
    plt.show()

    real_communities = {}   # formatting to make this function return data usable by other functions
    for node in communities:
        if communities[node] not in real_communities:
            real_communities[communities[node]] = []
        real_communities[communities[node]].append(countries[node])

    # plt.savefig("communities_louvain.html", format='svg', dpi=300) for some reason doesn't work will have to rely on .plot() for visualization
    return real_communities


def generate_graph(csv):
    graph = nx.Graph()
    with open(csv, "r") as input_data:
        next(input_data, None)  # skip header
        csv_reader = reader(input_data)
        for row in csv_reader:
            graph.add_edge(row[0], row[1])
        input_data.close()
    return graph


"""
Receives a dictionary with the following format: key = label of community, value = list of nodes in said community.
criteria is a list of nodes to look for, essentially this function will look through all the given communities and
will return a percentage representative of the amount of nodes of the given criteria that is in each community.
"""
def analyze_community(communities, criteria):
    communities_criteria_overlap = {} # percentage of nodes meeting criteria relative to nodes in all communities
    node_total = n_nodes
    #for community in communities:
    #    node_amount += len(communities[community])

    for community in communities:
        nodes_meeting_criteria = 0
        for node in communities[community]:
            if node in criteria:
                nodes_meeting_criteria += 1
        communities_criteria_overlap[community] = (nodes_meeting_criteria / node_total) * 100

    return communities_criteria_overlap


def main():
    graph = generate_graph('World - 2 columns.csv')
    number_country_relation = {}
    counter = 0
    for node in graph.nodes():
        number_country_relation[node] = counter
        counter += 1

    indexed_graph = nx.relabel_nodes(graph, number_country_relation, copy=True) # try using convert_node_labels_to_integers later if possible
    key_list = list(number_country_relation.keys())
    values_list = list(number_country_relation.values())
    for node in indexed_graph:
        indexed_graph.nodes[node]['country'] = key_list[values_list.index(node)]

    print(len(graph.edges), len(indexed_graph.edges))
    print(len(graph.nodes), len(indexed_graph.nodes))
    print("number communities graph: " + str(get_number_communities(graph)))    # get_number_communities seems to return 1 for this graph so i'll need to execute cluster_and_plot() with arbitrary values and analyze them
    print("number communities indexed_graph: " + str(get_number_communities(indexed_graph)))

    # Punto 1 [Comunidades]

    # Spectral clustering for given graph with 2, 3 and 4 communities
    communities_k2 = cluster_and_plot(indexed_graph, title='Comunidades en red de viajes aereos entre paises (k=2)', save=True, file_name="communities_k2.html", n_clusters=2)
    communities_k3 = cluster_and_plot(indexed_graph, title='Comunidades en red de viajes aereos entre paises (k=3)', save=True, file_name="communities_k3.html", n_clusters=3)
    communities_k4 = cluster_and_plot(indexed_graph, title='Comunidades en red de viajes aereos entre paises (k=4)', save=True, file_name="communities_k4.html", n_clusters=4)


    # Disclaimer: it should be remarked that in the previous study, with the elected metrics rich countries were about ~16.15 of the nodes of the network.
    print("========== Spectral Clustering ==========")
    print("========== k = 2 ==========")
    rich_countries_percentage_k2 = analyze_community(communities_k2, rich_countries)  # each community and the percentage of rich countries in them
    for community in rich_countries_percentage_k2:
        print("Percentage of rich countries in " + str(community) + " community: " + str(rich_countries_percentage_k2[community]))
    print("========== k = 3 ==========")
    rich_countries_percentage_k3 = analyze_community(communities_k3, rich_countries)
    for community in rich_countries_percentage_k3:
        print("Percentage of rich countries in " + str(community) + " community: " + str(rich_countries_percentage_k3[community]))
    print("========== k = 4 ==========")
    rich_countries_percentage_k4 = analyze_community(communities_k4, rich_countries)
    for community in rich_countries_percentage_k4:
        print("Percentage of rich countries in " + str(community) + " community: " + str(rich_countries_percentage_k4[community]))
    print("===========================")

    print("========== Louvain ==========")
    # Community detection using Louvain's algorithm (above parts may need to be commented for proper visualization)
    louvain_communities = louvain(indexed_graph)
    rich_countries_percentage_louvain = analyze_community(louvain_communities, rich_countries)

    for community in rich_countries_percentage_louvain:
        print("Percentage of rich countries in " + str(community) + " community: " + str(rich_countries_percentage_louvain[community]))

    print()
    print("Disclaimer: it should be remarked that in the previous study, with the elected metrics rich countries were about ~16.15 of the nodes of the network.")

    # Manually assembled list of countries from one of the communities returned by Louvain's algorithm from a past execution
    # Nota: cree esto porque la representacion grafica del algoritmo de Louvain cambia y a veces los nodos salen muy agrupados, decidi quedarme con la imagen de la consigna porque crei que la distribucion de los nodos era bastante buena en terminos de visualizacion
    american_louvain_community = ("Chile", "Mexico", "Brazil", "United States", "Canada", "Argentina", "Costa Rica", "Cuba", "Ecuador", "Colombia", "Peru", "Panama", "Dominican Republic", "Venezuela", "Jamaica", "El Salvador", "Guatemala", "Belize", "Falkland Islands", "Nicaragua", "Honduras", "Paraguay", "Bolivia", "Cayman Islands", "Saint Pierre and Miquelon", "Montserrat", "Anguilla", "American Samoa", "Dominica", "British Virgin Islands", "Saint Vincent and the Grenadines", "French Guiana", "Grenada", "Saint Kitts and Nevis", "Turks and Caicos Islands", "Guyana", "Bermuda", "Martinique", "Trinidad and Tobago", "Haiti", "Aruba", "Uruguay", "Antigua and Barbuda", "Guadeloupe", "Puerto Rico", "Virgin Islands", "Saint Lucia", "Suriname", "Bahamas", "Barbados", "Netherlands Antilles")
    # print(len(american_louvain_community)) == 51 > (229 * 0.2) = 51 > 22.9 * 2 = 51 > 45.8, contains more than 20% of the total nodes in the network

    american_graph = graph.copy()

    for node in graph:
        if node not in american_louvain_community:
            american_graph.remove_node(node)

    indexed_american_graph = nx.relabel_nodes(american_graph, number_country_relation, copy=True) # try using convert_node_labels_to_integers later if possible

    for node in indexed_american_graph:
        indexed_american_graph.nodes[node]['country'] = key_list[values_list.index(node)]

    american_louvain_subcommunities = louvain(indexed_american_graph)

    # Punto 2 [Motifs]
    # this part was executed in google's colab
    """
    motifs_lenght = 5
    motifs = calcular_motifs(indexed_american_graph, motifs_lenght)
    print("========== Motifs ==========")
    print(motifs)
    
    distribucion = distribucion_grados(indexed_american_graph)
    promedios_a, stds_a = motif_grafo_eleatorios(lambda: configuration_model(distribucion), MAX_NODOS_MOTIFS, iters=15)
    significant_p = significance_profile(motifs_graph, promedios_a, stds_a)
    print("Significant Profile red americana: ", significant_p)
    
    graficar_significant_profile(significant_p, 'Red America')
    
    extract_roles_and_plot(indexed_american_graph ,title='Roles en America', save=True, file_name='roles_america.svg', big=True)
    """
    # Punto 3 [Roles]
    extract_roles_and_plot(indexed_american_graph, title='Roles en America', save=True, file_name='roles_america.svg', big=True)

main()
