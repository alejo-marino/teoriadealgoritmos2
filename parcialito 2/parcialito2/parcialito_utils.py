# %%

import networkx as nx
import matplotlib.pyplot as plt
import json
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import numpy as np
import seaborn as sns

sns.set_style('darkgrid')


# %%

def load_graph(path):
    with open(path) as archivo:
        graph = nx.readwrite.json_graph.node_link_graph(json.load(archivo), multigraph=False,
                                                        attrs={'source': "source", "target": "target",
                                                               "key": "key", "link": "links"})
        return graph


# %%

# g = load_graph('Dataset/starwars-full-interactions-allCharacters.json')


# %% md

## Formato Gephi

# %%

def gephi_file(g, path):
    with open(path + "_edges.csv", 'w') as s:
        s.write('Source,Target,Type,Id,Label,timeset,Weight\n')
        seen = set()
        id = 0
        for v in g:
            for w in g.neighbors(v):
                if w in seen:
                    continue
                s.write(str(v) + "," + str(w) + ",Undirected," + str(id) + ",,,1\n")
                id += 1
            seen.add(v)

    labels = {node: g.nodes()[node]['name'] for node in g}
    with open(path + "_nodes.csv", "w") as s:
        s.write("Id,Label,timeset\n")
        for v in g:
            s.write(str(v) + "," + labels[v] + ",\n")


# %%

# gephi_file(g, 'starwars')


# %% md

## Dibujos base

# %%

def plot_graph(g, title, color, border_color, draw_weights=False, save=False, file_name=''):
    labels = {node: g.nodes()[node]['name'] for node in g}
    pos = nx.kamada_kawai_layout(g)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title(title, fontdict={'fontsize': 18})

    nx.draw_networkx_nodes(g, pos, nodelist=g.nodes(), alpha=0.7, node_color=color, node_size=700,
                           edgecolors=border_color, linewidths=2)
    nx.draw_networkx_edges(g, pos, width=0.3, alpha=0.5)
    nx.draw_networkx_labels(g, pos, labels=labels)
    if draw_weights:
        nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, 'weight'), rotate=False)

    if save:
        plt.savefig(file_name, format='svg', dpi=300)


# %%

# plot_graph(g, 'Escenas compartidas en Star Wars', '#c879f2', '#b241f0', save=False)


# %% md

## Análisis de métricas

# %%

def plot_degree_dist(G, title, save=False, file_name=''):
    fig, ax = plt.subplots(figsize=(10, 10))
    degrees = [G.degree(n) for n in G.nodes()]
    # print(degrees)
    plt.xlabel('Grado', fontdict={'fontsize': 20})
    plt.ylabel('P(k)', fontdict={'fontsize': 20})
    plt.title(title, fontdict={'fontsize': 18})
    plot = sns.countplot(degrees, ax=ax)
    plt.show()

    if save:
        plot.get_figure().savefig(file_name, format='svg', dpi=300)


# %%

def analyze(G):
    largest_connected_component_nodes = max(nx.connected_components(G), key=len)
    largest_connected_component = G.subgraph(largest_connected_component_nodes)

    print(f"Connected components: {nx.number_connected_components(G)}")
    print(f"Proportion of nodes in the giant component: {len(largest_connected_component) * 100 / len(G)}")
    print(
        f"Diameter of largest connected component: {nx.algorithms.distance_measures.diameter(largest_connected_component)}")
    print(f"Average clustering coefficient: {nx.algorithms.cluster.average_clustering(G)}")


# %% md

## Analisis Macroscópicos

# %%

# analyze(g)

# %%

# We keep just the giant component
# g = g.subgraph(max(nx.connected_components(g), key=len))

# %%

# Analisis de grados
# plot_degree_dist(g, 'Distribución de grados de Star Wars', save=False, file_name='img/degree_dist_4.svg')

# %% md

## Embeddings

# %% md

### A nivel grafo: Anonymous Walks

# %%

import sys

sys.path.append("/Users/mbuchwald/Documents/SocialNetworks/Material")
from embeddings import anoymous_walks

# caminos, emb = anoymous_walks(g, 7)
# print(emb)

# %% md

## Análisis Mesoscópicos

# %% md

### Motifs

# %%

from motifs.calculos import calcular_motifs, significance_profile, motif_grafo_eleatorios
from metricas import distribucion_grados
from modelos import configuration_model
from motifs.graficos import graficar_significant_profile

# %%

MAX_NODOS_MOTIFS = 5
# motifs = calcular_motifs(g, MAX_NODOS_MOTIFS)
# print(motifs)

# %%

#dist = distribucion_grados(g)
#promedios, stds = motif_grafo_eleatorios(lambda: configuration_model(dist), MAX_NODOS_MOTIFS, iters=20)
#SP = significance_profile(motifs, promedios, stds)
#print("SP:", SP)

# %%

#graficar_significant_profile(SP, 'Star Wars')

# %% md

### Roles

# %%

from graphrole import RecursiveFeatureExtractor, RoleExtractor


# %%

def extract_roles_and_plot(G, title='', save=False, file_name='', big=False):
    feature_extractor = RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()
    role_extractor = RoleExtractor(n_roles=None)
    role_extractor.extract_role_factors(features)
    labels = {node: G.nodes()[node]['country'] for node in G.nodes()}

    available_colors = {'role_0': '#E9D758', 'role_1': '#297373', 'role_2': '#ff8552', 'role_3': '#888888',
                        'role_4': '#00aa00', 'role_5': '#aaaa00', 'role_6': '#aa0000', 'role_7': '#0000aa'}

    colors = [available_colors[role_extractor.roles[node]] for node in G.nodes()]

    pos = nx.kamada_kawai_layout(G)
    if big:
        plt.figure(figsize=(15, 15))
    else:
        plt.figure(figsize=(10, 10))
    plt.title(title)

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=colors, alpha=0.7, node_size=700, linewidths=2)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels)

    if save:
        plt.savefig(file_name, format='svg', dpi=300)


# %%

#extract_roles_and_plot(g, title='Roles en Star Wars', save=False, file_name='img/roles_ep4.svg')


# %% md

### Deteccion de comunidades

# %%

def get_number_communities(G):
    lap_mat = nx.linalg.normalized_laplacian_matrix(G)
    #print("lap_mat: " + str(lap_mat))
    eigenvalues, eigenvectors = np.linalg.eig(lap_mat.toarray())
    #print("eigenval: " + str(eigenvalues))
    #print("eigenval: " + str(eigenvectors))
    sortedEigenvalues = sorted(eigenvalues)[:10]
    #print("sorted eigenval: " + str(sortedEigenvalues))
    #print("argmax: " + str(np.argmax(np.diff(sortedEigenvalues))))
    return np.argmax(np.diff(sortedEigenvalues)) + 1


# %%

def plot_eigenvalues(G, title='', save=False, file_name=''):
    fig, ax = plt.subplots(figsize=(10, 10))
    lap_mat = nx.linalg.normalized_laplacian_matrix(G)
    eigenvalues, eigenvectors = np.linalg.eig(lap_mat.toarray())
    sortedEigenvalues = sorted(eigenvalues)[:10]
    plt.xlabel('k', fontdict={'fontsize': 20})
    plt.ylabel('Eigenvalue', fontdict={'fontsize': 20})
    plt.title(title, fontdict={'fontsize': 18})
    plt.plot(sortedEigenvalues, marker='o')

    if save:
        plt.savefig(file_name, format='svg', dpi=300)


# %%

def cluster_and_plot(G, title='', save=False, file_name='', n_clusters=None):
    communities = {}
    labels = {node: G.nodes()[node]['country'] for node in G.nodes()}

    if not n_clusters:
        n_clusters = get_number_communities(G)
    sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
    sc.fit(nx.to_numpy_array(G))

    countries = nx.get_node_attributes(G, "country")
    for node in G.nodes:
        if sc.labels_[node] not in communities:
            communities[sc.labels_[node]] = []
        communities[sc.labels_[node]].append(countries[node])

    available_colors = {0: '#E9D758', 1: '#297373', 2: '#ff8552', 3: '#888888', 4: '#000000'}
    colors = [available_colors[sc.labels_[node]] for node in G.nodes()]

    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(20, 20))
    plt.title(title)

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=colors, alpha=0.7, node_size=700, linewidths=2)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels)

    if save:
        plt.savefig(file_name, format='svg', dpi=300)

    return communities


def cluster_and_plot_sw(G, title='', save=False, file_name='', n_clusters=None):
    labels = {node: G.nodes()[node]['name'] for node in G.nodes()}

    for n, label in labels.items():
        print(n, label)
    if not n_clusters:
        n_clusters = get_number_communities(G)
    sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
    sc.fit(nx.to_numpy_array(G))

    available_colors = {0: '#E9D758', 1: '#297373', 2: '#ff8552', 3: '#888888', 4: '#000000'}
    colors = []
    counter = 0
    for node in G.nodes():
        colors.append(available_colors[sc.labels_[node]])
        counter += 1


    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(10, 10))
    plt.title(title)

    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=colors, alpha=0.7, node_size=700, linewidths=2)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels)

    if save:
        plt.savefig(file_name, format='svg', dpi=300)


# %%

#plot_eigenvalues(g, title='Valor de los primeros 10 autovalores para Star Wars', save=False)

# %%

#cluster_and_plot(g, title='Comunidades en Star Wars', save=False)

# %%
