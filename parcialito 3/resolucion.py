from _csv import reader

import pandas as pd
import csv
import pickle
import os
import networkx as nx

gamma1 = 0.5
gamma2 = 0.5

"""
receives a path to a .csv file and returns it's contents as a .pkl file
"""


def csv_to_pkl(csv_path):
    pickle_network = pd.read_csv(csv_path)
    return pickle_network


def generate_graph(csv):
    graph = nx.Graph()
    print("number of nodes:" + str(graph.number_of_nodes()))
    with open(csv, "r") as input_data:
        next(input_data, None)  # skip header
        csv_reader = reader(input_data)
        counter = 0
        for row in csv_reader:
            if row[0] not in graph.nodes:
                graph.add_node(row[0], fairness=1)
            if row[1] not in graph.nodes:
                graph.add_node(row[1], goodness=1)
            graph.add_edge(row[0], row[1], rating=(float(row[2]) - 3) / 2, date=row[3], reliability=1)
            counter += 1
            if counter % 100000 == 0:
                print("number of edges:" + str(graph.number_of_edges()))
        input_data.close()
    return graph


def getFairness(element):
    return element[1]


def main():
    # select .csv file as input
    csv_file = "data/ratings_Electronics_network.csv"

    graph = generate_graph(csv_file)

    # sanitize products and reviewer's lists
    fairness_list = list(graph.nodes.data("fairness"))

    for node in fairness_list:  # it may seem like program hangs here but it just takes a long time
        if node[1] is None:
            fairness_list.remove(node)  # remove products from fairness_list
    print("Done cleansing fairness_list")

    goodness_list = list(graph.nodes.data("goodness"))
    for node in goodness_list:
        if node[1] is None:
            goodness_list.remove(node)  # remove reviewers frmo goodness_list
    print("Done cleansing goodness_list")

    reliability_list = list(graph.edges.data("reliability"))
    print("Done creating reliability_list")

    iter = 0
    total_iters = 100
    while iter < total_iters:
        # recalculate fairness for each reviewer (fairness = sumatory(reliability of each of the reviewer's reviews) / reviewer's out degree)
        for reviewer in fairness_list:
            reviewer_id = reviewer[0]
            total_reliability = 0
            reviewer_out_deg = 0
            for product in graph.neighbors(reviewer_id):
                product_id = product[0]
                edge_attr = graph.get_edge_data(reviewer_id, product_id)
                total_reliability += edge_attr["reliability"]
                reviewer_out_deg += 1

            new_fairness = total_reliability / reviewer_out_deg
            nx.set_node_attributes(graph, {reviewer_id: new_fairness}, name="fairness")

        # recalculate goodness for each product (goodness = sumatory(reliability of each reviewer that reviewed this product * score of the reviewer) / product's in-degree)
        for product in goodness_list:
            product_id = product[0]
            total_reliability_times_score = 0
            product_in_deg = 0
            for reviewer in graph.neighbors(product_id):
                reviewer_id = reviewer[0]
                edge_attr = graph.get_edge_data(reviewer_id, product_id)
                reliability = edge_attr["reliability"]
                score = edge_attr["rating"]
                total_reliability_times_score += reliability * score
                product_in_deg += 1

            new_goodness = total_reliability_times_score / product_in_deg
            nx.set_node_attributes(graph, {product_id: new_goodness}, name="goodness")

        # recalculate reliability for each review (reliability = (1 / gamma1 + gamma2) * [gamma1 * fairness_reviewer + gamma2 * (1 - {review's score - product's goodness} / 2)]
        for review in reliability_list:
            reviewer_id = review[
                0]  # aca por algun motivo creo que review[0] a veces es un product_id y review[1] un reviewer_id, sinceramente no entiendo donde esta el error que haga que pase esto
            product_id = review[1]
            edge_attributes = graph.get_edge_data(reviewer_id, product_id)
            reviewer_fairness = graph.nodes[reviewer_id]["fairness"]
            product_goodness = graph.nodes[product_id]["goodness"]
            current_reliability = edge_attributes["reliability"]
            score = edge_attributes["rating"]
            new_reliability = (1 / gamma1 + gamma2) * (
                        gamma1 * reviewer_fairness + gamma2 * (1 - (score - product_goodness) / 2))

            nx.set_edge_attributes(graph, {(reviewer_id, product_id): {"reliability": new_reliability}})

        iter += 1
        print("Done with iteration number " + str(iter) + " out of " + str(total_iters))

    sketchy_reviewers = list()
    # use fairness_list and order fairness_list by index 1 [1] aka fairness in ascending order, discard those with None (products) if it hasn't been already done
    fairness_by_least_fair = sorted(fairness_list, key=getFairness, reverse=True)
    # now iterate through fairness_list and while index 1 is =< 0.2, check graph.degree[fairness_list[i][0]], index 0 is the name of the
    # reviewer, this will return the degree of said node in the graph, if that value is >= 5, we do sketchy_reviewers.add(node)
    for reviewer in fairness_by_least_fair:
        if reviewer[1] > 0.2:
            break
        if graph.degree[reviewer[0]] >= 5:
            sketchy_reviewers.append(reviewer(0))

    print("=====    List of sketchiest reviewers (fairness =< 0.2)   =====")
    for index, reviewer_node in enumerate(sketchy_reviewers):
        reviewer = graph[reviewer_node]
        print(index, reviewer)

    quality_reviewers = list()
    # now we can check fairness_list, get rid of the elements whose index 1 = None aka products and then sort by descending order of index 1
    fairness_by_most_fair = sorted(fairness_list, key=getFairness)
    # we iterate through fairness_list as long as index 1 (fairness) >= 0.9, every element we iterate through we do quality_reviewers.append(node)
    for reviewer in fairness_by_most_fair:
        if reviewer[1] < 0.9:
            break
        if graph.degree[reviewer[0]] >= 10:
            quality_reviewers.append(reviewer(0))

    print("Proportion of extremely fair nodes: " + str(float(len(quality_reviewers) / len(fairness_list))))


main()
