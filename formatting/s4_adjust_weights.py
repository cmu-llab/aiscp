import networkx as nx
from s5_shortestpaths import load_graph, find_shortest_paths
import pickle
import networkx as nx
from argparse import ArgumentParser
import panphon 

from tools import preprocess_phone

parser = ArgumentParser()

parser.add_argument("--index-diachron",\
        default="index-diachronica/output/index_diachronica_unicode_4.6",\
        type=str)
parser.add_argument("--in-graph-pkl",\
        default="phone_graph.pkl",\
        type=str)
parser.add_argument("--out-graph-pkl",\
        default='phone_graph.pkl',\
        type=str)

args = parser.parse_args()

# Choose a value x \in [0.5, 1.0)
DISCOUNT = 0.98
graph = load_graph(args.in_graph_pkl)


# read index diachronica
with open(args.index_diachron, 'r') as f, open('bad-index-diachronica-rules', 'w') as debug_f:
    rules = f.readlines()

    ft = panphon.FeatureTable()

    # Whenever we find a correspondence X > Y in Index Diachronica,
    #   find all the edges in all the shortest paths from X > Y, 
    #   and multiply each edge weight by DISCOUNT
    print("Tracking progress: . = one rule")
    for rule in rules:
        print('.', end=' ', flush=True)
        # rule of the form A > B / ctx
        rule = eval(rule)  # TODO: remove the eval when the ID file becomes a CSV
        source_phone, target_phone = rule[:2]
        if source_phone == target_phone or len(source_phone) == 0 or len(target_phone) == 0:
            # rule improperly extracted
            continue

        source_phone = preprocess_phone(ft, source_phone)
        target_phone = preprocess_phone(ft, target_phone)

        try:
            shortest_paths = find_shortest_paths(source_phone, target_phone, args.in_graph_pkl)
        except (nx.exception.NodeNotFound, nx.exception.NetworkXNoPath):
            # skip the rule
            # print(source_phone, target_phone, '- one of these is not in the graph')
            debug_f.write(source_phone + ' or ' + target_phone + ' is not in the graph\n')
            continue

        for path in shortest_paths:
            assert len(path) > 1
            prev = source_phone
            for curr in path[1:]:
                # if a correspondence occurs multiple times in Index Diachronica,
                # it will be downweighted that many times
                graph[prev][curr]['weight'] *= DISCOUNT
                prev = curr
    print()

with open(args.out_graph_pkl, 'wb') as f:
    pickle.dump(graph, f)
