import networkx as nx
import panphon
from panphon.distance import Distance
import math
import yaml
import codecs
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
import torch
import pdb

from util import preprocess_phone, get_augmented_feature_vectors
from train_nn import FeatChangeNN, FeatChangeMatrix, \
        FeatChangeDNN, FeatChangeResNN, FeatChangeDResNN

# Format of sound change vector: [P(feat incr.), P(feat decr.), ...]
# Format of aug vector: [feat == -1, feat == 0, feat == 1, ...]

class DirectionalDistance(Distance):
    def wt_dir_fed(self, p1, p2, probs, *args, **kwargs):
        # Calculate probabilities
        assert len(probs) == 24
        self.fm.weights = [1 - p for p in probs]
        return self.weighted_feature_edit_distance(p1, p2, *args, **kwargs)

class FEDModel():

    def __init__(self, model_path, device):
        
        self.device = device
        self.model = torch.load(model_path).to(self.device)
        self.ft = panphon.FeatureTable()
        self.dist = DirectionalDistance()

    def get_wt_dir_fed(self, p1, p2):

        # Calculate weights
        emb1 = self.ft.fts(p1)
        emb2 = self.ft.fts(p2)

        aug_emb1 = get_augmented_feature_vectors(emb1)
        emb1 = emb1.numeric()
        emb2 = emb2.numeric()

        change_probs = self.model(torch.from_numpy(aug_emb1).float().to(self.device))
        change_probs = change_probs.detach().cpu().numpy()
        
        probs = []
        for i in range(len(emb1)):
            if emb1[i] == emb2[i]:
                probs.append(1)
            else:
                probs.append(change_probs[2*i + int(emb2[i] < emb1[i])])
        probs = np.array(probs)

        fed = self.dist.wt_dir_fed(p1, p2, probs)

        return fed


def manually_restrict_phones():
    ft = panphon.FeatureTable()
    with open('formatting/include_phones.txt', 'r') as f:
        phones = set([p.strip() for p in f.readlines()])
    phones = [p for p in phones if ft.fts(p)]
    print(len(phones), 'phones included')
    return phones


def modified_kruskal(graph, nodes, edges):
    # idea: find MST but add redundant edges with equal weights if they exist
    # sort all POSSIBLE edges by FED
    # then add edge if 
        # it doesn't already exist
        # it connects 2 disconnected groups (no path exists before them)
    prev_weight = -1
    prev_connected_comp = set()  # the newly connected component formed from last added edge
    edges = sorted(edges.items(), key=lambda x: x[1])
    for edge, weight in edges:
        # what's different from Kruskal:
        # add all edges of same weight and that which connect the same 2 components
        # ensure that the edge is not internal to either of the connected components
        if not nx.has_path(graph, *edge) or \
            (math.isclose(prev_weight, weight) and \
                (edge[0] in prev_connected_comp_0 and edge[1] in prev_connected_comp_1) \
                    != \
                (edge[0] in prev_connected_comp_1 and edge[1] in prev_connected_comp_0) \
            ):
            # only add an edge if, had you removed the previous edge, the edge in question would connect the same connected components
            prev_weight = weight
            # save the connected components of the endpoints b4 adding the edge
            # update the connected component when dealing with different connected components
            
            if not nx.has_path(graph, *edge):
                prev_connected_comp_0, prev_connected_comp_1 = nx.node_connected_component(graph, edge[0]), nx.node_connected_component(graph, edge[1])
                assert len(prev_connected_comp_0 & prev_connected_comp_1) == 0
            # multiply weight by 24 to be in units of feature edits
            graph.add_edge(*edge, weight=weight)

    assert nx.is_connected(graph)

    return graph

def create_nearestnbor_phone_graph():
    '''
    Initialize the graph with undirected edges, where an edge indicates a feature edit of 1
    
    Results in a connected graph where edges are only between nearest nbors or are needed to maintain connectedness
    '''
    graph = nx.Graph()
    ft = panphon.FeatureTable()
    dist = DiachronicDistance()
    
    # Undirected graph of all phones in panphon
        # there is an edge for each phone with feature edit 1
    phones = restrict_phones(ft.segments)

    # precompute FED between all pairs of phones
    fed = defaultdict(lambda: defaultdict(float))
    all_poss_edges = {}
    for idx, (phone, _) in enumerate(phones):
        for (other_phone, _) in phones[idx + 1:]:
            distance = dist.weighted_feature_edit_distance(phone, other_phone)
            fed[phone][other_phone] = distance
            fed[other_phone][phone] = distance
            all_poss_edges[(phone, other_phone)] = distance
            all_poss_edges[(other_phone, phone)] = distance

    # find nearest neighbor
    for idx, (phone, features) in enumerate(phones):
        feature_vector = features.numeric()

        nbors = [(p, dst) for (p, dst) in fed[phone].items()]
        nbors = sorted(nbors, key=lambda x: x[1])
        shortest_dist = nbors[0][1]
        nearest_nbors = filter(lambda x: math.isclose(x[1], shortest_dist), nbors)
        for p, dst in nearest_nbors:
            # multiply weight by 24 to be in units of feature edits
            graph.add_edge(phone, p, weight=dst)

    graph = modified_kruskal(graph, phones, all_poss_edges)

    graph = graph.to_directed()
    # insertions (∅ > phone), deletions (phone > ∅)
        # TODO: glottal stop should be closest to null
    INS_WEIGHT = 15
    DEL_WEIGHT = 10
    nodes = list(graph)
    for phone in nodes:
        graph.add_edge(phone, '∅', weight=DEL_WEIGHT)
        graph.add_edge('∅', phone, weight=INS_WEIGHT)

    assert nx.is_strongly_connected(graph)
    return graph


def create_phone_graph(model_path, device):
    """
    Results in a fully connected graph
    """

    M = FEDModel(model_path, device)

    graph = nx.DiGraph()

    print("Creating graph edges", flush=True)
    phones = manually_restrict_phones()
    for i, phone1 in enumerate(phones):
        print(f'p{i}', end=' ', flush=True)
        for j, phone2 in enumerate(phones):
            if i != j:
                weight = M.get_wt_dir_fed(phone1, phone2)
                graph.add_edge(phone1, phone2, weight=weight)
    print()

    # insertions (∅ > phone), deletions (phone > ∅)
        # TODO: glottal stop should be closest to null
    INS_WEIGHT = 15
    DEL_WEIGHT = 10
    nodes = list(graph)
    print("Adding edges to ∅", flush=True)
    for phone in nodes:
        graph.add_edge(phone, '∅', weight=DEL_WEIGHT)
        graph.add_edge('∅', phone, weight=INS_WEIGHT)

    print("Graph complete", flush=True)
    assert nx.is_strongly_connected(graph)
    return graph


def load_graph(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G


# Returns: list of lists where each list is a shortest path of phones
# FIXME - get rid of hardcoded
def find_shortest_paths(source, target, filename='phone_graph.pkl'):
    source = preprocess_phone(None, source)
    target = preprocess_phone(None, target)
    if source.startswith('*'):
        source = source[1:]
    assert source, "Empty source!"
    if target.startswith('*'):
        target = target[1:]
    assert target, "Empty target!"

    graph = load_graph(filename)

    # finds shortest path, not just the cost
    paths = nx.all_shortest_paths(graph, source, target, weight='weight')
    return list(paths)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out-graph-pkl",\
        default='phone_graph.pkl',\
        type=str)
    parser.add_argument("--model-path",\
            required=True,\
            type=str)
    parser.add_argument("--gpu_num",\
            type=int,\
            default=None,\
            help="GPU to use")
    args = parser.parse_args()

    if args.gpu_num is not None:
        args.device = torch.device("cuda:{}".format(args.gpu_num))
    else:
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")
            print("WARNING: Using CPU since no GPU available", flush=True)

    graph = create_phone_graph(args.model_path, args.device)

    print("Writing to pkl", flush=True)
    with open(args.out_graph_pkl, 'wb') as f:
        pickle.dump(graph, f)
    print("Written to", args.out_graph_pkl, flush=True)

    #nx.draw_networkx(graph)
    #plt.savefig("phone_graph.png")
    print("done")
