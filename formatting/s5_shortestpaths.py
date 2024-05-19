import networkx as nx
import panphon
from diachronic_panphon import DiachronicDistance
import math
import yaml
import codecs
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from argparse import ArgumentParser

from tools import preprocess_phone

def get_ipa_diacritics():
    # adapted from panphon/bin/generate_ipa_all.py
    dia_defs = 'diacritic_definitions.yml'
    defs = yaml.load(codecs.open(dia_defs, "r", "utf-8").read(), Loader=yaml.FullLoader)
    all_diacritics = set()
    for dia in defs['diacritics']:
        all_diacritics.add(dia['marker'])
    
    # ʰ (aspirated),  ̩ (syllabic), ˞ (rhotacized), ˀ (glottalized), ʷ (labialized), ʲ (palatalized),  ̃ (nasalized), ː (long), ˤ (pharyngealized)
    diacritics_allow_list = {
        'ʰ', '̃', "̥" # 'ʲ', '̩', '˞', 'ˀ', 'ʷ', 'ː', 'ˤ'
    }
    
    return all_diacritics, diacritics_allow_list, all_diacritics - diacritics_allow_list


def restrict_phones(phones):
    new_phones = []
    _, include_diacritics, exclude_diacritics = get_ipa_diacritics()
    DIACRITIC_THRESHOLD = 2  # exclusive

    # remove phones with duplicate features (FED 0)
        # ('s̪', 'θ'), ('z̪', 'ð'), ('t͡ɕ', 'c͡ç'), ('ɟ͡ʝ', 'd͡ʑ'), ('r', 'ɾ'), ('ɔ', 'ɵ'), ('ɜ', 'ʌ'), ('ɞ', 'o'), ('ɐ', 'ʌ'), ('a', 'æ'), ('ɘ', 'ə')
        # ('t̪͡s̪', 't̪͡θ'), ('t̪͡θ', 't̪͡s̪'), ('d̪͡z̪', 'd̪͡ð'), ('d̪͡ð', 'd̪͡z̪')
    EXCLUDE_PHONES = {
        's̪', 'z̪', 'c͡ç', 'ɟ͡ʝ', 'ɾ', 'ɵ', 'ɜ', 'ɞ', 'ɐ', 'æ', 'ɘ', 
        't̪͡s̪', 't̪͡θ', 'd̪͡z̪', 'd̪͡ð',
    }
    new_EXCLUDE_PHONES = set(EXCLUDE_PHONES)
    for phone in EXCLUDE_PHONES:
        new_EXCLUDE_PHONES.add(phone + '̃')
        new_EXCLUDE_PHONES.add(phone + 'ʰ')
    
    exclude_diacritics |= {'˩', '˨', '˧', '˦', '˥'}

    with open('excluded_phones.txt', 'w', encoding='utf-8') as f:
        for phone, feature in phones:
            # exclude phones with diacritics not in the allow list
            # also restrict to phones whose IPA representation has 0 or 1 allowed diacritic
            if phone not in new_EXCLUDE_PHONES and \
                len(exclude_diacritics & set(phone)) == 0 and sum(map(phone.count, include_diacritics)) < DIACRITIC_THRESHOLD:
                new_phones.append((phone, feature))
            else:
                f.write(phone)
                f.write('\n')
    
    print(len(new_phones), 'phones included, ', len(phones) - len(new_phones), 'excluded')
    return new_phones


def manually_restrict_phones():
    with open('formatting/include_phones.txt', 'r') as f:
        phones = set([p.strip() for p in f.readlines()])
    print(len(phones), 'phones included')
    return phones


def create_phone_graph(weighted_edit_dist=False):
    """
    Results in a fully connected graph
    """
    graph = nx.Graph()
    ft = panphon.FeatureTable()
    dist = DiachronicDistance()

    phones = manually_restrict_phones()
    for i, phone1 in enumerate(phones):
        for j, phone2 in enumerate(phones):
            if i < j:
                if weighted_edit_dist:
                    weight = dist.weighted_feature_edit_distance(phone1, phone2)
                else:
                    weight = dist.feature_edit_distance(phone1, phone2)
                graph.add_edge(phone1, phone2, weight=weight)

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
    parser.add_argument("--weighted_edit_dist",
                        type=lambda x: (str(x).lower() == 'true'),
                        default=False)
    parser.add_argument("--out-graph-pkl",\
        default='phone_graph.pkl',\
        type=str)
    args = parser.parse_args()

    graph = create_phone_graph(weighted_edit_dist=args.weighted_edit_dist)

    with open(args.out_graph_pkl, 'wb') as f:
        pickle.dump(graph, f)

    # nx.draw_networkx(graph)
    # plt.savefig("phone_graph.png")
