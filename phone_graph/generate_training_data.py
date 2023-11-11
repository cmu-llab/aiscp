import pandas as pd
import panphon
import sklearn
import numpy as np
import random
import pickle as pkl
from util import preprocess_phone, get_augmented_feature_vectors


SPLIT_DATA = False

def get_feature_changes(feature_table, source, target):
    """
    Return: Length 48 vector
    """
    # Each feature has 2 associated indices:
        # Does the feature increase?
        # Does the feature decrease?
    source_feats, target_feats = feature_table.fts(source), feature_table.fts(target)
    assert len(source_feats.items()) == len(target_feats.items())
    ft_changes = [] # [0] * (2 * len(source_feats.items()))
    for (src_feat, src_val), (trg_feat, trg_val) in zip(source_feats.iteritems(), target_feats.iteritems()):
        assert src_feat == trg_feat
        feat = src_feat
        # -1 to 1 or -1 to 0 or 0 to 1
        if src_val < trg_val:
            ft_changes.append(1)
            ft_changes.append(0)
        # 1 to -1 or 1 to 0 or 0 to -1
        elif src_val > trg_val:
            ft_changes.append(0)
            ft_changes.append(1)
        else:
            ft_changes.append(0)
            ft_changes.append(0)
    
    return np.array(ft_changes)


with open('formatting/include_phones.txt', 'r') as f:
    include_phones = [p.strip() for p in f.readlines()]
    include_phones = set(include_phones)
index_diach = pd.read_csv("index-diachronica/output/index_diachronica_output.csv", index_col='index')
# ignore Austronesian langs (index 776-1443)
    # fortunately, all the Austronesian langs are in a contiguous block
#index_diach = index_diach[~index_diach.index.isin(list(range(776,1444)))]
index_diach = index_diach[~index_diach.index.isin(list(range(640,715)))]
index_diach = index_diach[~index_diach.index.isin(list(range(602,625)))]
index_diach = index_diach[["source", "target"]]

examples = []
feature_table = panphon.FeatureTable()
for _, row in index_diach.iterrows():
    # source = proto-phoneme, target = child phoneme
    src, target = row["source"], row["target"]
    if pd.isna(src) or pd.isna(target):
        # empty
        continue
    src, target = preprocess_phone(feature_table, src), preprocess_phone(feature_table, target)
    
    # ignore phones not in our phone graph
    if src not in include_phones or target not in include_phones:
        print(src, 'or', target, 'not in included phone set')
        continue

    # X
    feat_vector = get_augmented_feature_vectors(feature_table.fts(src))
    # y
    feature_changes = get_feature_changes(feature_table, src, target)
    # List of tuples of (x,y)
    examples.append((feat_vector, feature_changes))


#examples = np.array(examples, dtype='object')

# train/dev/test split - random
random.seed(12345)
SPLIT_RATIO = (.8, .1, .1)
random.shuffle(examples)
if SPLIT_DATA:
    train = examples[:int(len(examples) * SPLIT_RATIO[0])]
    dev = examples[:int(len(examples) * SPLIT_RATIO[0])]
    test = examples[:int(len(examples) * SPLIT_RATIO[0])]

print()
print(f"{len(examples)} data total in set")

# save the output - 2D array: (# examples, 2)
# where each example is made of a (augmented feature vector, feature change) tuple
with open("phone_graph/train_data.pkl", 'wb') as f:
    pkl.dump(examples, f)
#np.save('phone_graph/train', train)
#np.save('phone_graph/dev', dev)
#np.save('phone_graph/test', test)
