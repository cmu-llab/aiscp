import panphon
import numpy as np

def get_augmented_feature_vectors(source_vec):
    """
    Return: Length 72 multi-hot encoding of each feature value
    """
    # 1[feature = -1], 1[feature = 0], 1[feature = +1] for each feature
    encoding = []
    for (src_feat, src_val) in source_vec.iteritems():
        encoding.append(int(src_val == -1))
        encoding.append(int(src_val == 0))
        encoding.append(int(src_val == 1))
    return np.array(encoding)

def preprocess_phone(ft, phone):
    if not ft:
        ft = panphon.FeatureTable()

    STOP = {
        'son': -1,
        'cont': -1
    }
    FRICATIVE = {
        'son': -1,
        'cont': 1
    }
    if phone[0] == "*":
        phone = phone[1:]
    elif len(phone) >= 2 and \
        ft.fts(phone[0]) and ft.fts(phone[0]).match(STOP) and ft.fts(phone[1]) and ft.fts(phone[1]).match(FRICATIVE):
        # add ligature to affricates (stop + fricative)
        phone = phone[0] + '͡' + phone[1]

    return phone
