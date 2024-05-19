from panphon import FeatureTable
from panphon.distance import Distance
import unicodecsv as csv


class DiachronicFeatureTable(FeatureTable):
    def __init__(self) -> None:
        super(DiachronicFeatureTable, self).__init__()

        # [h] and [ʔ] are sonorants synchronically but are obstruents diachronically
        self.seg_dict['h'].update({'son': -1})
        self.seg_dict['ʔ'].update({'son': -1})
        

class DiachronicDistance(Distance):
    def __init__(self) -> None:
        super(DiachronicDistance, self).__init__()
        file = "formatting/data/diachronic_feature_weights.csv"
        # source: https://github.com/dmort27/panphon/blob/master/panphon/featuretable.py - _read_weights
        with open(file, 'rb') as f:
            reader = csv.reader(f, encoding='utf-8')
            next(reader)
            weights = [float(x) for x in next(reader)]
        self.fm.weights = weights


if __name__ == '__main__':
    # test
    ft = DiachronicFeatureTable()
    for p, feat in ft.segments:
        if p == 'h' or p == 'ʔ':
            assert feat.__getitem__('son') == -1
