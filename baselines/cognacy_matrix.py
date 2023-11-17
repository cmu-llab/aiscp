import pandas as pd
from pathlib import Path


FILE = "../data/tukanoan_cognates_new.csv"
EMPTY_CELL = "?"


# for this baseline, the character states of phylogenetic inference are cognates
# this tracks lexical innovation
dataset = pd.read_csv(FILE, sep='\t')
dataset = dataset.rename(columns={"#id": "cognate_set"})
dataset = dataset.set_index("cognate_set")
def encode_matrix(cell):
    # 1 if the language has an entry for the cognate set
    return int(cell != EMPTY_CELL)
dataset = dataset.applymap(encode_matrix)
Path(f"cognacy/").mkdir(parents=True, exist_ok=True)
# this contains the proto-language, but the matrix fed into PHYLIP does not
dataset.to_csv(f"cognacy/cognacy_matrix.tsv", sep='\t', index=True)


with open('infile', 'w') as f:
    dataset = dataset.T
    n_langs, n_cognatesets = dataset.shape
    f.write(f"{n_langs - 1} {n_cognatesets}\n")

    for index, row in dataset.iterrows():
        lang = row.name

        # exclude the proto-language
        if "Proto" in lang:
            continue

        # penny only allow lang names to be 10 chars long
        f.write(lang[:10])
        if len(lang) < 10:
            f.write(" " * (10 - len(lang)))
        f.write('\t')
        f.write(''.join(map(str, row.values)))
        f.write('\n')
