import pandas as pd
import argparse
from pathlib import Path


# gold reflexes / sound correspondences
FILE = "../tukano-paper/D_reflexes.tsv"

# for this baseline, the character states of phylogenetic inference are shared innovations
reflexes_df = pd.read_csv(FILE, sep='\t',)
reflexes_df = reflexes_df.drop(columns=["Context", "Number"])
def encode_matrix(row):
    proto = row["Proto-sound"].strip("*")
    # 1 if the daughter language participated in the sound change for that row
    def did_daughter_participate(cell):
        return int(cell != proto)
    row.loc[row.index != 'Proto-sound'] = row.loc[row != 'Proto-sound'].map(did_daughter_participate)
    return row
reflexes_df = reflexes_df.apply(encode_matrix, axis=1)

Path(f"shared_innovations/").mkdir(parents=True, exist_ok=True)
reflexes_df.to_csv(f"shared_innovations/shared_innovations_matrix.tsv", sep='\t', index=False)

# number of langs / sound change
reflexes_df['n_participating'] = reflexes_df.eq(1).sum(axis=1)
reflexes_df.to_csv(f"shared_innovations/debug_shared_innovations.tsv", sep='\t', index=False)
# only keep n_participating column in the TSV for debugging
reflexes_df = reflexes_df.drop(['n_participating'], axis=1)

with open('infile', 'w') as f:
    reflexes_df = reflexes_df.T
    n_langs, n_cognatesets = reflexes_df.shape
    f.write(f"{n_langs - 1} {n_cognatesets}\n")

    for index, row in reflexes_df.iterrows():
        lang = row.name
        if lang == "Proto-sound":  # proto-lang
            continue

        # penny only allow lang names to be 10 chars long
        f.write(lang[:10])
        if len(lang) < 10:
            f.write(" " * (10 - len(lang)))
        f.write('\t')
        f.write(''.join(map(str, row.values)))
        f.write('\n')
