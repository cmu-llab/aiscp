import pdb
import argparse
import random
import pickle as pkl
import networkx as nx
import pandas as pd

from tools import preprocess_phone
        
BIG_NUMS = []

def reflex2change(infile: str, outfile: str, keep_oov: bool=False):
    """
    Args:
        infile (str): path to reflexes TSV file
        outfile (str): path to write changes TSV to
    """

    # Dealing with rules of the form A -> B / ctx, where ctx = C _ D
    
    indf = pd.read_table(infile)
    print("Read", infile, ", data frame shape:", indf.shape, flush=True)
    
    srs = []
    phones2add = []
    for idx, row in indf.iterrows():
        proto = row['Proto-sound'].strip('*')
        ctx = row['Context']
        num = row['Number']
        reflexes = row.values[3:]
        reflexes = list(set(list(reflexes)))
        reflexes = [r.strip('*') for r in reflexes]
        #if proto in reflexes:
        #    reflexes.remove(proto) # FIXME good idea?
        try:
            reflex_str = ','.join(reflexes)
        except:
            pdb.set_trace()
        # Construct dict to build off of for duplicated rows
        base_d = {"Number":num, "Proto":proto, "Context":ctx,\
                "Reflexes":reflex_str}
        # Go through each reflex and get shortest paths for each
        num_dups = 0
        for reflex in reflexes:
            # / marks multiple reflexes
            for reflex in reflex.split('/'):
                reflex = preprocess_phone(None, reflex)

                try:
                    short_paths = [[proto, reflex]] #find_shortest_paths(proto, reflex)
                except nx.exception.NetworkXNoPath:
                    print(f"{proto}>{reflex}", 'lacks a path in the phone graph', end=' ')
                    phones2add.append(proto); phones2add.append(reflex)
                    continue
                except nx.exception.NodeNotFound:
                    print(proto, 'or', reflex, 'is not in the phone graph', end=' ')
                    phones2add.append(proto); phones2add.append(reflex)
                    short_paths = [[proto, reflex]]
                    if not keep_oov:
                        continue
                
                # Make unique
                short_paths = [list(x) for x in set(tuple(x) for x in short_paths)]

                # short_paths represented as list of lists, where each list is a path 
                #   through phones
                # Loop through paths
                for path in short_paths:
                    # Loop through intermediate edges
                    for i in range(len(path) - 1):
                        src = path[i]
                        tgt = path[i+1]
                        dup_d = base_d.copy()
                        dup_d['Source'] = src
                        dup_d['Target'] = tgt
                        # Create pandas Series to add to final data frame
                        sr = pd.Series(dup_d)
                        srs.append(sr)
                        num_dups += 1
        # For the user to see progress
        #if num_dups > 1000:
        #    pdb.set_trace()
        print(num_dups, end=' ', flush=True)
        if num_dups > 1000:
            BIG_NUMS.append(num)
    print()

    # Write phones to add to a file
    with open("phones_not_in_graph.pkl", 'wb') as f:
        pkl.dump(phones2add, f)

    # Create data frame
    outdf = pd.DataFrame(srs)
    outdf.to_csv(outfile, index=False, sep='\t')
    print("Written to", outfile)
    print(outdf, flush=True)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--outfile", type=str, default="changes.tsv",\
            help="File to which we write the reflexes TSV")
    parser.add_argument("-i", "--infile", type=str, default="reflexes.tsv",\
            help="Directory to which W+L TSVs are written")
    parser.add_argument("--keep-oov", action="store_true",\
            help="Set to true to keep phones that are not in the graph")

    args = parser.parse_args()

    reflex2change(infile=args.infile, outfile=args.outfile, keep_oov=args.keep_oov)

    with open("nums-1000+.pkl", 'wb') as f:
        pkl.dump(BIG_NUMS, f)
