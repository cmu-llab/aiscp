import argparse
import os
import math
import pdb

import warnings # FIXME should replace df.append with pd.concat
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import pickle as pkl

from s5_shortestpaths import find_shortest_paths

# FIXME need to name "Number" an index somehow
CHECK_SHORT=False

#with open("lang2code.pkl", 'rb') as f: # FIXME hard coded
#    lang2code = pkl.load(f)

def extract_lang_name(fn: str) -> str:
    """
    Helper function
    """
    assert fn.endswith(".tsv")
    name = os.path.split(fn)[-1][:-4].split('_')[0][:3]
    if name == "Sio":
        name = "Sir"
    return name

def split_rule(rule: str) -> tuple:
    """
    Helper function
    splits str rule into three strs for proto, daughter, and ctx
    """
    #rule = rule.replace("C/V", "CV")
    if rule.count(' / ') != 1:
        print(f"WARNING: !=1 ' / ' char's in rule {rule}")
    if rule.count(' -> ') != 1:
        print(f"WARNING: !=1 ' -> ' char's in rule {rule}")

    arrow_idx = rule.index(' -> ')
    slash_idx = rule.index(' / ')

    proto = rule[:arrow_idx].strip()
    daught = rule[arrow_idx+4:slash_idx].strip()
    ctx = rule[slash_idx+3:].strip()
    return proto, daught, ctx

def check_short_enough(proto, subd, tol):
    if not CHECK_SHORT:
        return True
    for key in subd:
        len_ = len(find_shortest_paths(proto, subd[key]))
        print('>', end='', flush=True)
        if len_ > tol:
            return False
    return True

def clean_row(d: dict) -> dict:
    """
    Helper function
    Clean dictionary to be data frame row for reflexes data frame
    """
    for k in d:
        if k == "Number" or k == "Context":
            pass
        elif k == "Proto-sound":
            pure_proto = d[k].replace(' ', '')
            d[k] = '*' + pure_proto
        else:
            if type(d[k]) != str:
                if math.isnan(d[k]):
                    d[k] = pure_proto
        

def dict2df(d: dict) -> pd.DataFrame:
    """
    Takes dictionary of of this form:
    +-(proto, ctx)-+-daught_lang--daught_form
    |              |
    |              +-daught_lang--daught_form
    |              |
    |              +-...
    +-(proto, ctx)-+-daught_lang--daught_form
    |              |
    |              +-daught_lang--daught_form
    |              |
    |              +-...
    +-...

    and output DataFrame in form of relfexes TSV
    """

    # Collect daughter langs
    all_daught_langs = []
    for key in d:
        all_daught_langs += list(d[key].keys())
    all_daught_langs = list(set(all_daught_langs))

    df_cols = ['Number', 'Proto-sound', 'Context'] + all_daught_langs
    df_init = {col:[] for col in df_cols}
    df = pd.DataFrame(df_init)

    print("Beginning data frame loop", flush=True)
    num = 1
    for key in d:
        proto, ctx = key
        subd = d[key] # Mapping daughter lang to form
        if len(subd) == 1:
            continue # filter out changes that only apply to a single daughter
        subd['Number'] = int(num)
        subd['Proto-sound'] = proto
        subd['Context'] = ctx
        # Add extra keys
        for dl in all_daught_langs:
            if dl not in subd: # FIXME is this the right strategy?
                subd[dl] = proto.strip('*') # Default to proto instead of NaN
        # Check of short enough
        lil_subd = {dl: subd[dl] for dl in all_daught_langs}
        if check_short_enough(proto, lil_subd, tol=100):
            df = df.append(subd, ignore_index=True)
        # print log
        if num % 50 == 0 or CHECK_SHORT:
            print(num, end=' ', flush=True)
        num += 1
    print()
    
    print("Final DataFrame shape:", df.shape)
    return df


def clean_df(df: pd.DataFrame):
    """
    Remove rows where only one of the languages attests the sound change
    """
    return df # comment out
    new_rows = []
    for idx, row in indf.iterrows(): 
        proto = row

def populate_tsv(infiles: list, outfile: str='reflexes.tsv'):
    """
    Args:
        infiles (List[str]): list of filenames for W+L output tsvs
    """
   
    assert outfile.endswith(".tsv"), "Assuming file with *.tsv extn"

    # Organize as dict first
    d = {}
    re_d = {}
    # Loop through the TSV files (one for each daughter lang
    print("Looping through infiles", end=' ')
    for infile in infiles:
        
        ldf = pd.read_table(infile)
        lname = extract_lang_name(infile)
        lar = ldf.values
        for row in lar:
            # Real rule
            rule = row[1]
            proto, daught, ctx = split_rule(rule) 
            proto = proto.replace(' ', '')
            proto = '*' + proto
            daught = daught.replace(' ', '')
            if (proto, ctx) not in d:
                d[(proto, ctx)] = {}
            d[(proto, ctx)][lname] = daught
            # Regex rule
            re_rule = row[2]
            re_proto, re_daught, re_ctx = split_rule(re_rule)
            if (proto, re_ctx) not in re_d:
                re_d[(proto, re_ctx)] = {}
            re_d[(proto, re_ctx)][lname] = daught

        print('.', end=' ', flush=True)
    print()

    # Now format dictionary as DataFrame
    print("Dictionary length:", len(d))
    df = dict2df(d)
    print("Dataframe shape:", df.shape)
    df = clean_df(df)
    print("Cleaned dataframe shape:", df.shape)
    df.to_csv(outfile, sep='\t', index=False)
    print("TSV written to", outfile, flush=True)

    # Add extra output for regex and comparison
    re_outfile = outfile[:-4] + "_regex.tsv"
    re_df = dict2df(re_d)
    re_df = clean_df(re_df)
    re_df.to_csv(re_outfile, sep='\t', index=False)
    print("Written regex version to", re_outfile)
    print("Done")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-o", "--outfile", type=str, default="reflexes.tsv",\
            help="File to which we write the reflexes TSV")
    parser.add_argument("-i", "--indir", type=str, required=True,\
            help="Directory to which W+L TSVs are written")
    parser.add_argument("--filter-rules", action="store_true",\
            help="Whether to use filtered rules")

    args = parser.parse_args()

    if args.filter_rules:
        suffix = "rules_pruned_accuracy10.0_filtered.tsv"
    else:
        suffix = "rules_pruned_accuracy10.0.tsv"

    infiles = os.listdir(args.indir)
    infiles = [os.path.join(args.indir, fn) for fn in infiles if\
            fn.endswith(suffix)]

    populate_tsv(infiles=infiles, outfile=args.outfile)
