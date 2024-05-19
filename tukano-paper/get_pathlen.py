# author   : Johann-Mattis List
# email    : mattis.list@lingpy.org
# created  : 2015-08-05 20:32
# modified : 2015-08-05 20:32
"""
Script prepares the data for the parsimony experiment on Tukano languages.

What this script basically does is reading in the files on sound changes and
reflexes and creating a weighted network. All data is then stored as JSON, and
JSON is again loaded by the main script that runs the analysis.
"""

__author__="Johann-Mattis List"
__date__="2015-08-05"

from lingpy.util import read_text_file
import networkx as nx
import json
import pickle as pkl

import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument("--change-tsv", required=True, type=str)
parser.add_argument("--reflex-tsv", required=True, type=str)
parser.add_argument("--lang-tsv", required=True, type=str)
parser.add_argument("--graph-pkl", required=True, type=str)

args = parser.parse_args()


# csv2list from lingpy except NFD instead of NFC
def csv2list(
    filename,
    fileformat='',
    dtype=None,
    comment='#',
    sep='\t',
    strip_lines=True,
    header=False
):
    """
    Very simple function to get quick (and somewhat naive) access to CSV-files.

    Parameters
    ----------
    filename : str
        Name of the input file.
    fileformat : {None str}
        If not specified the file <filename> will be loaded. Otherwise, the
        fileformat is interpreted as the specific extension of the input file.
    dtype : {list}
        If not specified, all data will be loaded as strings. Otherwise, a
        list specifying the data for each line should be provided.
    comment : string (default="#")
        Comment character in the begin of a line forces this line to be
        ignored (set to None  if you want to parse all lines of your file).
    sep : string (default = "\t")
        Specify the separator for the CSV-file.
    strip_lines : bool (default=True)
        Specify whether empty "cells" in the input file should be preserved. If
        set to c{False}, each line will be stripped first, and all whitespace
        will be cleaned. Otherwise, each line will be separated using the
        specified separator, and no stripping of whitespace will be carried
        out.
    header : bool (default=False)
        Indicate, whether the data comes along with a header.

    Returns
    -------
    l : list
        A list-representation of the CSV file.

    """
    # check for correct fileformat
    if fileformat:
        infile = filename + '.' + fileformat
    else:
        infile = filename

    if dtype is None:
        dtype = []

    l = []

    # open the file
    infile = read_text_file(infile, lines=True, normalize="NFD")

    # check for header
    idx = 0 if header else -1

    for i, line in enumerate(infile):
        if line and (not comment or not line.startswith(comment)) and idx != i:
            if strip_lines:
                cells = [c.strip() for c in line.strip().split(sep)]
            else:
                cells = [c.strip() for c in line.split(sep)]
            if not dtype:
                l += [cells]
            else:
                l += [[f(c) for f, c in zip(dtype, cells)]]

    return l

# function to make local graph
def local_graph(changes):
    C = {}
    for line in changes:
        idx, prt, ctx = int(float(line[0])), line[1], line[2]
        source, target = line[-2].strip(), line[-1].strip()
        if source != target:
            try:
                C[idx, prt, ctx] += [[source, target]]
            except:
                C[idx, prt, ctx] = [[source, target]]
    return C


def get_weight_from_graph(graph, nodeA, nodeB, chars):

    try:
        d = nx.shortest_path_length(graph, nodeA, nodeB, weight='weight')
    except nx.NetworkXNoPath:
        d = len(chars) * 10
    except nx.NetworkXError:
        d = len(chars) * 10

    return d

# get correspondences and changes
corrs = csv2list(args.reflex_tsv, strip_lines=False)

# load the taxa
tdat = csv2list(args.lang_tsv, strip_lines=False)
taxa = [x[0] for x in tdat[1:]]

# get the header
header = corrs[0] # [h.title() for h in corrs[0]]

# get the main data
data = corrs[1:]

# get the proto-data
D = {}
for line in data:
    idx = int(float(line[0]))
    proto = line[1].strip()[1:]
    ctx = line[2]
    refs = [x.strip() for x in line]
    for i,ref in enumerate(refs):
        # possibly allophonic variation within a variety
        if '/' in ref:
            nrefs = [x.strip() for x in ref.split('/')]
        else:
            nrefs = [ref]
        refs[i] = nrefs
    
    tmp = dict(
            zip(
                header, 
                refs 
                )
            )
    tmph = sorted([h for h in tmp if h in taxa])
    patterns = [tmp[h] for h in tmph]
    
    D[idx, proto, ctx] = patterns

# load the two different sound change patterns
sc_complex = csv2list(args.change_tsv)[1:]
# make the dictionary and the graph
CL = local_graph(sc_complex)

avg_num_edges = []
for key, corr in CL.items():
    avg_num_edges.append(len(corr))
print('avg # edges', sum(avg_num_edges) / len(avg_num_edges), 'across', len(avg_num_edges), ' sound correspondences')

