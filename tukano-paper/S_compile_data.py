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

def global_graph(changes):
    G = nx.DiGraph()
    for line in changes:
        idx, prt, ctx = int(line[0]), line[1], line[2]
        source, target = line[-2].strip(), line[-1].strip()
        if source != target:
            G.add_edge(source, target)
    return G

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

# create the digraph
G = nx.DiGraph()

# start compiling the output dictionary to be then formatted to json
out = {}
out['patterns'] = []
out['chars'] = []
out['fitch.chars'] = []
out['fitch'] = []
out['protos'] = []

# we first make all patterns without the proto-forms
protos = sorted(D.keys())
for p in protos:
    
    # get the data
    
    pattern = D[p]
    out['patterns'] += [pattern]
    chars = []
    for px in pattern:
        for c in px:
            chars += [c]

    chars = sorted(set(chars))
    matrix = [[0 for p in chars] for c in chars]
    for i,c1 in enumerate(chars):
        for j,c2 in enumerate(chars):
            if i < j:
                matrix[i][j] = 1
                matrix[j][i] = 1
    out['fitch'] += [matrix]
    out['chars'] += [chars]
    out['fitch.chars'] += [chars]
    out['protos'] += [[p[0], p[1], p[2]]]
out['taxa'] = tmph

# we add three matrix types for now, one complete, one with the full network,
# and one with partial networks
out[''] = [] # ???
out['diwest'] = []
out['sankoff'] = []

# make the dictionary and the graph
CL = local_graph(sc_complex)

# HACK
with open(args.graph_pkl, 'rb') as f:
    full_dgraph = pkl.load(f)
full_ugraph = full_dgraph.to_undirected() 

for idx, (a, b, c) in enumerate(out['protos']):

    # TODO: remove (a,b,c) in CL b/c reflexes.tsv (D) and changes.tsv (CL) should be consistent
    if (a,b,c) in D and (a,b,c) in CL:
       
        # create graphs for local analyses
        cdl = nx.DiGraph()
        cdl.add_edges_from(CL[a, b, c])
        cul = cdl.to_undirected()

        # all chars we want to consider this time are in the complex undirected
        # graph (but also the complex local graph)
        chars = sorted(cul.nodes())
        char2idx = { c:idx for idx, c in enumerate(chars) }
        nlen = len(chars)

        # create the matrices
        PENALTY = 100
        m_cdl = [[PENALTY if x != y else 0 for x in range(nlen)] for y in range(nlen)] # diwest model
        m_cul = [[0 for x in range(nlen)] for y in range(nlen)] # sankoff model

        # Check if there is a node not contained in the digraph
        trigger_not_found = False
        for edge in CL[a, b, c]:
            for phon in edge:
                if phon not in full_dgraph.nodes:
                    trigger_not_found = True
                    phon_ = phon

        if trigger_not_found:
            print(phon_, "not found in graph", flush=True)
            assert nlen == 2, "Not expected: phone not in graph, but more than 2 edges"
            for edge in CL[a, b, c]:
                (src, dst) = edge
                if m_cdl[char2idx[src]][char2idx[dst]] > 0:
                    m_cdl[char2idx[src]][char2idx[dst]] = 1

            for i,nA in enumerate(chars):
                for j,nB in enumerate(chars):
                    m_cul[i][j] = 1 - int(nA == nB)

            out['diwest'] += [m_cdl]
            out['sankoff'] += [m_cul]
            out['chars'][idx] = chars
            continue

        # directionality
        #   we only care about phone pairs for which an edge exists on the path
        #   from proto > reflex.
        #   all other phone pairs get penalized for this correspondence
        # trace through each edge on the shortest paths from proto > reflex
        for edge in CL[a, b, c]:
            (src, dst) = edge
            m_cdl[char2idx[src]][char2idx[dst]] = get_weight_from_graph(full_dgraph, src, dst, chars)

        for i,nA in enumerate(chars):
            for j,nB in enumerate(chars):
                m_cul[i][j] = get_weight_from_graph(full_ugraph, nA, nB, chars) # HACK

        out['diwest']    += [m_cdl]
        out['sankoff']  += [m_cul]
        out['chars'][idx] = chars
    else:
        print(a,b,c)

all_chars = []
for charset in out['chars']:
    all_chars += charset
out['allchars'] = sorted(set(all_chars))
for i,c in enumerate(out['allchars']):
    print(i+1,c)

reflexes = []
for pattern in out['patterns']:
    for p in pattern:
        reflexes += p
out['reflexes'] = sorted(set(reflexes))
for r in out['reflexes']:
    print(r)

with open('I_data.json', 'w') as f:
    f.write(json.dumps(out, indent=2))



