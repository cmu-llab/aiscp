from newick import read
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tree", type=str, required=True, \
    help="Consensus tree from PHYLIP's consense")
args = parser.parse_args()


# PHYLIP only accepts lang names with 10 characters, so we must do this for the gold tree too
tree = read(args.tree)
def set_name(node):
    if node.name:
        node.name = node.name[:10]
        node.name = "\"" + node.name + "\""
tree[0].visit(set_name)
newick_tree = tree[0].newick

with open(args.tree + '.newick', 'w') as f:
    f.write(newick_tree)
    # add semicolon, otherwise quartet_dist will not terminate
    f.write(';\n')
