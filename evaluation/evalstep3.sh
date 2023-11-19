#!/bin/bash
# Usage: $0 <list of trees from DiWEST>
[ $# -eq 0 ] && { echo "Usage: $0 <list of trees from DiWEST>"; exit 1; }

# R tells consense to treat the DiWEST trees as rooted
printf "$1\nR\nY\n" | ./evaluation/consense-linux
# rename file so next run is not affected by presence of "outfile"
mv outfile $1\_consense_outputs
consense_tree=$1\_consense_tree
mv outtree $consense_tree
python3 evaluation/fix_tree.py -t $consense_tree

# see "Correction" in the README.md
sed -e 's/Sir/Sio/g' $consense_tree.newick > tmp
sed -e 's/Bar/Bas/g' tmp > $consense_tree.newick
rm tmp

GOLD_TREE=evaluation/tukano_chaconlist15_new.newick
hyp_tree=$consense_tree.newick

# butterflies_hyp=$(./evaluation/quartet_dist -v $hyp_tree $hyp_tree | awk -F '\t' '{print $5}')
# the 5th column of the quartet_dist output is the butterflies in both gold/hypothesis
butterflies_agree=$(./evaluation/quartet_dist-linux -v $hyp_tree $GOLD_TREE | awk -F '\t' '{print $5}')
# if you run quartet_dist on the gold tree compared to itself, # butterflies in the gold = # butterflies in the "hypothesis"
butterflies_gold=$(./evaluation/quartet_dist-linux -v $GOLD_TREE $GOLD_TREE | awk -F '\t' '{print $5}')
butterflies_diff=$(($butterflies_gold - $butterflies_agree))
gqd=$(echo "scale=5; $butterflies_diff / $butterflies_gold" | bc)
echo "$baseline GQD $gqd"

