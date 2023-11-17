#!/bin/bash

echo "Calculating GQD for baselines"
# GQD = resolved quartets (butterflies) differing in both trees / butterflies in gold

# cognacy - 16 varieties
# shared innovations - 21 varieties
GOLD_COGNACY_TREE=evaluation/tukano_chaconlist15_full.newick   # our tree not encoded, gold not encoded
GOLD_SHAREDINNO_TREE=evaluation/tukano_chaconlist.newick  # both our tree and the gold are encoded

for baseline in cognacy shared_innovations
do
    # format PHYLIP tree output correctly to work with quartet_dist
    python evaluation/fix_tree.py --tree baselines/$baseline/consensus_outtree

    if [ $baseline == "cognacy" ]
    then
        GOLD_TREE=$GOLD_COGNACY_TREE
    else
        GOLD_TREE=$GOLD_SHAREDINNO_TREE
    fi

    hyp_tree=baselines/$baseline/consensus_outtree.newick
    # butterflies_hyp=$(./evaluation/quartet_dist -v $hyp_tree $hyp_tree | awk -F '\t' '{print $5}')
    # the 5th column of the quartet_dist output is the butterflies in both gold/hypothesis
    butterflies_agree=$(./evaluation/quartet_dist -v $hyp_tree $GOLD_TREE | awk -F '\t' '{print $5}')
    # if you run quartet_dist on the gold tree compared to itself, # butterflies in the gold = # butterflies in the "hypothesis"
    butterflies_gold=$(./evaluation/quartet_dist -v $GOLD_TREE $GOLD_TREE | awk -F '\t' '{print $5}')
    # butterflies_both=$(($butterflies_hyp + $butterflies_gold))
    # butterflies_diff=$(($butterflies_both - 2 * $butterflies_agree))

    butterflies_diff=$(($butterflies_gold - $butterflies_agree))
    gqd=$(echo "scale=5; $butterflies_diff / $butterflies_gold" | bc)
    echo "$baseline GQD $gqd"
done
