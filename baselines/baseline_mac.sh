#!/bin/bash
# Usage: $0 <criterion>
[ $# -lt 1 ] && { echo "Usage: $0 <criterion>"; exit 1; }

CRITERION=$1
if [ $CRITERION == "cognacy" ]
then
    python cognacy_matrix.py
    DIR=cognacy
elif [ $CRITERION == "shared_innovations" ]
then
    python shared_innovations_matrix.py
    DIR=shared_innovations
else
    echo "Phylogenetic inference criterion $1 not supported"
    exit 1
fi

mkdir $DIR
# to hide output: printf "2\nY\n" | ./penny
printf "Y\n" | ./penny
mv outtree $DIR/pre_consensus_outtree
mv outfile $DIR/pre_consensus_outfile
echo "Successfully ran phylogenetic inference with penny" 

# to get a consensus tree because there could exist many trees with the same minimum cost
printf "$DIR/pre_consensus_outtree\n2\nY\n" | ./consense
mv outtree $DIR/consensus_outtree
mv outfile $DIR/consensus_outfile
