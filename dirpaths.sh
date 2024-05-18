#!/bin/bash
[ $# -eq 0 ] && { echo "Usage: $0 <dir to store R_trees> <eval script> <file for scores>"; exit 1; }

RESULTDIR=$1
EVALSCRIPT=$2
SCOREFILE=$3

python3 formatting/s5_shortestpaths.py --out-graph-pkl phone_graph.pkl
python3 formatting/alt_s3_dirpaths.py --infile tukano-paper/D_reflexes.tsv --outfile tukano-paper/S_changes.tsv
python3 tukano-paper/S_compile_data.py --change-tsv tukano-paper/S_changes.tsv --reflex-tsv tukano-paper/D_reflexes.tsv --lang-tsv tukano-paper/D_languages.tsv --graph-pkl phone_graph.pkl
rm -f R_diwest-*
python3 tukano-paper/C_analyze.py runs=100000 hp matrix=diwest
rm -rf $RESULTDIR
mkdir -p $RESULTDIR
mv R_diwest-* $RESULTDIR
bash $EVALSCRIPT $RESULTDIR/R_diwest-*
bash $EVALSCRIPT $RESULTDIR/R_diwest-* | grep GQD >> $SCOREFILE
