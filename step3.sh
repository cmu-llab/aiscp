#!/bin/bash
[ $# -eq 0 ] && { echo "Usage: $0 <path to NN> <dir to store R_trees> <eval script> <file for scores> <filter-rules option>"; exit 1; }

NNPATH=$1
RESULTDIR=$2
EVALSCRIPT=$3
SCOREFILE=$4
FILTRULES=$5 # May be --filter-rules or ""

python3 phone_graph/create_phone_graph.py --model-path $NNPATH
python3 formatting/s2_wlout2clreflex.py --indir mingen/output/tukanoan --outfile tukano-paper/S_reflexes.tsv $FILTRULES
python3 formatting/s3_clreflex2change.py --infile tukano-paper/S_reflexes.tsv --outfile tukano-paper/S_changes.tsv
python3 tukano-paper/S_compile_data.py --change-tsv tukano-paper/S_changes.tsv --reflex-tsv tukano-paper/S_reflexes.tsv --lang-tsv tukano-paper/D_languages.tsv --graph-pkl phone_graph.pkl
#rm -f R_diwest-*
python3 tukano-paper/C_analyze.py runs=100000 hp matrix=diwest
#rm -rf $RESULTDIR
#mkdir -p $RESULTDIR
mv R_diwest-* $RESULTDIR
bash $EVALSCRIPT $RESULTDIR/R_diwest-*
bash $EVALSCRIPT $RESULTDIR/R_diwest-* | grep GQD >> $SCOREFILE
