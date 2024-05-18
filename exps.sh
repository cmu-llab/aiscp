#!/bin/bash
[ $# -eq 0 ] && { echo "Usage: $0 <score file> <result dir> <eval script step2> <eval script step3>"; exit 1; }
# bash exps.sh scores.txt res evaluation/evaluate_win.sh evaluation/evalstep3_win.sh

SCOREFILE=$1
RESDIR=$2
EVALSCRIPT_step2=$3
EVALSCRIPT_step3=$4
echo "NEW RUN OF exps.sh !!!!!!!!!!!!!" >> $SCOREFILE
mkdir -p $RESDIR

# Create NN traning data
echo "Creating training data"
python3 phone_graph/generate_training_data.py

# Train NN models
echo "=== Model Training ==="
echo "Training 1-layer model"
python3 phone_graph/train_nn.py --vec_bitext phone_graph/train_data.pkl --save_file phone_graph/mod_1L.pt --model_type 1L
echo "Training 4-layer model"
python3 phone_graph/train_nn.py --vec_bitext phone_graph/train_data.pkl --save_file phone_graph/mod_4L.pt --model_type 4L
echo "Training 8-layer model"
python3 phone_graph/train_nn.py --vec_bitext phone_graph/train_data.pkl --save_file phone_graph/mod_8res.pt --model_type 8res
echo "Training 16-layer model"
python3 phone_graph/train_nn.py --vec_bitext phone_graph/train_data.pkl --save_file phone_graph/mod_16res.pt --model_type 16res

# Now for step 2
echo "=== Step 2 experiments ==="
# Usage: $0 <path to NN> <dir to store R_trees> <eval script> <file for scores> <keep-oov option>
echo "Result for step 2 1-layer NN, discarding OOV" >> $SCOREFILE 
bash step2.sh phone_graph/mod_1L.pt $RESDIR/step2_mod_1L $EVALSCRIPT_step2 $SCOREFILE "" 
echo "Result for step 2 4-layer NN, discarding OOV" >> $SCOREFILE
bash step2.sh phone_graph/mod_4L.pt $RESDIR/step2_mod_4L $EVALSCRIPT_step2 $SCOREFILE ""
echo "Result for step 2 8-layer NN, discarding OOV" >> $SCOREFILE
bash step2.sh phone_graph/mod_8res.pt $RESDIR/step2_mod_8res $EVALSCRIPT_step2 $SCOREFILE ""
echo "Result for step 2 16-layer NN, discarding OOV" >> $SCOREFILE
bash step2.sh phone_graph/mod_16res.pt $RESDIR/step2_mod_16res $EVALSCRIPT_step2 $SCOREFILE ""
echo "Result for step 2 1-layer NN, keeping OOV" >> $SCOREFILE
bash step2.sh phone_graph/mod_1L.pt $RESDIR/step2_mod_1L_kOOV $EVALSCRIPT_step2 $SCOREFILE "--keep-oov"
echo "Result for step 2 4-layer NN, keeping OOV" >> $SCOREFILE
bash step2.sh phone_graph/mod_4L.pt $RESDIR/step2_mod_4L_kOOV $EVALSCRIPT_step2 $SCOREFILE "--keep-oov"
echo "Result for step 2 8-layer NN, keeping OOV" >> $SCOREFILE
bash step2.sh phone_graph/mod_8res.pt $RESDIR/step2_mod_8res_kOOV $EVALSCRIPT_step2 $SCOREFILE "--keep-oov"
echo "Result for step 2 16-layer NN, keeping OOV" >> $SCOREFILE
bash step2.sh phone_graph/mod_16res.pt $RESDIR/step2_mod_16res_kOOV $EVALSCRIPT_step2 $SCOREFILE "--keep-oov"

# Now for step 3
echo "=== Step 3 experiments ==="
# Usage: $0 <path to NN> <dir to store R_trees> <eval script> <file for scores> <filter-rules option>
echo "Result for step 3 1-layer NN, unfiltered rules" >> $SCOREFILE
bash step3.sh phone_graph/mod_1L.pt $RESDIR/step3_mod_1L $EVALSCRIPT_step3 $SCOREFILE ""
echo "Result for step 3 4-layer NN, unfiltered rules" >> $SCOREFILE
bash step3.sh phone_graph/mod_4L.pt $RESDIR/step3_mod_4L $EVALSCRIPT_step3 $SCOREFILE ""
echo "Result for step 3 8-layer NN, unfiltered rules" >> $SCOREFILE
bash step3.sh phone_graph/mod_8res.pt $RESDIR/step3_mod_8res $EVALSCRIPT_step3 $SCOREFILE ""
echo "Result for step 3 16-layer NN, unfiltered rules" >> $SCOREFILE
bash step3.sh phone_graph/mod_16res.pt $RESDIR/step3_mod_16res $EVALSCRIPT_step3 $SCOREFILE ""
echo "Result for step 3 1-layer NN, filtered rules" >> $SCOREFILE
bash step3.sh phone_graph/mod_1L.pt $RESDIR/step3_mod_1L_kOOV $EVALSCRIPT_step3 $SCOREFILE "--filter-rules"
echo "Result for step 3 4-layer NN, filtered rules" >> $SCOREFILE
bash step3.sh phone_graph/mod_4L.pt $RESDIR/step3_mod_4L_kOOV $EVALSCRIPT_step3 $SCOREFILE "--filter-rules"
echo "Result for step 3 8-layer NN, filtered rules" >> $SCOREFILE
bash step3.sh phone_graph/mod_8res.pt $RESDIR/step3_mod_8res_kOOV $EVALSCRIPT_step3 $SCOREFILE "--filter-rules"
echo "Result for step 3 16-layer NN, filtered rules" >> $SCOREFILE
bash step3.sh phone_graph/mod_16res.pt $RESDIR/step3_mod_16res_kOOV $EVALSCRIPT_step3 $SCOREFILE "--filter-rules"

# Now for step 1
echo "=== Step 1 experiments ==="
# Usage: $0 <dir to store R_trees> <eval script> <file for scores> <keep-oov option>
echo "Result for step 1, discarding OOV" >> $SCOREFILE
bash step1.sh $RESDIR/step1 $EVALSCRIPT_step2 $SCOREFILE ""
echo "Result for step 1, keeping OOV" >> $SCOREFILE
bash step1.sh $RESDIR/step1_kOOV $EVALSCRIPT_step2 $SCOREFILE "--keep-oov"

# Now for freeway style
echo "=== Trying freeway style ==="
# Just like step1 args
echo "Result for freeway style, discarding OOV" >> $SCOREFILE
bash freeway.sh $RESDIR/freeway $EVALSCRIPT_step2 $SCOREFILE ""
echo "Result for freeway style, keeping OOV" >> $SCOREFILE
bash freeway.sh $RESDIR/freeway_kOOV $EVALSCRIPT_step2 $SCOREFILE "--keep-oov"

# Now for direct paths with now intermediate nodes whatsoever
echo "Result for using direct paths for correspondences with no intermediate nodes" >> $SCOREFILE
bash dirpaths.sh $RESDIR/dirpath $EVALSCRIPT_step2 $SCOREFILE

echo "Experiments not covered by this script yet:"
echo "- congacy matrix baseline"
echo "- shared innovations matrix baseline"
echo "- initializing tree with output from cognacy or shared innovations matrix"
