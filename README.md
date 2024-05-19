# Automating Sound Change Prediction for Phylogenetic Inference: A Tukanoan Case Study

This is the code for the paper "Automating Sound Change Prediction for Phylogenetic Inference: A Tukanoan Case Study."

[Feel free to take a look at our slides from LChange 2023.](lchange-slides.pdf)

# Running our code

## Installation
```pip install -r requirements.txt```

```conda install pynini```

You may need to revise the evaluation scripts to use the version of consense, penny, and quartet_dist for your platform (e.g ./consense-linux)
You could also replace consense, penny, and quartet_dist with a freshly downloaded version specific to your platform. Here are the download instructions:

* consense - used to get a consensus tree
    Install PHYLIP from https://phylipweb.github.io/phylip/getme-new1.html (use wget if needed).
    Follow instructions at https://phylipweb.github.io/phylip/install.html
    Copy consense to evaluation/: ```cp PATH_TO_PHYLIP/phylip-3.695/exe/consense.app/Contents/MacOS/consense evaluation``` (OS X) or ```cp PATH_TO_PHYLIP/phylip-3.697/exe/consense evaluation/consense-linux``` (linux)
    (Do the same for baselines/consense)

* penny - used for phylogenetic inference
    Assuming PHYLIP is downloaded, ```cp PATH_TO_PHYLIP/phylip-3.695/exe/consense.app/Contents/MacOS/penny evaluation``` (OS X) or ```cp PATH_TO_PHYLIP/phylip-3.697/exe/penny evaluation/penny-linux``` (linux)
    (Do the same for baselines/penny)

* quartet_dist - used to calculate GQD (generalized quartet distance)
    On a Linux machine,
    ```conda install -c bioconda tqdist```
    or
    Follow the instructions on https://www.birc.au.dk/~cstorm/software/tqdist/


## Evaluation
To calculate GQD on the tree outputs from our 10 runs, run the following script:
```
./results/reproduce_eval.sh
```
The GQDs will be in results/reproduce.txt.
You may need to revise ./results/reproduce_eval.sh, ./evaluation/evaluate.sh, and ./evaluation/evalstep3.sh to point to the path of consense, penny, and quartet_dist specific to your OS that you installed in the steps above.

#### Correction
* An earlier version of the 15-language tree (evaluation/tukano_chaconlist15.newick) used "Bar" for Barasano and "Sir" for Siona.
    * The "AISCP + ASLI" (Sec 3.3) and cognacy baseline results in the LChange paper can be reproduced with the earlier, incorrect tree.
    * The trees we generated from our 10 runs in results/ use the incorrect abbreviations, so we postcorrect the abbreviations in evaluation/evalstep3.sh
* The correct tree (evaluation/tukano_chaconlist15_new.newick, evaluation/tukano_chaconlist15_new_full.newick) uses "Bas" for Barasano and "Sio" for Siona.
* Our evaluation scripts and results in this repo use the correct tree.
    * The arXiv version reflects these updated results.


## Data Preparation
The 15-language dataset from https://github.com/lexibank/chacontukanoan is stored at data_15/input and contains:
* phonetic transcriptions
* cognacy annotations
* Proto-Tukanoan reconstructions

The 21-language dataset from https://github.com/lingpy/tukano-paper is used in Chacon and List 2016 for their phylogeny (which we treat as the gold tree) and contains:
* expert-induced sound laws (tukano-paper/D_changes.tsv)
* expert intermediate sound change predictions (tukano-paper/D_reflexes.tsv)


## Baseline models

We have two baseline models: cognacy and shared innovations (see Section 4.3 in the paper; lines 1 and 2 in Table 1 of the paper).

```
cd baselines
./baseline.sh cognacy
./baseline.sh shared_innovations
cd ..
./evaluation/evaluate_baseline.sh
```

Note that the cognacy baseline uses the 15 language subset from https://github.com/lexibank/chacontukanoan because the original transcriptions and cognacy annotations from the 21-language dataset in https://github.com/lingpy/tukano-paper (used to induce sound laws and intermediate sound changes) are not available. 

For the shared innovations baseline, we only need the gold sound laws, so we can use the 21-language dataset.


## Index Diachronica (training data for directional weighted feature edit distance)

```
cd index_diachronica
wget https://chridd.nfshost.com/diachronica/index-diachronica.tex?as=text
mv index-diachronica.tex?as=text data/index_diachronica.tex

python extract.py
python ipa_cleanup.py
```

The output will be in index_diachronica/output/index_diachronica_output.csv.


## Learning directional weighted feature edit distance

The script `phone_graph/train_nn.py` is used to train neural networks we use to compute DWFED. Its usage is:

```python3 phone_graph/train_nn.py --vec_bitext <pkl formatted train data> --save_file <output path> --model_type <1L|4L|8L|8res|16res>```

with example usage:

```python3 phone_graph/train_nn.py --vec_bitext phone_graph/train_data.pkl --save_file phone_graph/mod_1L.pt --model_type 1L```

along with other optional hyperparameters outlined in the script itself.


Remember to generate the training data from Index Diachronica first:
```
# Create NN training data
echo "Creating training data"
python3 phone_graph/generate_training_data.py
```

## ASLI (automatic sound law induction) - Sec 3.3

To automatically induce sound laws from protoform-reflex pairs (in data_15/), we use Wilson and Li (2021)'s implementation of Albright and Hayes (2002, 2003)'s
minimal generalization algorithm, originally designed to induce synchronic morphological rules (https://github.com/colincwilson/mingen).
It also uses a simple package for working with phonological features and BOS/EOS markers (https://github.com/colincwilson/phtrs).

* Please install pynini (conda recommended)
* Colin Wilson also shared his original (Java) implementation of mingen, usable from the commandline: https://github.com/colincwilson/MinimalGeneralizationLearner


### Alignment
Before we run ASLI on protoform-reflex pairs, we must align the protoforms and reflexes.
We use a modified version of Needleman-Wunsch alignment so that the substitution cost between two phones is the FED rather than a constant, ensuring similar phones align.

* note: add these two phones into ipa_all.csv under the directory of panphon module, possibly in a directory like "D:\Python\Anaconda\Lib\site-packages\panphon\data"
ˀtʰ,-,-,+,-,-,-,-,0,-,+,+,+,+,-,-,-,-,-,-,-,0,-,0,0
ˀkʰ,-,-,+,-,-,-,-,0,-,+,+,-,-,0,-,+,-,+,-,-,0,-,0,0
'''

```
python3 alignment/shared_innovation.py
```

### ASLI

The output is in mingen/output/tukanoan

```
cd mingen
python3 01_prepare_data.py --language_family tukanoan --alignment-file alignment/output/alignment_Prototucanoan
python3 02_run_model.py
python3 filter_rules.py
```


## Phylogenetic inference
We use Chacon and List (2016)'s parsimony-based phylogenetic inference method (https://github.com/lingpy/tukano-paper).

- `formatting`: creating the sound change transition matrix (recording the cost of sound changes) from the phone graph
    - The scripts do not follow the following order. The actual order depends which experiment we are running. See step{1,2,3}.sh for the actual order.
    - `s2_wlout2clreflex.py`: convert mingen rules to reflexes.tsv
    - `s3_clreflex2change.py`: given sound correspondences, predict intermediate sound changes with the phone graph (AISCP)
    - `s5_shortestpaths.py`: use networkx to get the shortest paths from source to target phone in the phone graph (uses the most recently generated one, phone_graph.pkl)
    - `s4_adjust_weights.py`: naively downweight the weight of a sound change / phone transition based on Index Diachronica (freeway / naive FED downweighting experiment)
    - `include_phones.txt`: the phones we include in our phone graph (we started with panphon’s set and restrict the phone set to only include those with a certain number of diacritics)
    - `diachronic_panphon`: we modified the weights of each feature to reflect diachronic phonology
- `tukano-paper`: their phylogenetic inference pipeline
    - `D_reflexes.tsv`: expert sound correspondences including proto-phoneme and context
        - AISCP: use the expert sound correspondences
        - AISCP + ASLI: the sound correspondences from automatic sound law induction (`S_reflexes.tsv`)
    - `S_changes.tsv`: our predicted intermediate sound changes from AISCP
        - `D_changes.tsv`: expert intermediate sound changes
    - `S_compile_data.py`: reads in intermediate sound changes, reflexes, creates directed graph, encodes the matrix
    - `L_parsimony.py`: parsimony-based phylogenetic inference using the sound change transition matrix (we optimized the code with sets and priority queues)
    - `C_analyze.py`: phylogenetic inference manager


## End-to-end pipeline

The designated script to run our experiments for our paper is `exps.sh`. Its usage is as follows:

```bash exps.sh <score file> <result dir> <eval script step2> <eval script step3>```

with an example usage:

```bash exps.sh scores.txt results evaluation/evaluate.sh evaluation/evalstep3.sh```

This overall script does the following:
- Trains four varieties of neural networks to compute DWFED, by running `phone_graph/train_nn.py`
- Runs AISCP experiments using all neural networks and toggling the heuristic of whether out of vocabulary words are kept, by running `step2.sh` eight distinct times
    - Rows 5-6 of Table 1 in the paper (OOV discarded as standard)
- Runs AISCP + ASLI experiments using all neural networks and toggling whether automatically inferred sound laws are filtered, by running `step3.sh` eight distinct times
    - Rows 7-10 of Table 1 in the paper (rules filtered as standard)
- Runs unweighted FED ablation experiments toggling the heuristic of whether out of vocabulary words are kept, by running `step1.sh` two distinct times
    - Row 3 of Table 1 in the paper (OOV discarded as standard)
- Runs an ablation where FED is weighted by naively downweighting FED between phones for attested sound changes (instead of neurally) and toggling the heuristic of whether out of vocabulary words are kept, by running `freeway.sh` two distinct times
    - Results not presented in the paper but discussed in the final paragraph of section 5.2
- Runs an ablation where direct paths are used instead of intermediate sound changes, by running `dirpaths.sh`
    - Row 4 of Table 1 in the paper

Note: `step2.sh` and `step3.sh`, indicating experiments with AISCP only and with AISCP + ASLI respectively, have two different evaluation scripts. This is because we used a fewer number of Tukanoan languages for experiments with ASLI, due to data limitations (explanation in the paper).

Note: exps.sh takes a long time to run, so make sure to run it in tmux.


### Script for AISCP experiments (gold sound laws, predict sound changes)

The script for AISCP experiments is `step2.sh`. Its usage is as follows:

```bash step2.sh <path to NN> <dir to store R_trees> <eval script> <file for scores> <keep-oov option>```

with example usage:

```bash step2.sh phone_graph/mod_1L.pt results/step2_mod_1L evaluation/evaluate.sh scores.txt ""```

This script:
- Creates a DWFED-based phone graph using `phone_graph/create_phone_graph.py`
- Predicts intermediate sound changes via `formatting/s3_clreflex2change.py`
- Runs the Chacon and List (2016) algorithm via `tukano-paper/S_compile_data.py`and `tukano-paper/C_analyze.py`
- Evaluates via the designated evaluation script


### Script for ASLI experiments (predict sound laws, predict sound changes)

The script for experiments with AISCP and ASLI together is `step3.sh`. Its usage is:

```bash step3.sh <path to NN> <dir to store R_trees> <eval script> <file for scores> <filter-rules option>```

with example usage:

```bash step3.sh phone_graph/mod_1L.pt results/step3_mod_1L_kOOV evaluation/evalstep3.sh scores.txt --filter-rules```

This script:
- Creates a DWFED-based phone graph using `phone_graph/create_phone_graph.py`
- Infers sound laws via `formatting/s2_wlout2clreflex.py`
- Predicts intermediate sound changes via `formatting/s3_clreflex2change.py`
- Runs the Chacon and List (2016) algorithm via `tukano-paper/S_compile_data.py`and `tukano-paper/C_analyze.py`
- Evaluates via the designated evaluation script


### Scripts for ablations

The following scripts were used for ablations.

- FED ablation: `step1.sh`
- direct paths ablation: `dirpaths.sh`
- naive FED downweighting: `freeway.sh`


# Citing our paper

Please cite our paper as follows:

Kalvin Chang, Nathaniel R. Robinson, Anna Cai, Ting Chen, Annie Zhang, and David R. Mortensen. 2023. Automating Sound Change Prediction for Phylogenetic Inference: A Tukanoan Case Study. In *Proceedings of the 4th Workshop on Computational Approaches to Historical Language Change (LChange 2023)*, Singapore.

```
@inproceedings{chang-robinson-cai-etal-2023-aiscp,
    title = "Automating Sound Change Prediction for Phylogenetic Inference: A Tukanoan Case Study",
    author = "Chang, Kalvin and
      Robinson, Nathaniel R. and
      Cai, Anna and
      Chen, Ting and
      Zhang, Annie and
      Mortensen, David R.",
    booktitle = "Proceedings of the 4th Workshop on Computational Approaches to Historical Language Change",
    month = december,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
```