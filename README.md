# Automating Sound Change Prediction for Phylogenetic Inference: A Tukanoan Case Study

This is the code for the paper "Automating Sound Change Prediction for Phylogenetic Inference: A Tukanoan Case Study."


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


## Data Preparation
The 15-language dataset from https://github.com/lexibank/chacontukanoan is stored at data_15/input and contains:
* phonetic transcriptions
* cognacy annotations
* Proto-Tukanoan reconstructions

The 21-language dataset from https://github.com/lingpy/tukano-paper is used in Chacon and List 2016 for their phylogeny (which we treat as the gold tree) and contains:
* expert-induced sound laws (tukano-paper/D_changes.tsv)
* expert intermediate sound change predictions (tukano-paper/D_reflexes.tsv)


## Baseline models

We have two baseline models: cognacy and shared innovations (see Section 4.3 in the paper).

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

```
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
```


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