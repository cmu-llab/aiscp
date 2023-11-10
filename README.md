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

* penny - used for phylogenetic inference
    Assuming PHYLIP is downloaded, ```cp PATH_TO_PHYLIP/phylip-3.695/exe/consense.app/Contents/MacOS/penny evaluation``` (OS X) or ```cp PATH_TO_PHYLIP/phylip-3.697/exe/penny evaluation/penny-linux``` (linux)

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