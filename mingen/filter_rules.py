# -*- coding: utf-8 -*-
# Ex. python 02_run_model.py --language eng --learn_rules --score_rules --prune_rules --rate_wugs

# todo: scalar features, phonology, impugnment, ...

import configargparse, pickle, sys
from pathlib import Path
import pandas as pd

import config
from features import *
from rules import *
import mingen
import scoring
import pruning
from phtrs import config as phon_config


def filter_rules_old(args, lang=None):
    LANG_FAMILY = args.language_family
    LANGUAGE = args.language if lang is None else lang
    SCORE_TYPE = args.score_type
    if SCORE_TYPE == 'accuracy':
        s0 = args.accuracy_smooth
        SCORE_TYPE = f'accuracy{s0}'

    OUT_DIR = Path(args.out_dir)

    print(f"processing language {LANGUAGE}")

    rules = pd.read_csv(
        OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_pruned_{SCORE_TYPE}.tsv',
        sep='\t')

    rules = rules[~rules['rule_regex'].str.contains('∅ -> ∅ /')]
    rules = rules[rules['rule_regex'].str[3] == '>']

    rules.to_csv(OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_pruned_{SCORE_TYPE}_filtered.tsv',
                 sep='\t', index=False)


def filter_rules(args, lang=None):
    LANG_FAMILY = args.language_family
    LANGUAGE = args.language if lang is None else lang
    SCORE_TYPE = args.score_type
    if SCORE_TYPE == 'accuracy':
        s0 = args.accuracy_smooth
        SCORE_TYPE = f'accuracy{s0}'

    OUT_DIR = Path(args.out_dir)

    print(f"processing language {LANGUAGE}")

    rules = pd.read_csv(
        OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_pruned_{SCORE_TYPE}.tsv',
        sep='\t')

    rules = rules[rules["hits"] / rules["scope"] >= 0.6]
    # rules = rules[rules["accuracy10.0"] > 0.1]

    rules.to_csv(OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_pruned_{SCORE_TYPE}_filtered.tsv',
                 sep='\t', index=False)


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add(
        '--language_family',
        type=str,
        choices=['polynesian'],
        default='polynesian')
    parser.add(
        '--language',
        type=str,
        default='ALL')
    parser.add(
        '--score_type',
        type=str,
        choices=['confidence', 'accuracy'],
        default='accuracy')
    parser.add(
        '--accuracy_smooth',
        type=float,
        default=10.0,
        help='denominator for smoothed accuracy')
    parser.add(
        '--data_dir',
        type=str,
        default='data'
    )
    parser.add(
        '--out_dir',
        type=str,
        default='output'
    )

    args = parser.parse_args()

    LANG_FAMILY = args.language_family
    DATA_DIR = Path(args.data_dir)

    # Import config (as created by 01_prepare_data)
    config_save = pd.read_pickle(
        open(DATA_DIR / f'{LANG_FAMILY}_config.pkl', 'rb'))
    for key, val in config_save.items():
        setattr(config, key, val)
    phon_config.init(config_save)

    if args.language == 'ALL':
        langs = config.dat_train['language'].unique()
        for lang in langs:
            filter_rules(args, lang=lang)
    else:
        filter_rules(args)
