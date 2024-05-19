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


def generate_rules(args, lang=None):
    LANG_FAMILY = args.language_family
    LANGUAGE = args.language if lang is None else lang
    SCORE_TYPE = args.score_type
    if SCORE_TYPE == 'accuracy':
        s0 = args.accuracy_smooth
        SCORE_TYPE = f'accuracy{s0}'

    OUT_DIR = Path(args.out_dir)

    print(f"processing language {LANGUAGE}")

    # Make word-specific (base) rules, apply recursive minimal generalization
    if args.learn_rules:
        dat_train = config.dat_train
        if LANGUAGE not in dat_train['language'].unique():
            raise ValueError(f"language {LANGUAGE} cannot be found in train data")
        dat_train = dat_train[dat_train['language'] == LANGUAGE]
        print('Base rules ...')
        # R_base = [base_rule(w1, w2) for (w1, w2) in zip(dat_train['stem'], dat_train['output'])]
        R_base = [rule for (w1, w2) in zip(dat_train['stem'], dat_train['output']) for rule in unit_base_rules(w1, w2)]

        if args.cross_contexts:
            R_base = cross_contexts(R_base)

        base_rules = pd.DataFrame({'rule': [str(rule) for rule in R_base]})
        base_rules.to_csv(
            OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_base.tsv',
            index=False,
            sep='\t')

        R_ftr = [FtrRule.from_segrule(R) for R in R_base]
        R_all = mingen.generalize_rules_rec(R_ftr)
        print(f'Learned {len(R_all)} rules')

        rules = pd.DataFrame({
            'rule_idx': [idx for idx in range(len(R_all))],
            'rule': [str(rule) for rule in R_all]
        })
        rules['rule_regex'] = [repr(rule) for rule in R_all]
        rules['rule_len'] = [len(x) for x in rules['rule']]
        rules.to_csv(
            OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_out.tsv',
            index=False,
            sep='\t')

        # Compute hits and scope and for each learned rule
        # Hit and scope on train data
        R_all = [FtrRule.from_str(rule) for rule in rules['rule']]
        hits_all, scope_all = scoring.score_rules(R_all, lang=LANGUAGE)
        rules['hits'] = hits_all
        rules['scope'] = scope_all

        rules.to_csv(
            OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_scored.tsv',
            sep='\t',
            index=False)

    # Score rules for confidence or accuracy
    if args.score_rules:
        rules = pd.read_csv(
            OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_scored.tsv', sep='\t')

        # Confidence
        # todo: adjustable alpha
        if SCORE_TYPE == 'confidence':
            rules['confidence'] = [scoring.confidence(h, s) \
                for (h,s) in zip(rules['hits'], rules['scope'])]

        # Smoothed accuracy: hits / (scope + s0)
        if re.match('^accuracy', SCORE_TYPE):
            s0 = args.accuracy_smooth
            rules[SCORE_TYPE] = [float(h)/(float(s) + s0) \
                for (h, s) in zip(rules['hits'], rules['scope'])]

        rules.to_csv(
            OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_scored.tsv',
            sep='\t',
            index=False)

    # Prune rules that are bounded by more general rules or have scores <= 0
    if args.prune_rules:
        rules = pd.read_csv(
            OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_scored.tsv', sep='\t')

        rules_max = pruning.prune_rules(rules, SCORE_TYPE)
        rules_max.to_csv(
            OUT_DIR / f'{LANG_FAMILY}/{LANGUAGE}_rules_pruned_{SCORE_TYPE}.tsv',
            sep='\t',
            index=False)


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
        '--learn_rules',
        action='store_true',
        default=True,
        help='recursive minimal generalization')
    parser.add(
        '--cross_contexts',
        action='store_true',
        default=False,
        help='make cross-context base rules (aka dopplegangers)')
    parser.add(
        '--score_rules',
        action='store_true',
        default=True,
        help='confidence or accuracy')
    parser.add(
        '--prune_rules',
        action='store_true',
        default=True,
        help='maximal rules by generality, score, and length')
    parser.add(
        '--score_type',
        type=str,
        choices=['confidence', 'accuracy'],
        default='confidence')
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
            generate_rules(args, lang=lang)
    else:
        generate_rules(args)
