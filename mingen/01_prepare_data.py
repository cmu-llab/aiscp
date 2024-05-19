# -*- coding: utf-8 -*-

import configargparse, pickle, re, sys
from pathlib import Path
import pandas as pd

import config
from phtrs import config as phon_config, features, str_util

# String environment
config.epsilon = 'ϵ'
config.bos = '⋊'
config.eos = '⋉'
config.zero = '∅'
config.save_dir = Path(__file__).parent / 'data'
phon_config.init(config)


def format_strings(dat, extra_seg_fixes=None):
    seg_fixes = config.seg_fixes
    if extra_seg_fixes is not None:
        seg_fixes = seg_fixes | extra_seg_fixes

    # Fix transcriptions (conform to phonological feature set)
    dat['stem'] = [str_util.retranscribe(x, seg_fixes) \
        for x in dat['wordform1']]
    dat['output'] = [str_util.retranscribe(x, seg_fixes) \
        for x in dat['wordform2']]
    dat['stem'] = [str_util.add_delim(x) for x in dat['stem']]
    dat['output'] = [str_util.add_delim(x) for x in dat['output']]

    # Remove prefix from output
    if config.remove_prefix is not None:
        dat['output'] = [re.sub('⋊ ' + config.remove_prefix, '⋊', x) \
            for x in dat['output']]
    return dat


def segment_string(word):
    return ' '.join('∅' if char == '-' else char for char in word.split())


# Select language and transcription conventions
parser = configargparse.ArgParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser)
parser.add(
    '--alignment_filename',
    type=str,
    default="../alignment/output/alignment_Prototucanoan"
)
parser.add(
    '--language_family',
    type=str,
    choices=['polynesian'],
    default='polynesian')
args = parser.parse_args()
ALIGN_FILE = args.alignment_filename
LANG_FAMILY = args.language_family

wordform_omit = None
# config.seg_fixes = {'t͡ʃ': 'tʃ'}
config.seg_fixes = {}
config.remove_prefix = None

# # # # # # # # # #
# Train
fdat = ALIGN_FILE
# with open(fdat, 'rb') as f:
#     dat_lines = pickle.load(f)
# dat_lines = [(' '.join(src), ' '.join(tgt), lang) for tgt, src, lang in dat_lines]
with open(fdat, 'r') as f:
    dat_lines = f.readlines()
dat_lines = [(line.split('/')[0].split('>')[1].strip(), line.split('/')[0].split('>')[0].strip(), line.split('/')[1].strip()) for line in dat_lines]
dat = pd.DataFrame(dat_lines, columns=['wordform1', 'wordform2', 'language'])
# dat = pd.read_csv(fdat, sep='\t', header=None,
#                   names=['wordform1', 'wordform2', 'language'])
dat['wordform1'] = dat['wordform1'].apply(segment_string)
dat['wordform2'] = dat['wordform2'].apply(segment_string)
# special_sym = [r'\*', 'ʰ', '̃', '̆', '̈', '̣', '̥', r'\(', r'\.', r'0', r'\<', r'\?', 'q', r'\|', 'ʔ', '͡', 'β', '’']  # removing temporarily, should deal with them before alignment
# for char in special_sym:
#     dat = dat[~dat['wordform1'].str.contains(char)]
#     dat = dat[~dat['wordform2'].str.contains(char)]

# Format strings and save
dat = format_strings(dat)
dat.to_csv(config.save_dir / f'{LANG_FAMILY}_dat_train.tsv', sep='\t', index=False)
config.dat_train = dat
print('Training data')
print(dat)
print()

# # # # # # # # # #
# Phonological features
segments = set()
for stem in dat['stem']:
    segments |= set(stem.split())
for output in dat['output']:
    segments |= set(output.split())
segments -= {config.bos, config.eos}
segments = [x for x in segments]
segments.sort()
print(f'Segments that appear in training data: '
      f'{segments} (n = {len(segments)})')
print()

# Import features from file
feature_matrix = features.import_features(
    Path(__file__).parent / f'data/{LANG_FAMILY}.ftr',
    segments)

# Fix up features for mingen
ftr_matrix = feature_matrix.ftr_matrix
ftr_matrix = ftr_matrix.drop('sym', axis=1)  # Redundant with X (Sigma*)
config.phon_ftrs = ftr_matrix
config.ftr_names = list(ftr_matrix.columns.values)
config.syms = list(ftr_matrix.index)

# Map from symbols to feature-value dictionaries and feature vectors
config.sym2ftrs = {}
config.sym2ftr_vec = {}
for i, sym in enumerate(config.syms):
    ftrs = config.phon_ftrs.iloc[i, :].to_dict()
    config.sym2ftrs[sym] = ftrs
    config.sym2ftr_vec[sym] = tuple(ftrs.values())

# # # # # # # # # #
# Save config
config_save = {}
for key in dir(config):
    if re.search('__', key):
        continue
    config_save[key] = getattr(config, key)

with open(config.save_dir / f'{LANG_FAMILY}_config.pkl', 'wb') as f:
    pickle.dump(config_save, f)
