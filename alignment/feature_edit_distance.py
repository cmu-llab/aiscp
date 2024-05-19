#!/usr/bin/env python
'''
Note: the IPA generated by panphon is under
D:\Python\Anaconda\Lib\site-packages\panphon\data
'''


"""
The Needleman-Wunsch Algorithm
==============================
This is a dynamic programming algorithm for finding the optimal alignment of
two strings.
Example
-------
    >>> x = "GATTACA"
    >>> y = "GCATGCU"
    >>> print(nw(x, y))
    G-ATTACA
    GCA-TGCU
LICENSE
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.
In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
For more information, please refer to <http://unlicense.org/>
"""

import numpy as np
import unicodecsv as csv
import panphon
from panphon.distance import Distance




'''
Important Note
add these two phones into ipa_all.csv under the directory of panphon module
possibly in directory like "D:\Python\Anaconda\Lib\site-packages\panphon\data"
ˀtʰ,-,-,+,-,-,-,-,0,-,+,+,+,+,-,-,-,-,-,-,-,0,-,0,0
ˀkʰ,-,-,+,-,-,-,-,0,-,+,+,-,-,0,-,+,-,+,-,-,0,-,0,0
'''

# class DiachronicDistance(Distance):
#     def __init__(self) -> None:
#         super(panphon.FeatureTable(), self).__init__()
#         file = "manual_weights.csv"
#         # source: https://github.com/dmort27/panphon/blob/master/panphon/featuretable.py - _read_weights
#         with open(file, 'rb') as f:
#             reader = csv.reader(f, encoding='utf-8')
#             next(reader)
#             weights = [float(x) for x in next(reader)]
#         # self.fm.weights = weights
def preprocess_phone(ft, phone):
    if not ft:
        ft = panphon.FeatureTable()

    STOP = {
        'son': -1,
        'cont': -1
    }
    FRICATIVE = {
        'son': -1,
        'cont': 1
    }
    if phone[0] == "*":
        phone = phone[1:]
    elif len(phone) >= 2 and \
        ft.fts(phone[0]) and ft.fts(phone[0]).match(STOP) and ft.fts(phone[1]) and ft.fts(phone[1]).match(FRICATIVE):
        # add ligature to affricates (stop + fricative)
        phone = phone[0] + '͡' + phone[1]

    return phone

def to_ipa_seg(word):
    if " " in word:
        return [preprocess_phone(ft,seg) for seg in word.split()]
    return ft.ipa_segs(word)

def nw_plain(x, y, match = 1, mismatch = 1, gap = 1):
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    F[0,:] = np.linspace(0, -ny * gap, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch
                # t[0] = F[i,j] - mismatch_score(x[i], y[j])
            t[1] = F[i,j+1] - gap
            t[2] = F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            j -= 1
    # Reverse the strings.
    rx = ''.join(rx)[::-1]
    ry = ''.join(ry)[::-1]
    return '\n'.join([rx, ry])

def mismatch_score(x, y, f):
    '''
    takes in two phonemes and a distance function from panphon.distance.Distance()
    return the distance between the two phonemes
    '''
    # add penalty to vowel-to-consonant
    extra_penalty = 0
    try:
        vx = ft.segment_to_vector(x)
        vy = ft.segment_to_vector(y)
        sylx = vx[0]
        consx = vx[2]
        syly = vy[0]
        consy = vy[2]
        if (sylx == '-' and consx == '+' and syly == '+' and consy == '-') or (syly == '-' and consy == '+' and sylx == '+' and consx == '-'):
            extra_penalty = 1000
        
        return f(x,y) + extra_penalty
    except:
        print(f"not in panphon? {x} and {y}")

def insdel_score(x,f):
    '''
    takes in one phoneme and a distance function from panphon.distance.Distance()
    return the insertion / deletion score (insdel penalty)
    '''
    return f("", x)

def nw_feature(x, y, f, insdel = True):
    '''
    take in two forms in string, a distance evaluation function
    optional whether consider insdel score using the function
    return a tuple of two lists containing the alignment
    '''
    # gap penalty initialized to 1 but not used when insdel = True
    gap = 1
    # match = 7 yields relatively ideal result
    match = 7
    x, y = to_ipa_seg(x), to_ipa_seg(y)
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    # TODO: should be an list of insdel costs adding up
    horizontal = [0]
    vertical = [0]
    horizontal_total = 0
    vertical_total = 0
    for a in x:
        horizontal_total -= insdel_score(a, f) if insdel else gap
        horizontal.append(horizontal_total)
    for b in y:
        vertical_total -= insdel_score(b, f) if insdel else gap
        vertical.append(vertical_total)
    # F[:,0] = np.linspace(0, -nx * gap, nx + 1)
    # F[0,:] = np.linspace(0, -ny * gap, ny + 1)
    F[:,0] = horizontal
    F[0,:] = vertical
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:,0] = 3
    P[0,:] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i,j] + match
            else:
                t[0] = F[i,j] - mismatch_score(x[i], y[j], f)
            t[1] = F[i,j+1] - insdel_score(x[i],f) if insdel else F[i,j+1] - gap
            t[2] = F[i+1,j] - insdel_score(y[j],f) if insdel else F[i+1,j] - gap
            tmax = np.max(t)
            F[i+1,j+1] = tmax
            if t[0] == tmax:
                P[i+1,j+1] += 2
            if t[1] == tmax:
                P[i+1,j+1] += 3
            if t[2] == tmax:
                P[i+1,j+1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    while i > 0 or j > 0:
        if P[i,j] in [2, 5, 6, 9]:
            rx.append(x[i-1])
            ry.append(y[j-1])
            i -= 1
            j -= 1
        elif P[i,j] in [3, 5, 7, 9]:
            rx.append(x[i-1])
            ry.append('-')
            i -= 1
        elif P[i,j] in [4, 6, 7, 9]:
            rx.append('-')
            ry.append(y[j-1])
            j -= 1
    # Reverse the strings.
    rx.reverse()
    ry.reverse()
    #rx = ''.join(rx)[::-1]
    #ry = ''.join(ry)[::-1]
    return (rx, ry)



dst = Distance()
ft = panphon.FeatureTable()

def update_weights(syl, cons):
    dst.fm.weights = [syl,1,cons,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    


def is_valid_alignment(token_d, token_p):
    # print(ft.word_array(["cons"], token_d))
    # print(ft.word_array(["cons"], token_p))
    if token_d == "-" or token_p == "-": return True
    if ft.word_array(["cons"], token_d) == ft.word_array(["cons"], token_p):
        return True
    return False

def test():
    x = "manaavaa"
    y = "manawa"
    # print(nw_plain(x, y) + "\n")
    # print(nw_feature(x, y, dst_old.feature_edit_distance))
    # print(nw_feature(x, y, dst.feature_edit_distance))
    # print(nw_feature(x, y, dst_old.weighted_feature_edit_distance))
    print(nw_feature(x, y, dst.weighted_feature_edit_distance))
    
    # print(nw_feature(x, y, dst.weighted_feature_edit_distance, insdel = True))
    # print(nw_feature(x, y, dst.feature_edit_distance, match = 7.5))
    # print(nw_feature(x, y, dst.weighted_feature_edit_distance, match = 7.5))
    # print(nw_feature(x, y, dst.feature_edit_distance, match = 30))
    # print(nw_feature(x, y, dst.weighted_feature_edit_distance, match = 30))

"""
syllabic weight = 5
(['m', 'a', 'n', 'a', 'a', 'v', 'a', 'a'], ['m', 'a', 'n', '-', 'a', '-', 'w', 'a'])
(['m', 'a', 'n', 'a', '-', 'a', 'v', 'a', 'a'], ['m', 'a', 'n', 'a', 'w', '-', '-', '-', 'a'])


syllabic weight = 2
(['m', 'a', 'n', 'a', 'a', 'v', 'a', 'a'], ['m', 'a', 'n', '-', 'a', '-', 'w', 'a'])
(['m', 'a', 'n', 'a', 'a', 'v', 'a', 'a'], ['m', 'a', 'n', '-', 'a', 'w', '-', 'a'])
"""