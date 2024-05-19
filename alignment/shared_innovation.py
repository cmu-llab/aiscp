import re
import pandas as pd
# from util.align import needleman_wunsch
import feature_edit_distance
import json

"""
Gets shared innovations between a protoform and daughter form using Needleman-Wunsch alignment (Needleman and Wunsch 1970; Hamilton and Ben-Hur 2019)
to identify insertions, deletions, and substitutions between a protoform and daughter form

In the format expected by PHYLIP
https://evolution.genetics.washington.edu/phylip/doc/main.html

Adapted from construct_shared_innov_matrix.py (Maggie Huang)
"""


"""
json output may contain unprintable characters
\u014b = ŋ
\u0263 = ɣ
\u0294 = ʔ

"""

import pickle

class CognacyError(Exception):
    """Daughter is too innovative (i.e. too different from protoform). Suggests partial cognacy"""
    pass

class EnvironmentError(Exception):
    pass

"""
daughter: string in IPA
protoform: string in IPA
"""
def get_innovations_with_little_environment(daughter, protoform):
    daughter, protoform = daughter.strip(), protoform.strip() # remove weird spaces
    innovations = set()

    # get alignment -> then zip the aligned strings -> innovations are pairs where they differ
    # get the aligned strings, not the distance score

    alignment = feature_edit_distance.nw_feature(daughter, protoform,   
                                                         feature_edit_distance.dst.feature_edit_distance)
    aligned_daughter, aligned_proto = alignment

    aligned_daughter.insert(0, "#")
    aligned_daughter.insert(-1, "#")
    aligned_proto.insert(0, "#")
    aligned_proto.insert(-1, "#")

    for i in range(1, len(aligned_daughter)-1):
        if aligned_daughter[i] != aligned_proto[i]:
            environment = get_environment(aligned_daughter, aligned_proto, i)
            sound_change = (aligned_daughter[i], aligned_proto[i])
            innovations.add((environment, sound_change))


    # if number of innovations > len(protoform): exclude the daughter from consideration
    #@TODO: threshold for reduplication and other wierd sound change
    if len(innovations) > len(protoform):
        raise CognacyError(daughter, protoform)
    #print(innovations)
    
    return innovations

def get_alignment(daughter, protoform):
    alignment = feature_edit_distance.nw_feature(daughter, protoform,
                                                 feature_edit_distance.dst.feature_edit_distance)
    # aligned_daughter, aligned_proto = trim_partial_cognate(*(alignment))
    aligned_daughter, aligned_proto = alignment
    return aligned_daughter, aligned_proto


def get_environment(daughter, proto, index):
    forward_step = 1
    backward_step = 1
    before = ""
    after = ""
    while index - forward_step >= 0:
        if daughter[index-forward_step] != "-":
            before = daughter[index-forward_step]
            break
        forward_step += 1
    while index + backward_step < len(daughter):
        if daughter[index+backward_step] != "-":
            after = daughter[index+backward_step]
            break
        backward_step += 1       

    if before == "" or after == "":
        raise EnvironmentError("no enviroment")
    return (before, after)

# need to input aligned daughter and proto with equal length
def trim_partial_cognate(daughter, proto):
    found_d, indel_start_d, indel_end_d = get_consecutive_indel(daughter, 2)
    found_p, indel_start_p, indel_end_p = get_consecutive_indel(proto, 2)
    if found_d:
        daughter = daughter[:indel_start_d] + daughter[indel_end_d:]
        proto = proto[:indel_start_d] + proto[indel_end_d:]
    elif found_p:
        daughter = daughter[:indel_start_p] + daughter[indel_end_p:]
        proto = proto[:indel_start_p] + proto[indel_end_p:]
    return daughter, proto

def get_consecutive_indel(seq, length):
    conseq_count = 0
    begin = 0
    end = 0
    index = 0
    while index < len(seq):
        if seq[index] == "-":
            begin = index
            conseq_count += 1
            while index+conseq_count < len(seq) and seq[index+conseq_count] == "-":
                conseq_count += 1
            if conseq_count >= length:
                return True, begin, begin + conseq_count
            else:
                index = begin + conseq_count
                conseq_count = 0
        else:
            index += 1
    return False, begin, end
 
    
def innovation_tostring(innovation):
    environment = innovation[0]
    change = innovation[1]
    result = [change[0], " > ", change[1], " / ", environment[0], " _ ", environment[1]]
    return "".join(result)
    

def get_proto_sound(innovation):
    return innovation[0][0]

def get_daughter_sound(innovation):
    return innovation[0][1]


def run(input, PROTOLANG):
    # skip header because header is tab delimited
    columns = pd.read_csv(input, nrows=1, sep='\t').columns.tolist()
    data = pd.read_csv(input, sep='\t', skiprows=1, names=columns)
    data = data.set_index('#id')
    language_to_rules = {}
    rules_to_language = {}
    sound_change_table = {}
    alignments = []
    
    # feature_edit_distance.update_weights(syl, cons)

    # maps innovation -> set of langs that contain the innovation
    # if the set size is > 1 then the innovation is shared among at least 2 daughters
    shared_innovations = {}
    daughters = data.columns[data.columns != PROTOLANG]
    for index, row in data.iterrows():
        # this column will constantly get overwritten
        protoform = str(row[PROTOLANG])

        # skip rows without a protoform
        if protoform == "?" or protoform == "-":
            continue
        if protoform[0] == "*":
            protoform = protoform[1:]
        try:
            for daughter in daughters:
                # skip entries where the language does not have something that belongs to the cognate class
                # print(row[daughters])
                if row[daughter] != "-" and row[daughter] != "?":
                    daughter_form = row[daughter]
                    # / separates syllables
                    # ex: te/au/ (Nukuoro) is te au
            
                    # for polynesian: daughter_form = daughter_form.replace("/", "").strip()
                    # for tuknoan hard code version
                    daughter_form = daughter_form.replace("ú/", "").strip()
                    daughter_form = daughter_form.replace("í/", "").strip()
                    # replace engma bigram with IPA version
                    daughter_form = daughter_form.replace("ng", "ŋ")

                    innovations = get_innovations_with_little_environment(str(daughter_form), protoform)
                    d, p  = get_alignment(daughter_form, protoform)
                    word = [d, p, daughter]
                    alignments.append(word)
                    for innovation in innovations:
                        if innovation not in shared_innovations:
                            shared_innovations[innovation] = {daughter}
                        else:
                            shared_innovations[innovation].add(daughter)

                        # construct language to sound change rule json
                        inno = innovation_tostring(innovation)
                        proto_sound = get_proto_sound(innovation)
                        daughter_sound = get_daughter_sound(innovation)
                        language = daughter

                        if proto_sound not in ["-", "#"]:
                            if proto_sound not in sound_change_table:
                                sound_change_table[proto_sound] = dict()
                            
                            if language not in sound_change_table[proto_sound]:
                                sound_change_table[proto_sound][language] = dict()
                                
                            if daughter_sound not in sound_change_table[proto_sound][language]:
                                sound_change_table[proto_sound][language][daughter_sound] = 1
                            else:
                                sound_change_table[proto_sound][language][daughter_sound] += 1

        except CognacyError as e:
            # don't count the shared innovations for this entry
            pass


    dupe_innovations = {}
    for key in shared_innovations:
        if len(shared_innovations[key]) > 1:
            dupe_innovations[key] = shared_innovations[key]
    lines = [f"{len(daughters)} {len(dupe_innovations)}\n"]
    # length of the name of the longest daughter - used for padding
    # longest_daughter = max([len(name) for name in daughters])
    for daughter in daughters:
        # PHYLIP restricts the name of each daughter language to 10 characters
        line = ""
        if len(daughter) > 10:
            line = daughter[:10]
        else:
            line = daughter + " " * (10 - len(daughter))
        line += "\t"
        for key in shared_innovations:
            if daughter in shared_innovations[key]:
                line += "1"
            else:
                line += "0"
        lines.append(line+"\n")
        
        
    lines = []
    pickleline = []
    total_misalignment = 0
    total_insdel = 0
    for align in alignments:
        # PHYLIP restricts the name of each daughter language to 10 characters
        total_misalignment += consonont_to_vowel_alignment(align[0], align[1])
        total_insdel += insdel_count(align[0], align[1])
        line = " ".join(align[0]) + " > " + " ".join(align[1]) + " / " + "".join(align[2]) + '\n'
        pickleline.append([align[0], align[1], align[2]])
        lines.append(line)

    with open(f"output/alignment_Prototucanoan", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"misalignment {total_misalignment}, insdel {total_insdel}")

    # TODO: preprocessing - get rid of dialectal variation

    # TODO: preprocessing (ng -> engma and ' -> glottal stop)
    # turn asterisk into glottal stop - Bouchard-Cote seems to have removed this mistakenly

    # TODO: CVCV alignment

    # if __name__ == "__main__":
    #     # Mayan examples from Campbell 2013
    #     print(get_innovations("inik", "winik"))
    #     print(get_innovations("ŋe:h", "we:w"))
    #     print(get_innovations("tseʔ", "teʔ"))
    
def get_original_protoform(line):
    proto = line.split(">")[1].split("/")[0].strip()
    result = []
    for token in proto:
        if token != "-" and token != " ":
            result.append(token)
    return "".join(result)
    
def compare(file_1, file_2):
    '''
    compare the content in two files line by line
    and generate vertically aligned result
    '''
    F1 = open(file_1, encoding = 'utf-8')
    F2 = open(file_2, encoding = 'utf-8')
    out = open("test", "w", encoding = 'utf-8')
    index1, index2 = 0,0
    f1, f2 = F1.readlines(), F2.readlines()
    while index1 < len(f1) and index2 < len(f2):
        p1 = get_original_protoform(f1[index1])
        p2 = get_original_protoform(f2[index2])
        lan1 = f1[index1].split(">")[1].split("/")[1].strip()
        lan2 = f2[index2].split(">")[1].split("/")[1].strip()
        d1 = f1[index1].split(">")[0].strip()
        d2 = f2[index2].split(">")[0].strip()
        
        if p1 == p2 and lan1 == lan2 and d1 != d2:
            pr1 = f1[index1].split(">")[1].split("/")[0].strip()
            pr2 = f2[index2].split(">")[1].split("/")[0].strip()
            out.write(d1)
            out.write("    ")
            out.write(d2)
            out.write("\n")
            out.write(pr1)
            out.write("    ")
            out.write(pr2)
            out.write("\n\n")
            index1 += 1
            index2 += 1
        else:
            if index2+1 < len(f2) and p1 == get_original_protoform(f2[index2+1]):
                index2 += 1
            elif index1+1 < len(f1) and p2 == get_original_protoform(f1[index1+1]):
                index1 += 1
            else:
                index1 += 1
                index2 += 1
    out.close

def consonont_to_vowel_alignment(daughter, proto):
    count = 0
    for token_d, token_p in zip(daughter, proto):
        if not feature_edit_distance.is_valid_alignment(token_d, token_p):
            # ignore semivowels
            if token_d not in ["j", "w", "ʔ"] and token_p not in ["j", "w", "ʔ"]:
                print(f"{token_d} > {token_p}")
                count += 1
    return count

def insdel_count(daughter, proto):
    return daughter.count("-") + proto.count("-")


input2 = "data_15/tukanoan_cognates_new.csv"
PROTOLANG2 = "Prototucanoan"



# TODO: process pre-laryngealized phoones???
# note that this code does not run right now
run(input2, PROTOLANG2)
file1 = "alignment/output/alignment_Prototucanoan"
file2 = "data_15/expert_alignment.csv"
compare(file1, file2)

'''
(1,1): misalignment 815, insdel 4292 / (1,1): misalignment 45, insdel 4292
(1,2): misalignment 760, insdel 4292
(1,3): misalignment 731, insdel 4292

(10,10): misalignment 746, insdel 4312

(5,3): misalignment 788, insdel 4292
(4,3): misalignment 776, insdel 4292
(3,3): misalignment 766, insdel 4292
(2,3): misalignment 750, insdel 4292
(1,3): misalignment 731, insdel 4292
(5,4): misalignment 775, insdel 4294
(4,4): misalignment 766, insdel 4292
(3,4): misalignment 750, insdel 4292
(2,4): misalignment 731, insdel 4292
(1,4): misalignment 718, insdel 4300 / (1,4): misalignment 28, insdel 4300

                                     / (15,15): misalignment 16, insdel 4312
                                     / (10,10): misalignment 16, insdel 4312

(20,20): misalignment 653, insdel 4613
(25,25): misalignment 653, insdel 4613
(50,50): misalignment 653, insdel 4613
'''
