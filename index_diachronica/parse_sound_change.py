import re
from tipa_cleanup import *


def if_subtitle(line):
    pattern = r'\d. family name: (.*?) to (.*?)\n'
    match_result = re.search(pattern, line)
    if match_result != None:
        return match_result.group(1), match_result.group(2)
    return None

def parse_line(line):
    '''
       capture group
       return a list of sound change rules from the line
    '''
    if line.count("$") > 1: return None
    if not line.endswith("\\\\"): line += "\\\\"
    rules = []
    result = []
    if line.count("\\change\\") != 1:
        tokens = line.split(" \\change\\ ")
        tokens[-1] = tokens[-1][:-1]
        for i in range(len(tokens)-1):
            rule = tokens[i] + " \\change\\ " + tokens[i+1] + "\\\\"
            rules.append(rule)
    else:
        rules.append(line)
    
    pattern = r"\\(.*?) \\change\\ \\(.*?)( / .*?)?\\\\"
    for rule in rules:
        match_result = re.search(pattern, rule)
        if match_result != None:
            r = seperate_sound(match_result.group(1), match_result.group(2), match_result.group(3))
            for rule in r:
                result.append(rule)
    return None if result == [] else result

def seperate_sound(source, target, context):
    '''
        return a list of sound changes from one line
    '''
    #return [(source, target, context)]
    result = []
    in_ipa_source = False
    in_ipa_target = False
    source += "\\\\"
    target += "\\\\"
    if source.find("ipa") != -1:
        in_ipa_source = True
        ipa_source = re.search("ipa{(.*?)}\\\\", source).group(1)
        if ipa_source == None: print("not match")
        else: source = ipa_source
    if target.find("ipa") != -1:
        in_ipa_target = True
        ipa_target = re.search("ipa{(.*?)}\\\\", target).group(1)
        if ipa_target == None: raise Exception(f"ipa parsing error in target {target}")
        else: target = ipa_target

    source = source.replace(r"\super ", r"\super")
    source = source.replace(" ", ",")
    source = source.replace(r"\super", r"\super ")
    
    target = target.replace(r"\super ", r"\super")
    target = target.replace(" ", ",")
    target = target.replace(r"\super", r"\super ")
    
    if len(source.split(",")) > 0: sourcelist = source.split(",")
    else: sourcelist = [source]
    if len(target.split(",")) > 0: targetlist = target.split(",")
    else: targetlist = [target]
    print(sourcelist, targetlist)
    
    # make them equal len
    while len(sourcelist) != len(targetlist):
        if len(sourcelist) == 1:
            sourcelist = sourcelist.append(sourcelist[0])
        elif len(targetlist) == 1:
            targetlist = targetlist.append(targetlist[0])
        else:
            #print(f"input output does not match for {source}>{target} in {context}")
            raise Exception("what")
    
    
    for s, t in zip(sourcelist, targetlist):
        s = "\ipa{" + s + "}" if in_ipa_source else s
        t = "\ipa{" + t + "}" if in_ipa_target else t
        result.append((s, t, context))
    return result    
    

def if_title(line):
    pattern = r"\d. family name: (.*?)"
    return None
    #TODO: find family name

def parse_rules(path):
    rules = []
    unmatched = []
    file = open(path, "r")
    lines = file.readlines()
    index = 0
    daughter_name = ""
    proto_name = ""
    #family = "not implemented yet"
    match = 0
    unmatch = 0
    while index < len(lines):
        try:
            is_title = if_title(lines[index])
            if is_title != None:
                family = is_title
                index += 1
            else:
                is_subtitle = if_subtitle(lines[index])
                if is_subtitle != None:
                    daughter_name, proto_name = is_subtitle
                    index += 1
                else:
                    lines[index] = lines[index].replace("--- ", "")
                    
                    is_legal_rule = parse_line(lines[index])
                    #print(is_legal_rule)
                    if is_legal_rule != None:
                        match += 1
                        for rule in is_legal_rule:
                            source, target = rule[0], rule[1]
                            context = rule[2] if rule[2] != None else "N/A"
                            if target == "O\\\\\\": target = "O"
                            source, target = tipa_to_unicode(source), tipa_to_unicode(target)
                            # TODO: also clean up context! but context has C for consonant which conflict with letter map
                            source = source.replace("\\", "")
                            target = target.replace("\\", "")
                            rule = [source, target, context, daughter_name, proto_name]
                            rules.append(rule)
                            
                    else:
                        unmatched.append(lines[index])
                        unmatch += 1
            index += 1
        except:
            index += 1
            
    print(match, unmatch)
    with open("output/index_diachronica_unicode", "w", encoding="utf-8") as f:
        f.writelines('\n'.join(''.join(str(r)) for r in rules))
    
    with open("output/index_diachronica_output.csv", "w") as f:
        f.write("index,source,target,context,proto language,daughter language\n")
        for i, rule in enumerate(rules):
            [source, target, context, proto_name, daughter_name] = rule
            f.write(f'{i + 1},{source},{target},{context},{proto_name},{daughter_name}\n')

    with open("output/index_diachronica_not_match_unicode", "w", encoding="utf-8") as f:
        f.writelines(''.join(str(u) for u in unmatched))


parse_rules("data/rules.txt")
