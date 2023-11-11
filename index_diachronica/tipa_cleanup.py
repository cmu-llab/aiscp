# parse tipa_py.py (latex format) into unicode

from tipa_py import *

command_map_hs = get_command_map()
letter_map_hs = get_letter_map()
command_map_py = dict()
letter_map_py = dict()
def generate_command_map():
    global command_map_py
    for line in command_map_hs:
        tipa = line[0]
        repre = line[1]
        command_map_py[tipa] = repre
        
def generate_letter_map():
    global letter_map_py
    for line in letter_map_hs:
        letter_map_py[line[0]] = line[1]

generate_letter_map()
generate_command_map()

def single_check_letter(result, raw):
    start = 0
    end = 1
    count = 0
    end_mark = 0
    while start < len(raw):
        while end <= len(raw):
            current = raw[start:end]
            if "\\" in current:
                if raw[start:end-1] != "":
                    result.append(raw[start:end-1])
                return raw[end:len(raw)]
            if current in letter_map_py:
                if raw[end_mark:start] != "":
                    result.append(raw[end_mark:start])
                result.append(letter_map_py[current][0])
                count += 1
                end_mark = end
                start = end
            end += 1
        start += 1
        end = start + 1
    if count == 0:
        if raw != "":
            result.append(raw)
    else:
        if end_mark < len(raw):
            if raw[end_mark:len(raw)] != "":
                result.append(raw[end_mark:len(raw)])
    return ""

def single_check_command_old(result, raw):
    start = 0
    end = 1
    count = 0
    while start < len(raw):
        while end <= len(raw):
            current = raw[start:end]
            if current in command_map_py:
                if raw[0:start] != "":
                    result.append(raw[0:start])
                result.append(command_map_py[current][0])
                count += 1
                end_mark = end
                start = end
            end += 1
        start += 1
        end = start + 1
    if count == 0:
        if raw != "":
            result.append(raw)
    else:
        if end_mark < len(raw):
            if raw[end_mark:len(raw)] != "":
                result.append(raw[end_mark:len(raw)])

def single_check_command(result, raw):
    start = len(raw)
    end = len(raw)-1
    count = 0
    while start > 0:
        while end >= 0:
            current = raw[end:start]
            if current in command_map_py:
                if raw[0:end] != "":
                    result.append(raw[0:end])
                result.append(command_map_py[current][0])
                count += 1
                end_mark = start
                start = end
            end -= 1
        start -= 1
        end = start - 1
    if count == 0:
        if raw != "":
            result.append(raw)
    else:
        if end_mark < len(raw):
            if raw[end_mark:len(raw)] != "":
                result.append(raw[end_mark:len(raw)])

def tokenize(raw):
    raw = raw.replace("\\ipa{", "")
    raw = raw.replace("{", "")
    raw = raw.replace("}", "")
    raw = raw.replace("(", "")
    raw = raw.replace(")", "")
    raw = raw.replace("\\super ", "\\super")

    result = []
    raw = single_check_letter(result, raw)

    raw_list_result = []
    raw_list = raw.split("\\")
    for i in range(0, len(raw_list)):
        if raw_list[i] != "":
            raw_list_result.append(raw_list[i])
    for j in range(0, len(raw_list_result)):
        raw_list_result[j] = "\\" + raw_list_result[j]
        if raw_list_result[j] in command_map_py:
            result.append(command_map_py[raw_list_result[j]][0])
        else:
            single_check_command(result, raw_list_result[j])
            single_check_letter(result, raw_list_result[j])

    return result


def tipa_to_unicode(raw):
    tokens = tokenize(raw)
    return "".join(tokens)

