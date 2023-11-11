# extract the sound change rules

def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def extract(line):
    result = ""
    left = 0
    right = 0
    for letter in line:
        if letter == '{':
            left += 1
        elif letter == '}':
            right += 1
        else:
            result += letter
        if left != 0 and left == right:
            return result
    
def extractRules(filename):
    document = readFile(filename)
    result = ""
    for line in document.splitlines():
        if "\subsection" in line:
            line = line[11:]
            familyName = extract(line)
            result += "1. family name: " + familyName + "\n"
        if "\subsubsection" in line:
            line = line[14:]
            familyName = extract(line)
            result += "2. family name: " + familyName + "\n"
        if "\paragraph" in line:
            line = line[10:]
            familyName = extract(line)
            result += "3. family name: " + familyName + "\n"
        if "\subparagraph" in line:
            line = line[13:]
            familyName = extract(line)
            result += "4. family name: " + familyName + "\n"
        if " \change\ " in line:
            result += line + "\n"
    writeFile("data/rules.txt", result)
    return result

print(extractRules("index_diachronica.tex"))
