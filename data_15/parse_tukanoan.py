import pandas as pd
languages = pd.read_csv("input/languages.csv")
cognates = pd.read_csv("input/cognates.csv")
forms = pd.read_csv("input/forms.csv")


def process_prefix(word):
  # \u0330 is the creaky voice
  # \u02c0 is the glottal stop diacritic
  result = []
  for token in word.split():
    if token[0] == "ˀ":
      print(token)
      token = token.replace("ˀ", "")
      token = token + "\u0330"
      result.append(token)
    else:
      result.append(token)
      
  return " ".join(result)

all_languages = languages["ID"]
all_segments = forms["Segments"]
language_list = [all_languages[i] for i in range(len(all_languages))]

# generate cognate information
# in form of all_cognates[cognate name][language name] = (daughter form, expert alignment)
all_cognates = dict()
for index in cognates.index:
  cognacy = cognates["Cognateset_ID"][index]
  daughter_ID = cognates["Form_ID"][index]
  daughter_form = all_segments.get(index)
  language_name = daughter_ID.split("-")[0]
  alignment = cognates["Alignment"][index]
  # print(cognacy, daughter_ID, daughter_form)
  if cognacy not in all_cognates:
    all_cognates[cognacy] = {language_name: (daughter_form, alignment)}
  else:
    all_cognates[cognacy][language_name] = (daughter_form, alignment)

# print(all_cognates)

with open("temp.csv", "w", encoding='utf-8') as f:
  f.write("#id\t")
  f.write("\t".join(language_list))
  f.write("\n")
  for cog in all_cognates:
    f.write(cog)
    for lan in language_list:
      if lan in all_cognates[cog]:
        f.write("\t")
        word = all_cognates[cog][lan][0].replace('\u0067','\u0261')
        word = word.replace("+", "")
        # TODO: pre-laryngealization to a phone with a ~ at the bottom        word = word.replace('\u0067','\u0261')
        # \u0330 is the creaky voice
        # \u02c0 is the glottal stop diacritic
        # word = process_prefix(word)
        f.write(word)
      else:
        f.write("\t?")
    f.write("\n")
        
f.close()


with open("expert_alignment.csv", "w", encoding='utf-8') as f:
  for cog in all_cognates:
    if "Prototucanoan" not in all_cognates[cog]:
      # there are 8 cognates without a protoform!!!!
      print("????")
    else:
      proto_alignment = all_cognates[cog]["Prototucanoan"][1]
      for lan in language_list:
        if lan == "Prototucanoan":
          continue
        if lan in all_cognates[cog]:
          word = all_cognates[cog][lan][1].replace('\u0067','\u0261')
          word = word.replace("+", "")
          f.write(word)
          f.write(" > ")
          f.write(proto_alignment)
          f.write(" / ")
          f.write(lan)
          f.write("\n")
          
f.close()
