# author   : Johann-Mattis List
# email    : mattis.list@lingpy.org
# created  : 2016-01-04 13:45
# modified : 2016-01-04 13:45

"""
Analyze the data using the parsimony models
"""

__author__="Johann-Mattis List"
__date__="2015-08-05"

import json
from lingpy import *

from sys import argv
from L_newick import *
from L_parsimony import *
import networkx as nx
from sys import argv
from html import parser
import os

data = json.loads(open('I_data.json').read())

# try to get a tree file
tree = False
for arg in argv:
    if arg.startswith('tree'):
        tree = LingPyTree(open(arg.split('=')[-1]).read())

# search for matrix in argv
matrix = False
for arg in argv:
    if arg.startswith('matrix'):
        matrix = arg.split('=')[1]

# search for runs
runs = 15000
for arg in argv:
    if arg.startswith('runs'):
        runs = int(arg.split('=')[1])

if 'check' in argv:

    for i,(chars,matrix) in enumerate(zip(data['chars'], data[matrix])):
        if len(matrix) != len(chars):
            print(i+1, len(matrix), len(chars), chars,data['patterns'][i])

if 'hp' in argv:
    trees = heuristic_parsimony(
            data['taxa'],
            data['patterns'],
            data[matrix],
            data['chars'] if matrix!='fitch' else data['fitch.chars'],
            guide_tree = tree.newick if tree else '',
            verbose=True,
            iterations = runs,
            sample_steps = 250,
            log = True,
            stop_iteration=True
            )

    with open('R_'+matrix+'-'+str(trees[1])+'.trees','w') as f:
        for t in trees[0]:
            f.write(t+';\n')

    # change the log files
    os.system('mv log.log R_'+matrix+'.trees.log')
    
if 'plot' in argv:

    
    if os.path.isdir('html'):
        pass
    else:
        os.mkdir('html')


    alias = {
            'diwest' : 'WDT',
            'fitch' : 'FITCH',
            'sankoff' : 'SANKOFF',
            'chacon2014' : 'Chacon (2014)',
            'consensus' : 'New Consensus'
            }
    
    argd = dict([arg.split('=') for arg in argv if '=' in arg])
    
    if 'alias' in argd:
        model_name = argd['alias']
    elif matrix in alias:
        model_name = alias[matrix]
    else:
        model_name = matrix

    if not tree:
        raise ValueError("No tree specified!")

    txt = '<h3> Parsimony Analysis for the {0} Model</h3>'.format(model_name) 
    txt += """<table class="table"><tr>
    <th>Number</th>
    <th>Proto-Form</th>
    <th>Context</th>
    <th>Weight</th>
    <th>Inferred Proto-Forms</th>
    <th>Scenario</th></tr>
    """

    template = """
    <tr>
    <td>{0}</td>
    <td>*{1}</td>
    <td>{2}</td>
    <td>{3}</td>
    <td>{4}</td>
    <td><a class="button" target="other" href="pattern-{5}-{0}.html">SHOW</a></td>
    </tr>"""
    
    total_weight = 0
    good = 0
    
    # assign all labels for the data
    L = {}
    for node in tree.postorder:
        L[node] = {}

    for idx,(p,m,c,pr) in enumerate(zip(data['patterns'], data[matrix], 
        data['chars'] if matrix != 'fitch' else data['fitch.chars'],
            data['protos'])):
        
        w,p,r = sankoff_parsimony(
                p,
                data['taxa'],
                tree,
                m,
                c
                )
        print(pr[0], pr[1], w)
        labels = {}
        _data = {}
        for t,c in p[0]:
            labels[tree[t]['label']] = '<b>['+c+']</b> <sup>'+tree[t]['label']+'</sup>'
            _data[tree[t]['label']] = c
    
        tree.output('html', filename='html/pattern-'+model_name+'-'+str(pr[0]), labels=labels,
                data=_data)

        # add node states for each label according to the first analysis
        for t,c in p[0]:
            L[t][pr[0],pr[1],pr[2]] = c
        
        # count the number of proposed reconstructions
        reconstructions = []
        for reconstruction in p:
            _tmp = dict(reconstruction)
            proto_form = _tmp[tree.root]
            reconstructions += [proto_form]
        # get the set
        pformset = sorted(set(reconstructions), key=lambda x:
                reconstructions.count(x), reverse=True)
        pforms = ', '.join([x+' ({0:.2f})'.format(
            reconstructions.count(x) / len(reconstructions)
            ) for x in pformset])
        if pr[1] in pformset:
            if len(pformset) == 1:
                pass
            else:
                pforms = '<span style="color:DarkBlue">'+pforms+'</span>'
        else:
            pforms = '<span style="color:red">'+pforms+'</span>'

        good += reconstructions.count(pr[1]) / len(reconstructions)

        txt += template.format(
                pr[0],
                pr[1],
                pr[2] if pr[2] else '',
                w[0],
                pforms,
                model_name
                )
        total_weight += w[0]
    
    # total reconstruction score
    total_good = good / len(data['patterns'])
    
    # create the style for the output-file
    style = """<style>
    .button {text-decoration: none; background: green; color: white; border: 1px solid gray;}
    .table {border: 2px solid black;} .table th {border: 2px solid gray;}
    .table td {border: 1px solid lightgray;}
    </style>"""

    # calculate innovations fore each node
    new_labels = {}
    new_data = {}
    new_labels['root'] = 'Proto-Tukano'
    new_data['root'] = ''

    # table template
    ttemp = r"""<table><tr><th>Number</th><th>Proto-Form</th><th>Context</th><th>Parent Form</th><th>New Form</th></tr>"""

    for node in tree.preorder[1:]:
        
        # get the parent
        parent = tree[node]['parent']
        
        label = []
        # iterate over proto-forms
        for a,b,c in data['protos']:
            pform = L[parent][a,b,c]
            cform = L[node][a,b,c]

            if pform != cform:
                label += ['<tr><td>{0}</td><td>*{1}</td><td>{2}</td><td>{3}</td><td>{4}</td></td></tr>'.format(
                    a, b, a, pform, cform)]

        new_labels[tree[node]['label']] = """<span style="cursor:pointer" onclick="document.getElementById('popup').innerHTML = '{0}'; document.getElementById('popup').style.display='block';">{1}</span>""".format(
                '<b>Innovations at node '+tree[node]['label']+':</b>'+ttemp+''.join(label)+'</table>',
                tree[node]['label'] + ' ('+str(len(label))+')'
                )
        new_data[tree[node]['label']] = str(len(label))
        
    tree.output('html', filename='html/pattern-'+model_name+'-summary',
            labels=new_labels, data=new_data)
    
    
    # complete table
    txt += """<tr><td colspan="3">Total Weight</td><td>"""
    txt += str(total_weight)+'</td></td>'
    txt += '<td>'+'{0:.2f}'.format(total_good)+'</td>'
    txt += '<td><a target="other" href="pattern-'+model_name+'-summary.html" class="button">SHOW</a></td></tr></table>'


        

        
    with open('html/navi-'+model_name+'.html', 'w') as f:
        f.write('<html><head><meta charset="utf-8"</meta></head><body>'+style+txt+'</body></html>')

if 'homoplasy' in argv:
    if not tree:
        raise ValueError("No tree specified!")
    homoplasy = 0
    H = {}
    for idx,(p,m,c,pr) in enumerate(zip(data['patterns'], data[matrix], 
        data['chars'] if matrix != 'fitch' else data['fitch.chars'],    
        data['protos'])):
        
        C = {}
        w,p,r = sankoff_parsimony(
                p,
                data['taxa'],
                tree,
                m,
                c
                )
        
        # convert patterns to dictionary
        P = dict(p[0])

        # count the homoplasy
        char = P[tree.root]
        
        for node in tree.preorder[1:]:

            pChar = P[tree[node]['parent']]
            tChar = P[node]

            if pChar != tChar:
                try:
                    C[pChar,tChar] += 1
                except KeyError:
                    C[pChar,tChar] = 0

                try:
                    H[pChar,tChar] += 1
                except KeyError:
                    H[pChar,tChar] = 1

        homoplasy += sum(C.values())
        print(idx+1, pr[1], homoplasy, sum(C.values()))
    print('TOTAL', '{0:.2}'.format(homoplasy / len(data['protos'])))
    
    with open('R_sound-change-frequencies-'+matrix+'.tsv', 'w') as f:
        f.write('SOURCE\tTARGET\tFREQUENCY\n')
        for (s,t),v in sorted(H.items(), key=lambda x: x[1], reverse=True):
            f.write('{0}\t{1}\t{2}\n'.format(s,t,v))
    
    G = nx.DiGraph()
    for a,b in H:
        G.add_edge(a,b,weight=H[a,b])
    nx.write_gml(G,'.tmp.gml')
    tmp = open('.tmp.gml').read()
    with open('R_scf-'+matrix+'.gml', 'w') as f:
        f.write(parser.unescape(tmp))

if 'proto' in argv:
    if not tree:
        raise ValueError("No tree specified!")
    
    C = {}
    for idx,(p,m,c,pr) in enumerate(zip(data['patterns'], data[matrix], 
        data['chars'] if matrix != 'fitch' else data['fitch.chars'],
            data['protos'])):
        

        w,p,r = sankoff_parsimony(
                p,
                data['taxa'],
                tree,
                m,
                c
                )
        for pt in p:
            char = dict(pt)[tree.root]
            
            try:
                C[idx+1,pr[1]] += [char]
            except KeyError:
                C[idx+1,pr[1]] = [char]
        print(idx+1, w)
    
    good = 0
    bad = 0
    close = 0
    for k,v in sorted(C.items(), key=lambda x:x[0][0]):

        if len(set(v)) == 1 and k[1] == v[0]:
            print('[{0}] correctly identified *{1}.'.format(k[0], v[0]))
            good += 1
        elif k[1] in v:
            print('[{0}] proposed the right character *{1} out of {2} ({3}).'.format( 
                k[0], k[1], len(v), '/'.join(v)))
            close += 1
        else:
            print('[{0}] wrong reconstruction for *{1} ({2}).'.format(
                k[0], k[1], '/'.join(v)))
            bad += 1
    
    total = good + bad + close
    print('GOOD: {0} ({1}%)\nBAD: {2} ({3}%)\nCLOSE: {4} ({5}%)'.format(

        good,
        int(100 * good / total + 0.5),
        bad,
        int(100 * bad / total + 0.5),
        close,
        int(100 * close / total + 0.5)
        ))

        

