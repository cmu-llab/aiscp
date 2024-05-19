# author   : Johann-Mattis List
# email    : mattis.list@lingpy.org
# created  : 2015-07-19 12:00
# modified : 2015-07-19 12:00
"""
Newick module for parsing of Newick strings.

This is an experimental module for newick string parsing. Once it has been
expanded and sufficiently tested, it is planned to replace the current Newick
handling in LingPy.
"""

__author__="Johann-Mattis List"
__date__="2015-07-19"

import json
import pdb

def sort_tree(tree):
    """
    Simple method to sort a given tree.
    """
    
    def get_leaves(tree):
        return [''.join([y for y in x if y not in '();,']) for x in
                tree.split(',')]
    nodes = parse_newick(tree)

    out = '{root}'

    queue = [nodes['root']]
    while queue:
        node = queue.pop(0)
        label = nodes[node]['label']
        
        # get the children
        children = nodes[node]['children']
        if children:

            # sort the children
            kids = [''.join(sorted(get_leaves(child))) for child in children] 

            # sort children
            offspring = '('+','.join(['{'+nodes[a]['label']+'}' for a,b in sorted(zip(children, kids),
                key=lambda x: x[1])])+')'
            out = out.replace('{'+label+'}', offspring)
            queue += children

    return ''.join([x for x in out if x not in '{}'])


def nodes_in_tree(tree):
    """
    This methods yields all nodes in a tree in newick format without labels for
    nodes and branch lengths being assigned.
    """
    
    stack = [tree[1:-1]]
    out = [tree]

    yield tree
    
    nit_iter = 0
    while stack:
        tmp = stack.pop()
        nit_iter += 1
        if nit_iter > 10000:
            raise RuntimeError("Likely infinite loop in nodes_in_tree")
        brackets = 0

        idx = 0
        for i,c in enumerate(tmp):
            if c == '(':
                brackets += 1
            elif c == ')':
                brackets -= 1

            if not brackets and c == ',':
                tree = tmp[idx:i]
                idx = i+1

                yield tree

                if tree.startswith('('):
                    tree = tree[1:-1]

                if ',' in tree:
                    stack += [tree]
        
        tree = tmp[idx:]
        yield tree
        if tree.startswith('('):
            tree = tree[1:-1]
        


        if ',' in tree:
            stack += [tree]

    return out

def safe_newick_string(newick):
    
    if newick.endswith(';'):
        newick = newick[:-1]
        
    newick = clean_newick_string(newick)

    leaves = sorted(
            [n for n in nodes_in_tree(newick) if ',' not in n],
            key = lambda x: len(x),
            reverse = True
            )
    
    for i,leave in enumerate(leaves):
        newick = newick.replace(leave, '<{0}>'.format(i))

    for i,leave in enumerate(leaves):
        newick = newick.replace('<{0}>'.format(i), '"'+leave+'"')

    return newick


def clean_newick_string(newick):
    """
    Helper function to reduce all branch-lengths from a Newick string.

    Note
    ----
    This function "cleans" a Newick string by reducing all of its branch
    lengths. As a result, the pure topological Newick string is returned.
    """
    start = newick
    out = ''
    
    while start:
        idxA = start.find(':') 
        idxB = start.find(')')

        colon = False

        if idxA != -1 and idxB != -1:
            if idxA < idxB:
                idx = idxA
                colon = True
            else:
                idx = idxB
        elif idxA != -1:
            idx = idxA
            colon = True
        elif idxB != -1:
            idx = idxB
        else:
            out += start
            return out.replace('"','')
        
        if colon:
            out += start[:idx]
            start = start[idx+1:]
            while start and (start[0].isdigit() or start[0] == '.'):
                start = start[1:]
        else:
            out += start[:idx+1]
            start = start[idx+1:]
            while start and start[0] not in ',;)':
                start = start[1:]
    
    out += start

    return out.replace('"','')

def parse_newick(newick):
    """
    Function parses a Newick tree to JSON format.

    Notes
    -----
    The format is a dictionary with sufficient information to further parse the
    tree, and also to use it as input for d3 and other JavaScript libraries.

    Examples
    --------
      
      >>> from lingpy.basic import newick as nwk
      >>> a = '((a,b),c);'
      >>> nwk.parse_newick(a)
      {'((a,b),c)': {'branch_length': '0', 'children': ['(a,b)', 'c'], 'label': 'root', 'leave': False, 'root': True}, '(a,b)': {'branch_length': '0', 'children': ['a', 'b'], 'label': 'edge_1', 'leave': False, 'parent': '((a,b),c)', 'root': False}, 'a': {'branch_length': '0', 'children': [], 'label': 'a', 'leave': True, 'parent': '(a,b)', 'root': False}, 'b': {'branch_length': '0', 'children': [], 'label': 'b', 'leave': True, 'parent': '(a,b)', 'root': False}, 'c': {'branch_length': '0', 'children': [], 'label': 'c', 'leave': True, 'parent': '((a,b),c)', 'root': False}, 'leaves': ['c', 'a', 'b'], 'nodes': ['((a,b),c)', '(a,b)', 'c', 'a', 'b'], 'root': '((a,b),c)'}
      
    """
    # create the dictionary to host the data
    D = {}
    
    # check for correct newick ending and remove semi-colon in the end
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    # get label and branch length
    nwk, label, blen = label_and_blen(newick)

    root = '('+clean_newick_string(nwk)+')'
    
    D['root'] = root
    D['nodes'] = [root]
    D['leaves'] = []

    D[root] = dict(
            children=[], 
            branch_length = blen, 
            root=True,
            leave=False,
            label = label or 'root'
            )

    label_count = 1

    pn_iter = 0
    for node, label, blen, parent in all_nodes_of_newick_tree(newick):
        
        pn_iter += 1
        if pn_iter > 10000:
            raise RuntimeError("Likely infinite loop in parse_newick")
        cnode = clean_newick_string(node)
        cparent = clean_newick_string(parent)
        
        D['nodes'] += [cnode]

        D[cnode] = dict(parent=cparent, children=[], branch_length=blen,
                root=False, label=label or 'edge_'+str(label_count))
        
        if not label:
            label_count += 1

        if ',' in node:
            D[cnode]['leave'] = False
        else:
            D[cnode]['leave'] = True
            D['leaves'] += [cnode]
        
        D[cparent]['children'] += [cnode]

    return D

def label_and_blen(nwk):
    """
    Helper function parses a Newick string and returns the highest-order label and branch-length.
    
    Returns
    -------
    data : tuple
        A tuple consisting of the node, deprived of brackets and label, the
        label of the node, and the branch length. If either of branch length or
        label is missing, the tuple contains an empty string.

    Examples
    --------

      >>> from lingpy.basic import newick as nwk
      >>> nwk.label_and_blen('(a:1,b:2)ab:20')
      ('a:1,b:2', 'ab', '20')      
      
    """

    # get the first index of a bracket
    idx = nwk[::-1].find(')')

    # no brackets means we are dealing with a leave node
    if idx == -1:
        nwk_base = nwk
        idx = nwk[::-1].find(':')
        if idx == -1:
            return nwk, nwk, '0'
        else:
            nwk_base = nwk[:-idx-1]
            return nwk_base, nwk_base, nwk[-idx:]
    
    # if index is 0, there's no label and no blen
    elif idx == 0:
        return nwk[1:-1], '', '0'

    # else, we carry on
    nwk_base = nwk[:-idx]
    label = nwk[-idx:]

    idx = label[::-1].find(':')
    if idx == -1:
        label = label
        blen = 0
    else:
        blen = label[-idx:]
        label = label[:-idx-1]
    
    return nwk_base[1:-1], label, blen

def all_nodes_of_newick_tree(newick):
    """
    Function returns all nodes of a tree passed as Newick string.
    
    Notes
    -----
    This function employs a simple search algorithm and splits a tree in binary
    manner in pieces right until all nodes are extracted.

    Examples
    --------
    >>> from lingpy.basic import newick as nwk
    >>> list(nwk.all_nodes_of_newick_tree('((a:1,b:1)ab:2,c:2)abc:3;'))
    [('(a,b)', 'ab', '2', '((a,b),c)'), ('c', 'c', '2', '((a,b),c)'), ('a', 'a', '1', '(a,b)'), ('b', 'b', '1', '(a,b)')]
    
    Raises
    ------
    ValueError, if the number of brackets is wrong.

    Returns
    -------
    nodes : generator
        A generator that yields tuples for all nodes of a Newick string. Each
        tuple contains four entries: The current node in Newick format, the
        label of the current node, the branch length, and the parent node (root
        node is not returned). 

    """
    # look for bracket already, don't assume they are around the tree! 
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]

    # check for whitespace in tree and raise error if this is the case
    if ' ' in newick:
        raise ValueError('[!] The string contains whitespace which is not allowed!')

    # fill the queue
    nwk, label, blen = label_and_blen(newick)
    queue = [(nwk, label, blen)]
    
    # raise error if the number of brackets doesn't fit
    nr_opening_brackets = newick.count('(')
    nr_closing_brackets = newick.count(')')
    if nr_opening_brackets != nr_closing_brackets:
        raise ValueError("The number of brackets is wrong!")
    
    while queue:
                
        # find un-bracketed part inbetween
        nwk, label, blen = queue.pop(0)

        brackets = 0
        idxs = [-1]
        for i,k in enumerate(nwk):

            if k == '(':
                brackets += 1
            elif k == ')':
                brackets -= 1
            
            if not brackets and k == ',':
                idxs += [i]

        idxs += [i+1]

        for i,idx in enumerate(idxs[1:]):

            nwk_tmp = nwk[idxs[i]+1:idx]
            nwk_tmp, label, blen = label_and_blen(nwk_tmp)

            nnwk = clean_newick_string(nwk_tmp)
            npar = clean_newick_string(nwk)

            nnwk = '('+nnwk+')' if ',' in nnwk else nnwk
            npar = '('+npar+')' if ',' in npar else npar
            
            yield nnwk, label, blen, npar

            if ',' in nwk_tmp:
                
                queue += [(nwk_tmp, label, blen)]


def postorder(tree):
    """
    Carry out a post-order traversal of a LingPyTree object.

    Notes
    -----
    This function carries out a post-order traversal of a LingPyTree object and
    returns all nodes of the tree in post-order.

    """
    
    # make the stack
    stack = [tree['root']]

    # make the output
    out = []

    # make copy of tree
    ctree = dict([(k,[x for x in tree[k]['children']]) for k in tree['nodes']])

    # climb down the tree

    while stack:
        
        node = stack[-1]
        children = ctree[node]
        
        # if we are at a leave-node, we remove the item from the stack 
        if not children:
            stack.pop()
            out += [node]
            if stack:
                ctree[stack[-1]].pop(0)

        else:
            stack += [children[0]]
    
    return out

class LingPyTree(object):

    def __init__(self, newick):
        """
        Tree Class for a simple handling of various tree operations, based on Newick as an input format.

        """
        
        self.newick = newick
        self._dict = parse_newick(newick)
        self.root = self._dict['root']
        self.nodes = self._dict['nodes']
        self.leaves = self._dict['leaves']
        self.preorder = self.nodes
        self.postorder = postorder(self._dict)

    def output(self, dtype, filename=None, labels=None, data=None):
        """
        Parameters
        ----------
        dtype : str {"json", "html", "nwk" }
            Specify the type of the output:
            
            * *json*: JSON format, suitable for use in d3.
            * *nwk*: Newick format (identical with input upon initialization).
            * *html*: Simple interactive HTML-representation with collapsible nodes.

        """
        
        if dtype == 'json':
            if filename:
                with open(filename+'.'+dtype, 'w') as f:
                    f.write(json.dumps(self._dict, indent=2))
            else:
                return json.dumps(self._dict, indent=2)
        
        elif dtype == 'html':

            # make simple label function
            get_label = lambda x: labels[x] if labels else x
            get_data = lambda x: data[x] if data else x
           
            start = '<div id="root" class="node-container">root.content</div>'
            
            clean_label = lambda x: ''.join([y for y in sort_tree(x) if y not in '();']).replace(',','_')

            template = '<div class="node-container"><div id="#node_name:label" data-value="#data_value" class="node-label">#node_label</div><div class="node-content">#node_children:{node}</div></div>'

            leave = '<div id="#node_leave_name:label" data-value="#data_value" class="node-leave"><div class="inner_leave">#node_leave</div></div>'

            txt = template.format(node=self.root).replace('#node_label',
                    get_label(self[self.root]['label'])).replace('#node_name',
                            self[self.root]['label']).replace(
                                    '#data_value',
                                    get_data(self[self.root]['label']))
            
            # transform function helps to make the transformation with check
            # for leave or child
            transform = lambda x: template.format(node=x).replace('#node_label', \
                        get_label(self[x]['label'])).replace('#node_name',
                                self[x]['label']).replace(
                                        '#data_value',
                                        get_data(self[x]['label'])) if \
                        not self[x]['leave'] else leave.replace(
                                '#node_leave_name',
                                self[x]['label']).replace('#node_leave',
                                        get_label(self[x]['label'])).replace(
                                                '#data_value',
                                                get_data(self[x]['label']))


            for i,node in enumerate(self.nodes):

                # write all children
                children = self[node]['children']
                
                node_children = '\n'.join([transform(child) for child in
                        children])
                                        
                txt = txt.replace(
                        '#node_children:'+node, 
                        node_children
                        )

            # get the templates
            html = open('T_lexical_change.html').read()
            #html = util.read_text_file('templates/lexical_change.html')
            css = open('T_lexical_change.css').read()
            #css = util.read_text_file('templates/lexical_change.css')
            
            # add the nodes to js
            js = '\nvar TREE = '+json.dumps(self._dict, indent=2)+';\n\n'
            js += open('T_lexical_change.js').read()
            #js += util.read_text_file('templates/lexical_change.js')

            title = 'LingPy Tree Class'

            html = html.format(STYLE=css, SCRIPT=js, TITLE=title, TREE=txt)
            filename = filename or 'lingpy.basic.newick'
            
            f = open(filename+'.html', 'w')
            f.write(html)
            f.close()
            #util.write_text_file(
            #        filename+'.html',
            #        html
            #        )

    def __getitem__(self, idx):
        """
        Allow for the tree class to be used as a dictionary.
        """

        try:
            return self._dict[idx]
        except:
            raise KeyError(idx)


