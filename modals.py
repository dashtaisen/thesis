"""Tools for analyzing AMR modals"""
#from __future__ import print_function
# -*- coding: utf-8 -*-
import sys
import os
import re
import codecs
sys.path.insert(0, '../amrz/')
sys.path.insert(0, '../amrz/evaluation/smatch/')
from nltk.tree import Tree
from preprocess import read_amrz
from amr import AMR
from smatch import get_amr_line
import random

AMR_DATA_DIR = os.path.join(os.pardir,'amrz','data')
RESULTS_DIR = os.path.join(os.pardir,'results')
#path to training AMRs
GOLD_AMRS = os.path.join(AMR_DATA_DIR, 'amr_zh_10k.txt.amr')
print(GOLD_AMRS)
#path to dev AMRs
DEV_AMRS = os.path.join(AMR_DATA_DIR, 'amr_zh_all.txt.dev.amr.basic_abt_feat.1.parsed')
#path to testing AMRs

GOLD_MODALS = os.path.join(os.curdir, 'gold_modal_trees.txt')
DEV_MODALS = os.path.join(os.curdir, 'dev_modal_trees.txt')

#path to training/testing unsegmented sents
#UNSEG_SENTS = '../amrz/data/amr_zh_10k.txt'
UNSEG_SENTS = os.path.join(AMR_DATA_DIR, 'amr_zh_all.txt')

MODAL_ZH = ['能','可能','可以','得起','得了','必须','应该',
        '不得不','会','要','得出来','得起来','得出','该','需','得','可','想','想要','要求']

"""
# CAMR (Chinese Abstract Meaning Representation) release v0.1
# generated on 2017-06-16 22:58:20
# ::id export_amr.1 ::2017-02-10 13:29:53
# ::snt 最近 ， 我们 通过 知情 人士 从 衡阳市 殡葬 管理处 财务 部门 复印 出 部分 原始 发票 凭证 11 份 （ 共计 有 30余 份 ） 。
# ::wid x1_最近 x2_， x3_我们 x4_通过 x5_知情 x6_人士 x7_从 x8_衡阳市 x9_殡葬 x10_管理处 x11_财务 x12_部门 x13_复印 x14_出 x15_部分 x16_原始 x17_发票 x18_凭证 x19_11 x20_份 x21_（ x22_共计 x23_有 x24_30余 x25_份 x26_） x27_。 x28_
"""

def rephrase_complement(snt):
    """Remove complement construction"""
    pos_complement = re.compile(r'(\w+)\s[得]\s[了起到出来错完上下]')
    neg_complement = re.compile(r'(\w+)\s[不]\s[了起到出来错完上下]')

    rephrased0 = pos_complement.sub(r'能 \g<1>',snt)
    rephrased1 = neg_complement.sub(r'不 能 \g<1>',snt)

    if rephrased1 != snt:
        print("Befor: {}".format(snt))
        print("After: {}".format(rephrased1))
        print()
    return rephrased1

def rephrase_amrs(all_amr_file=None):
    """Rephrase AMRs by removing complement constructions"""
    if all_amr_file is None:
        all_amr_file = GOLD_AMRS
    rephrase_count = 0
    rephrased_amrs = list()
    comments_and_amrs = read_amrz(all_amr_file) #(comment_list, amr_list)
    comments = comments_and_amrs[0] #{'snt','id'}
    amrs = comments_and_amrs[1]
    for i in range(len(comments)):
        id = comments[i]['id']
        snt = comments[i]['snt']
        amr = amrs[i]
        rephrased_snt = rephrase_complement(snt)
        if rephrased_snt != snt:
            rephrase_count += 1
        rephrased_amrs.append((id,snt,amr))
        #possible_ids.append((comments[i]['id'].encode('utf8'),comments[i]['snt'].encode('utf8'),amrs[i].encode('utf8')))
    print("Total number of rephrased AMRs : {}".format(rephrase_count))
    return sorted(rephrased_amrs,key=lambda x: int(x[0].split(' ')[0].split('.')[1])) #sort by id number

def get_possible_amrs(all_amr_file=None):
    """Get the IDs of all AMRs with 'possible' concept
    Inputs:
        amr_file: file with all the AMRs
    Returns:
        list of (id, snt, amr) tuples
    """
    if all_amr_file is None:
        all_amr_file = GOLD_AMRS
    possible_ids = list() #(id,snt)
    comments_and_amrs = read_amrz(all_amr_file) #(comment_list, amr_list)
    comments = comments_and_amrs[0] #{'snt','id'}
    amrs = comments_and_amrs[1]
    for i in range(len(amrs)):
        amr_graph = AMR.parse_AMR_line(amrs[i])
        node_values = amr_graph.node_values
        if 'possible' in node_values:
            possible_ids.append((comments[i]['id'],comments[i]['snt'],amrs[i]))
            #possible_ids.append((comments[i]['id'].encode('utf8'),comments[i]['snt'].encode('utf8'),amrs[i].encode('utf8')))
    print("Total number of AMRs with 'possible': {}".format(len(possible_ids)))
    return sorted(possible_ids,key=lambda x: int(x[0].split(' ')[0].split('.')[1])) #sort by id number

#def write_possible_amrs(destfile,all_amr_file=None):
def write_possible_amrs(amr_list,destfile):
    """Write all trees with 'possible' as a concept"""
    #if all_amr_file is None:
        #all_amr_file = GOLD_AMRS
    #possible_ids = get_possible_amrs(all_amr_file)
    with open(destfile,'w') as dest:
        for ids,snts,trees in amr_list:
            dest.write('# ::id {}\n'.format(ids))
            dest.write('# ::snt {}\n'.format(snts))
            #tree_string = trees.decode('utf8')
            tree_string = trees
            try:
                tree_string = Tree.fromstring(tree_string).pformat()
            except ValueError:
                tree_string += ")"
                tree_string = Tree.fromstring(tree_string).pformat()
            #dest.write('{}\n\n'.format(tree_string.encode('utf8')))
            dest.write('{}\n\n'.format(tree_string))

def get_possible_devs(all_amr_file=None, dev_amr_file=None):
    """Get sents in the dev set that *should* have 'possible' concept
    Inputs:
        all_amr_file: filename with all amrs
        dev_amr_file: filename with dev amrs
    Returns:
        possible_devs: list of (id, snt, amr) tuples
    """
    if all_amr_file is None:
        all_amr_file = GOLD_AMRS
    if dev_amr_file is None:
        dev_amr_file = DEV_AMRS
    possible_ids = [item[0] for item in get_possible_amrs(GOLD_AMRS)]
    possible_devs = list()
    dev_tuples = read_amrz(dev_amr_file)
    dev_comments = dev_tuples[0]
    dev_amrs = dev_tuples[1]
    for i in range(len(dev_comments)):
        if dev_comments[i]['id'] in possible_ids:
            #possible_devs.append({'id':dev_comments[i]['id'],'snt':dev_comments[i]['snt'],'amr':dev_amrs[i]})
            possible_devs.append((dev_comments[i]['id'],dev_comments[i]['snt'],dev_amrs[i]))
    #possible_devs = sorted(possible_devs, key=lambda x:int(x['id'].split(' ')[0].split('.')[1]))
    possible_devs = sorted(possible_devs, key=lambda x: int(x[0].split(' ')[0].split('.')[1]))
    print("Total number of sentences in dev/test sets with 'possible': {}".format(len(possible_devs)))
    return possible_devs

def write_possible_amrs_from_dict(possible_amrs, destfile):
    """Write AMRs with 'possible' to file
    Inputs:
        possible_amrs: list of (id, snt, amr) tuples
        destfile: file to write to
    Returns:
        None (writes to file)
    """
    with codecs.open(destfile,'w') as dest:
        for possible_amr in possible_amrs:
            dest.write('# ::id {}\n'.format(possible_amr['id']))
            dest.write('# ::snt {}\n'.format(possible_amr['snt'].encode('utf-8')))
            tree_string = possible_amr['amr']
            tree_string = Tree.fromstring(tree_string).pformat()
            dest.write('{}\n\n'.format(tree_string.encode('utf8')))


def find_mismatch():
    """Find cases where dev amr should have 'possibilty' but doesn't"""
    possibility_missing = list()
    possibility_spurious = list()

    gold_tuples = get_possible_amrs()
    dev_tuples = get_possible_devs()

    for dev_tuple in dev_tuples:
        gold_matches = [gold_tuple for gold_tuple in gold_tuples if gold_tuple[0] == dev_tuple[0]]
        if len(gold_matches) > 0:
            gold_match = gold_matches[0]
            dev_amr = AMR.parse_AMR_line(dev_tuple[2])
            dev_nodes = dev_amr.node_values
            gold_amr = AMR.parse_AMR_line(gold_match[2])
            gold_nodes = gold_amr.node_values
            if 'possible' in gold_nodes and 'possible' not in dev_nodes:
                possibility_missing.append(dev_tuple[0])
            elif 'possible' in dev_nodes and 'possible' not in gold_nodes:
                possibility_spurious.append(dev_tuple[0])
    return possibility_missing, possibility_spurious

def sample_tree():
    """Draw a sample tree"""
    gold_tuples = read_amrz(GOLD_MODALS)
    sample_amr = random.choice(gold_tuples[1]).replace('/','')
    t = Tree.fromstring(sample_amr)
    t.draw()

def get_modal_sents():
    modal_sents = dict()
    with open(UNSEG_SENTS) as source:
        current_id = ""
        current_sent = ""
        for line in source:
            if line.startswith('# ::id'):
                print("On {}".format(line))
                current_id = line
            elif line.startswith("# ::snt"):
                current_sent = line
                for modal in MODAL_ZH:
                    if modal in current_sent:
                        try:
                            modal_sents[modal].add((current_id, current_sent))
                        except KeyError:
                            modal_sents[modal] = set((current_id, current_sent))
    return modal_sents

if __name__ == "__main__":
    rephrased = rephrase_amrs(all_amr_file=GOLD_AMRS)
    write_possible_amrs(amr_list=rephrased,destfile='../results/amr_zh_10k_rephrased.txt')
    #possible_amrs = get_possible_amrs(all_amr_file=GOLD_AMRS)
    #write_possible_amrs(destfile='../results/possible_amrs.txt',all_amr_file=GOLD_AMRS)
    #possible_devs = get_possible_devs(all_amr_file=GOLD_AMRS, dev_amr_file=DEV_AMRS)
    #write_possible_amrs(amr_list=possible_devs,destfile='../results/possible_devs.txt')
    #missing,spurious = find_mismatch()
    #print("Missing 'possibility' concepts: {}".format(len(missing)))
    #print("Spurious 'possibility' concepts: {}".format(len(spurious)))
    #print(len(mismatch))
    #sample_tree()
    #write_possible_ids()
    #get_possible_devs()
    """
    possible_ids = get_possible_ids()
    with open('modal_result.txt','w') as dest:
        for ids,snts in possible_ids:
            dest.write('id: {} snt: {}\n'.format(ids,snts))
    """

    #comments_and_amrs = read_amrz(UNSEG_SENTS) #(comment_list, amr_list)
    """
    nodes: names of nodes in AMR graph, e.g. "a11", "n"
    node_values: values of nodes AMR graph, e.g. "group" for a node named "g"
    relations: list of relations between two nodes
    attributes: list of attributes (links between one node and one constant value)

    """

    """
    #sample_amr = comments_and_amrs[1][2]
    with open(GOLD_AMRs) as source:
        possibilities = list()
        for line in source:
            amr_line = get_amr_line(source)
            sample_amr = AMR.parse_AMR_line(amr_line)
            #print(sample_amr.nodes)
            if 'possible' in sample_amr.node_values:
                possibilities.append(sample_amr)
        print(len(possibilities))
            #print(sample_amr.relations)
            #print(sample_amr.attributes)

    #a = AMR()
    #graph = a.parse_AMR_line(sample_amr)
    #print(graph)
    #modal_sents = get_modal_sents()
    #for key in modal_sents.keys():
        #print("{}:{}".format(key, len(modal_sents[key])))
    #print(modal_sents[' 得了 '])
    """
