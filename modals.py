#from __future__ import print_function
# -*- coding: utf-8 -*-
import sys
import codecs
sys.path.insert(0, '../amrz/')
sys.path.insert(0, '../amrz/evaluation/smatch/')
from nltk.tree import Tree
from preprocess import read_amrz
from amr import AMR
from smatch import get_amr_line
import random

GOLD_MODALS = './gold_modal_trees.txt'
DEV_MODALS = './dev_modal_trees.txt'

#path to training/testing unsegmented sents
#UNSEG_SENTS = '../amrz/data/amr_zh_10k.txt'
UNSEG_SENTS = '../amrz/data/amr_zh_all.txt'

#path to training/testing segmented sents

#path to training AMRs
GOLD_AMRS = '../amrz/data/amr_zh_10k.txt.amr'

#path to dev AMRs
DEV_AMRS = '../amrz/data/amr_zh_all.txt.dev.amr.basic_abt_feat.1.parsed'
#path to testing AMRs

MODAL_ZH = ['能','可能','可以','得起','得了','必须','应该',
        '不得不','会','要','得出来','得起来','得出','该','需','得','可','想','想要','要求']

"""
# CAMR (Chinese Abstract Meaning Representation) release v0.1
# generated on 2017-06-16 22:58:20
# ::id export_amr.1 ::2017-02-10 13:29:53
# ::snt 最近 ， 我们 通过 知情 人士 从 衡阳市 殡葬 管理处 财务 部门 复印 出 部分 原始 发票 凭证 11 份 （ 共计 有 30余 份 ） 。
# ::wid x1_最近 x2_， x3_我们 x4_通过 x5_知情 x6_人士 x7_从 x8_衡阳市 x9_殡葬 x10_管理处 x11_财务 x12_部门 x13_复印 x14_出 x15_部分 x16_原始 x17_发票 x18_凭证 x19_11 x20_份 x21_（ x22_共计 x23_有 x24_30余 x25_份 x26_） x27_。 x28_
"""

def sample_tree():
    gold_tuples = read_amrz(GOLD_MODALS)
    sample_amr = random.choice(gold_tuples[1]).replace('/','')
    t = Tree.fromstring(sample_amr)
    t.draw()

def find_mismatch():
    mismatches = list()
    gold_tuples = read_amrz(GOLD_MODALS)
    gold_comments = gold_tuples[0]
    gold_ids = [gold_comment['id'] for gold_comment in gold_comments]
    gold_snts = [gold_comment['snt'] for gold_comment in gold_comments]
    gold_amrs = gold_tuples[1]
    gold_tuples = zip(gold_ids,gold_snts,gold_amrs)

    dev_tuples = read_amrz(DEV_MODALS)
    dev_comments = dev_tuples[0]
    dev_ids = [dev_comment['id'] for dev_comment in dev_comments]
    dev_snts = [dev_comment['snt'] for dev_comment in dev_comments]
    dev_amrs = dev_tuples[1] 
    dev_tuples = zip(dev_ids,dev_snts,dev_amrs)
    print(len(dev_tuples))
    print(dev_tuples[0][0])
    print(gold_tuples[0][0])
    for dev_tuple in dev_tuples:
        gold_matches = [gold_tuple for gold_tuple in gold_tuples if gold_tuple[0] == dev_tuple[0]]
        if len(gold_matches) > 0:
            gold_match = gold_matches[0]
            dev_amr = AMR.parse_AMR_line(dev_tuple[2])
            dev_nodes = dev_amr.node_values
            gold_amr = AMR.parse_AMR_line(gold_match[2])
            gold_nodes = gold_amr.node_values
            if 'possible' in gold_nodes and 'possible' not in dev_nodes:
                mismatches.append(dev_tuple[0])
            elif 'possible' in dev_nodes and 'possible' not in gold_nodes:
                mismatches.append(dev_tuple[0])

    return mismatches

def get_possible_ids():
    possible_ids = list() #(id,snt)
    comments_and_amrs = read_amrz(GOLD_AMRS) #(comment_list, amr_list)
    comments = comments_and_amrs[0] #{'snt','id'}
    amrs = comments_and_amrs[1]
    for i in range(len(amrs)):
        #print(comments[i])
        amr_graph = AMR.parse_AMR_line(amrs[i])
        node_values = amr_graph.node_values
        if 'possible' in node_values:
            possible_ids.append((comments[i]['id'].encode('utf8'),comments[i]['snt'].encode('utf8'),amrs[i].encode('utf8')))
            #print("Added {}".format(comments[i]['id']))
    return sorted(possible_ids,key=lambda x: int(x[0].split(' ')[0].split('.')[1])) #sort by id number

def write_possible_ids():
    possible_ids = get_possible_ids()
    with open('gold_trees.txt','w') as dest:
        for ids,snts,trees in possible_ids:
            dest.write('# ::id {}\n'.format(ids))
            dest.write('# ::snt {}\n'.format(snts))
            tree_string = trees.decode('utf8')
            tree_string = Tree.fromstring(tree_string).pformat()
            dest.write('{}\n\n'.format(tree_string.encode('utf8')))    

def get_possible_devs():
    possible_ids = [item[0] for item in get_possible_ids()]
    possible_devs = list()
    dev_tuples = read_amrz(DEV_AMRS)
    dev_comments = dev_tuples[0]
    dev_amrs = dev_tuples[1]
    for i in range(len(dev_comments)):
        if dev_comments[i]['id'] in possible_ids:
            possible_devs.append({'id':dev_comments[i]['id'],'snt':dev_comments[i]['snt'],'amr':dev_amrs[i]})
    possible_devs = sorted(possible_devs, key=lambda x:int(x['id'].split(' ')[0].split('.')[1]))
    
    with codecs.open('possible_dev.txt','w') as dest:
        for possible_dev in possible_devs:
            dest.write('# ::id {}\n'.format(possible_dev['id']))
            dest.write('# ::snt {}\n'.format(possible_dev['snt'].encode('utf-8')))
            tree_string = possible_dev['amr']
            tree_string = Tree.fromstring(tree_string).pformat()
            dest.write('{}\n\n'.format(tree_string.encode('utf8')))    
    
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
    mismatch = find_mismatch()
    print(len(mismatch))
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
