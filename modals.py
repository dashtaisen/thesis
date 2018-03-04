# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/Users/nicholasmiller/Documents/2018spring/thesis/amrz/')
sys.path.insert(0, '/Users/nicholasmiller/Documents/2018spring/thesis/amr-evaluation-master/smatch/')

from preprocess import read_amrz
from amr import AMR
from smatch import get_amr_line

#path to training/testing unsegmented sents
UNSEG_SENTS = '../amrz/data/camrnew/amr_zh_10k.txt'

#path to training/testing segmented sents

#path to training AMRs
GOLD_AMRs = '../amrz/data/camrnew/amr_zh_10k.txt.amr'

#path to dev AMRs
DEV_AMRs = '../amr_zh_all.txt.dev.amr.basic_abt_feat.1.parsed'
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
    #comments_and_amrs = read_amrz(UNSEG_SENTS) #(comment_list, amr_list)
    """
    nodes: names of nodes in AMR graph, e.g. "a11", "n"
    node_values: values of nodes in AMR graph, e.g. "group" for a node named "g"
    relations: list of relations between two nodes
    attributes: list of attributes (links between one node and one constant value)

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
