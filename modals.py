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

COMPLEMENT_TYPO = [1453, 1485, 1497, 1558, 1615, 1916, 1934, 1954, 2247, 2397, 2437, 2447, 2449, 2451, 2453]
#REGEX missed 过 通 回来 响 懂 及 受
#1453 1954 2397 2451? has typo

AMR_DATA_DIR = os.path.join(os.pardir,'amrz','data')
RESULTS_DIR = os.path.join(os.pardir,'results')
#path to training AMRs
GOLD_AMRS = os.path.join(os.curdir, 'amr_zh_all.txt.amr')
print(GOLD_AMRS)
#path to dev AMRs
REPHRASED_GOLD = os.path.join(os.curdir, "amr_zh_all_rephrased.txt.amr")
BASIC_TEST = os.path.join(AMR_DATA_DIR,'amr_zh_all.txt.test.amr.basic_abt_feat.parsed')
REPHRASED_TEST = os.path.join(os.curdir,'amr_zh_all_rephrased.txt.test.amr.basic_abt_feat.parsed')
SIBLING_TEST = os.path.join(AMR_DATA_DIR,'amr_zh_all.txt.test.amr.sibling_feat.parsed')
SIBLING_BIGRAM_TEST = os.path.join(AMR_DATA_DIR,'amr_zh_all.txt.test.amr.sibling_bigram_feat.parsed')
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

def get_dependency_tuple_elements(deptup):
    relre = re.compile(r"(\w+)\((\w+-\d+), ([^\-]+-\d+)\)")
    relre_match = relre.match(deptup)
    if relre_match:
        return (relre_match[1],relre_match[2],relre_match[3])

def latex_dependency(sentstring,tupstring,dest_fname):
    with open(dest_fname,'w') as dest:
        dest.write("\\begin{dependency}[theme = simple]\n")
        dest.write("\\begin{deptext}[column sep=13m]\n")
        toks = sentstring.split(" ")
        sent =" \\& ".join(toks) + " \\\\\n"
        dest.write(sent)
        dest.write("\\end{deptext}\n")
        deps = tupstring.split("\n")
        for dep in deps:
            #print(dep)
            if len(dep) > 0:
                rel, first, second = get_dependency_tuple_elements(dep)
                first_num = first.split('-')[1]
                second_num = second.split('-')[1]
                if rel == 'root':
                    dest.write("\\deproot{{{}}}{{'ROOT'}}\n".format(second_num))
                else:
                    #dest.write("\\depedge{{{}}}{{{}}}{{{}}}\n".format(second_num,first_num,rel))
                    dest.write("\\depedge{{{}}}{{{}}}{{{}}}\n".format(first_num,second_num,rel))
        dest.write("\\end{dependency}")

def rephrase_amrs(dest_file,all_amr_file=None):
    """Rephrase AMRs by removing complement constructions"""
    #pos_complement = re.compile(r'(\w+)\s[得]\s[了起到出来错完上下住]')
    #neg_complement = re.compile(r'(\w+)\s[不]\s[了起到出来错完上下住]')
    #leaving out guo because of errors
    pos_complement = re.compile(r"(\S)\s[得]\s[了起到出来错完上下住通回响懂及受]")
    neg_complement = re.compile(r"(\S)\s[不]\s[了起到出来错完上下住通回响懂及受]")

    rephrased_ids = set()
    rephrased_snt = None
    current_id = None
    if all_amr_file is None:
        all_amr_file = UNSEG_SENTS
    with open(all_amr_file) as source:
        with open(dest_file,'w') as dest:
            for line in source:
                if line.startswith("# ::id"):
                    current_id = line
                    dest.write(line)
                elif line.startswith("# ::snt"):
                    if pos_complement.search(line):
                        line =  pos_complement.sub(r'能 \g<1>',line)
                        rephrased_ids.add(current_id)
                    elif neg_complement.search(line):
                        line =  neg_complement.sub(r'不 能 \g<1>',line)
                        rephrased_ids.add(current_id)
                    rephrased_snt = line.replace("# ::snt ","")
                    dest.write(line)
                elif line.startswith("# ::wid"):
                    sent = rephrased_snt
                    toks = [" " if tok == "\n" else tok for tok in rephrased_snt.split(" ")]
                    #print(toks)
                    numbered_toks = ["x{}_{}".format(num,tok) for num, tok in enumerate(toks,1)]
                    wids = ' '.join(numbered_toks)
                    dest.write("# ::wid {}\n".format(wids))
                else:
                    line = line.replace('possible','能-01')
                    dest.write(line)

        #print("Rephrased sentences: {}".format(rephrased_ids))

def get_amrs_with_concept(concept,all_amr_file=None):
    """Get the IDs of all AMRs with 'possible' concept
    Inputs:
        amr_file: file with all the AMRs
    Returns:
        list of (id, snt, amr) tuples
    """
    if all_amr_file is None:
        all_amr_file = GOLD_AMRS
    match_amrs = list() #(id,snt)
    comments_and_amrs = read_amrz(all_amr_file) #(comment_list, amr_list)
    comments = comments_and_amrs[0] #{'snt','id'}
    amrs = comments_and_amrs[1]
    for i in range(len(amrs)):
        amr_graph = AMR.parse_AMR_line(amrs[i])
        node_values = amr_graph.node_values
        if concept in node_values:
            match_amrs.append((comments[i]['id'],comments[i]['snt'],amrs[i]))
            #possible_ids.append((comments[i]['id'].encode('utf8'),comments[i]['snt'].encode('utf8'),amrs[i].encode('utf8')))
    print("Total number of AMRs with '{}': {}".format(concept,len(match_amrs)))
    return sorted(match_amrs,key=lambda x: int(x[0].split(' ')[0].split('.')[1])) #sort by id number

def sub_parsed_with_gold(concept, gold_amrs, parsed_amrs,destfile):
    """Switch gold with parsed for given concept"""
    subbed_amrs = list()
    gold_with_concept = get_amrs_with_concept(concept,gold_amrs)
    print("Found {1} ex of {0} in gold file".format(concept,len(gold_with_concept)))
    gold_ids = [gold[0] for gold in gold_with_concept]
    print("Ids: {}".format(len(gold_ids)))
    parsed_comments, parsed_amrs = read_amrz(parsed_amrs)
    print("Parsed file length: {}".format(len(parsed_comments)))
    for i in range(len(parsed_comments)):
        if parsed_comments[i]['id'] in gold_ids:
            gold_match = [match for match in gold_with_concept if match[0] == parsed_comments[i]['id']][0]
            subbed_amrs.append(gold_match)
        else:
            subbed_amrs.append((parsed_comments[i]['id'],parsed_comments[i]['snt'],parsed_amrs[i]))
    print("Total: {}".format(len(subbed_amrs)))
    write_match_amrs(subbed_amrs,destfile)

def count_verb_complements():
    complement_counts = dict()
    complement_phrase_counts = dict()
    concept = "possible"
    possible_amrs = get_amrs_with_concept(concept,all_amr_file=GOLD_AMRS)
    pos_complement = re.compile(r'(\w+)\s([得])\s([了起到出来错完上下过通回响懂及受])')
    neg_complement = re.compile(r'(\w+)\s([不])\s([了起到出来错完上下过通回响懂及受])')
    for possible_amr in possible_amrs:
        sent = possible_amr[1]
        pos_match = pos_complement.search(sent)
        if pos_match:
            complement = pos_match.group(2) + pos_match.group(3)
            phrase = pos_match.group(0)
            complement_counts[complement] = complement_counts.get(complement,0) + 1
            complement_phrase_counts[phrase] = complement_phrase_counts.get(phrase,0) + 1
        neg_match = neg_complement.search(sent)
        if neg_match:
            complement = neg_match.group(2) + neg_match.group(3)
            phrase = neg_match.group(0)
            complement_counts[complement] = complement_counts.get(complement,0) + 1
            complement_phrase_counts[phrase] = complement_phrase_counts.get(phrase,0) + 1
    for complement in sorted(complement_counts.keys()):
        print("{}: {}".format(complement,complement_counts[complement]))
    for phrase in sorted(complement_phrase_counts.keys()):
        print("{}: {}".format(phrase,complement_phrase_counts[phrase]))

#def write_possible_amrs(destfile,all_amr_file=None):
def write_match_amrs(amr_list,destfile):
    """Write all trees with 'possible' as a concept"""
    #if all_amr_file is None:
        #all_amr_file = GOLD_AMRS
    #possible_ids = get_possible_amrs(all_amr_file)
    with open(destfile,'w') as dest:
        for ids,snts,trees in amr_list:
            token_ids = enumerate(snts.split(),1)
            wids = ["x{}_{}".format(num, token) for num, token in token_ids]
            wid_string = ' '.join(wids)
            dest.write('# ::id {}\n'.format(ids))
            dest.write('# ::snt {}\n'.format(snts))
            dest.write('# ::wid {}\n'.format(wid_string))
            #tree_string = trees.decode('utf8')
            tree_string = trees
            """
            try:
                tree_string = Tree.fromstring(tree_string).pformat()
            except ValueError:
                tree_string += ")"
                tree_string = Tree.fromstring(tree_string).pformat()
            #dest.write('{}\n\n'.format(tree_string.encode('utf8')))
            """
            dest.write('{}\n\n'.format(tree_string))

def compare_concepts(match_amr_list, comparison_amr_file=None):
    """Compare results with dev or test set
    Inputs:
        all_amr_file: filename with all amrs
        dev_amr_file: filename with dev amrs
    Returns:
        possible_devs: list of (id, snt, amr) tuples
    """
    if comparison_amr_file is None:
        comparison_amr_file = BASIC_TEST
    match_ids = [item[0] for item in match_amr_list]
    comparison_matches = list()
    comparison_tuples = read_amrz(comparison_amr_file)
    comparison_comments = comparison_tuples[0]
    comparison_amrs = comparison_tuples[1]
    for i in range(len(comparison_comments)):
        if comparison_comments[i]['id'] in match_ids:
            #possible_devs.append({'id':dev_comments[i]['id'],'snt':dev_comments[i]['snt'],'amr':dev_amrs[i]})
            comparison_matches.append((comparison_comments[i]['id'],comparison_comments[i]['snt'],comparison_amrs[i]))
    #possible_devs = sorted(possible_devs, key=lambda x:int(x['id'].split(' ')[0].split('.')[1]))
    comparison_matches = sorted(comparison_matches, key=lambda x: int(x[0].split(' ')[0].split('.')[1]))
    print("Total number of sentences in comparison set that should match with all_amr_set: {}".format(len(comparison_matches)))
    return comparison_matches

def concept_mismatch(all_amr_list, comparison_amr_list, concept):
    """Find cases where dev amr should have 'possibilty' but doesn't"""
    missing = list()
    spurious = list()
    correct = list()
    for comparison_amr in comparison_amr_list:
        matches = [base_tuple for base_tuple in all_amr_list if base_tuple[0] == comparison_amr[0]]
        if len(matches) > 0:
            match = matches[0]
            match_amr = AMR.parse_AMR_line(match[2])
            match_nodes = match_amr.node_values
            comparison_nodes = AMR.parse_AMR_line(comparison_amr[2]).node_values
            id_number = int(comparison_amr[0].split('::')[0].split('.')[1])
            if concept in match_nodes and concept not in comparison_nodes:
                #missing.append(comparison_amr[0])
                missing.append((id_number, comparison_amr[1]))
            elif concept in comparison_nodes and concept not in match_nodes:
                #spurious.append(comparison_amr[0])
                spurious.append((id_number, comparison_amr[1]))
            else:
                #correct.append(comparison_amr[0])
                correct.append((id_number, comparison_amr[1]))
    return sorted(correct), sorted(missing), sorted(spurious)

def test_concepts():
    pos_complement = re.compile(r'(\w+)\s([得])\s([了起到出来错完上下过通回响懂及受])')
    neg_complement = re.compile(r'(\w+)\s([不])\s([了起到出来错完上下过通回响懂及受])')

    model_comparisons = dict()
    test_tuples = [(GOLD_AMRS, BASIC_TEST, "能-01"),
                    (REPHRASED_GOLD, REPHRASED_TEST, "能-01"),
                    (GOLD_AMRS, BASIC_TEST, "possible"),
                    (GOLD_AMRS, SIBLING_TEST, "possible"),
                    (GOLD_AMRS, SIBLING_BIGRAM_TEST, "possible")
    ]


    for gold_amr, test_amr, concept in test_tuples:
        print("Testing identification of {} with {} as gold and {} as test".format(concept,gold_amr,test_amr))
        gold_concept_matches = get_amrs_with_concept(concept,gold_amr)
        test_concept_matches = compare_concepts(gold_concept_matches,test_amr)
        correct, missing, spurious = concept_mismatch(gold_concept_matches,test_concept_matches,concept)
        correct_ids = [item[0] for item in correct]
        correct_sents = [item[1] for item in correct]
        missing_ids = [item[0] for item in missing]
        missing_sents = [item[1] for item in missing]
        print("Missing '{0}': {1} \n {2}".format(concept, len(missing),missing_ids))

        #print("Sents missing '{0}': {1}".format(concept, missing_sents))
        #print("Sents correctly identifying '{0}': {1}".format(concept, correct_sents))

        if concept == "possible":
            missing_complement_counts = dict()
            correct_complement_counts = dict()
            missing_phrase_counts = dict()
            correct_phrase_counts = dict()
            for sent in missing_sents:
                pos_complement_match = pos_complement.search(sent)
                if pos_complement_match:
                    verb = pos_complement_match.group(1)
                    particle = pos_complement_match.group(2)
                    complement = pos_complement_match.group(3)
                    phrase = pos_complement_match.group(0)
                    #for item in (verb, particle, complement):
                        #missing_complement_counts[item] = missing_complement_counts.get(item,0) + 1
                    missing_complement_counts[particle + complement] = missing_complement_counts.get(particle+complement,0) + 1
                    missing_phrase_counts[phrase] = missing_phrase_counts.get(phrase,0) + 1
                neg_complement_match = neg_complement.search(sent)
                if neg_complement_match:
                    verb = neg_complement_match.group(1)
                    particle = neg_complement_match.group(2)
                    complement = neg_complement_match.group(3)
                    phrase = neg_complement_match.group(0)
                    #for item in (verb, particle, complement):
                        #missing_complement_counts[item] = missing_complement_counts.get(item,0) + 1
                    missing_complement_counts[particle + complement] = missing_complement_counts.get(particle+complement,0) + 1
                    missing_phrase_counts[phrase] = missing_phrase_counts.get(phrase,0) + 1

            for sent in correct_sents:
                pos_complement_match = pos_complement.search(sent)
                if pos_complement_match:
                    verb = pos_complement_match.group(1)
                    particle = pos_complement_match.group(2)
                    complement = pos_complement_match.group(3)
                    phrase = pos_complement_match.group(0)

                    #for item in (verb, particle, complement):
                        #correct_complement_counts[item] = correct_complement_counts.get(item,0) + 1
                    correct_complement_counts[particle + complement] = correct_complement_counts.get(particle+complement,0) + 1
                    correct_phrase_counts[phrase] = correct_phrase_counts.get(phrase,0) + 1

                neg_complement_match = neg_complement.search(sent)
                if neg_complement_match:
                    verb = neg_complement_match.group(1)
                    particle = neg_complement_match.group(2)
                    complement = neg_complement_match.group(3)
                    phrase = neg_complement_match.group(0)

                    #for item in (verb, particle, complement):
                        #correct_complement_counts[item] = missing_complement_counts.get(item,0) + 1
                    correct_complement_counts[particle + complement] = correct_complement_counts.get(particle+complement,0) + 1
                    correct_phrase_counts[phrase] = correct_phrase_counts.get(phrase,0) + 1


            print("Complement counts for missing {}:".format(concept))
            for key in sorted(list(missing_complement_counts.keys())):
                print("{}: {}".format(key, missing_complement_counts[key]))

            print("Complement counts for correct {}:".format(concept))
            for key in sorted(list(correct_complement_counts.keys())):
                print("{}: {}".format(key, correct_complement_counts[key]))

            print("Phrase counts for missing {}:".format(concept))
            for key in sorted(list(missing_phrase_counts.keys())):
                print("{}: {}".format(key, missing_phrase_counts[key]))

            print("Phrase counts for correct {}:".format(concept))
            for key in sorted(list(correct_phrase_counts.keys())):
                print("{}: {}".format(key, correct_phrase_counts[key]))


    gold_concept_matches = get_amrs_with_concept("possible",GOLD_AMRS)
    basic_test_concept_matches = compare_concepts(gold_concept_matches,BASIC_TEST)
    sibling_test_concept_matches = compare_concepts(gold_concept_matches,SIBLING_TEST)

    basic_correct, basic_missing, basic_spurious = concept_mismatch(gold_concept_matches,basic_test_concept_matches,"possible")
    sibling_correct, sibling_missing, sibling_spurious = concept_mismatch(gold_concept_matches,sibling_test_concept_matches,"possible")

    basic_not_sibling_possible = len([item[0] for item in basic_correct if item in sibling_missing])
    print("Possibles identified in the basic model but not the sibling model:{}".format(basic_not_sibling_possible))

    sibling_not_basic_possible = len([item[0] for item in sibling_correct if item in basic_missing])
    print("Possibles identified in the sibling model but not the basic model:{}".format(sibling_not_basic_possible))

    gold_concept_matches = get_amrs_with_concept("能-01",GOLD_AMRS)
    basic_test_concept_matches = compare_concepts(gold_concept_matches,BASIC_TEST)
    rephrased_test_concept_matches = compare_concepts(gold_concept_matches,REPHRASED_TEST)
    basic_correct, basic_missing, basic_spurious = concept_mismatch(gold_concept_matches,basic_test_concept_matches,"能-01")
    rephrased_correct, rephrased_missing, rephrased_spurious = concept_mismatch(gold_concept_matches,rephrased_test_concept_matches,"能-01")
    rephrased_extra_neng = len([item[0] for item in rephrased_missing if item in basic_correct])
    print("Nengs missing from rephrased model that were actually nengs in the basic model:{}".format(rephrased_extra_neng))


if __name__ == "__main__":
    #sub_parsed_with_gold("possible",GOLD_AMRS,SIBLING_BIGRAM_TEST,"sibling_subbed.txt")
    #count_verb_complements()
    test_concepts()
    #rephrase_amrs("amr_zh_all_rephrased02.txt",all_amr_file=UNSEG_SENTS)
    #write_possible_amrs(amr_list=rephrased,destfile='../results/amr_zh_10k_rephrased.txt')

    #latex_dependency(snt1,depstring1,"dep1.txt")
    #latex_dependency(snt2,depstring2,"dep2.txt")

    #write_match_amrs(possible_match,"possible.txt")
