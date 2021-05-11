import sys
sys.path.append('../')
from random import sample
from load_umls import UMLS
from tqdm import tqdm
from convert import convert
import os
import pandas as pd
import sys


if __name__ == "__main__":
    umls_path = sys.argv[1]
    umls = UMLS(umls_path, only_load_dict=False)
    
    df = pd.read_csv('build_index.csv')
    lui_list = df['lui'].tolist()
    relation_list = df['relation'].tolist()
    order_list = df['order'].tolist()
    
    with open('dataset.txt', 'w', encoding='utf-8') as f:
        for i in range(len(lui_list)):
            luis = lui_list[i].split('|')
            cuis = [umls.lui2cui[lui] for lui in luis]
            rel = relation_list[i]
            lui_str = umls.lui2str[luis[0]][2]
            order = order_list[i]
            #print(luis, cuis, i, rel, lui_str, order)
            answer_cui = set()
            if order == 0:
                for cui in cuis:
                    if cui in umls.cui0_rel:
                        if rel in umls.cui0_rel[cui]:
                            answer_cui.update(umls.cui0_rel[cui][rel])
                cui0_str = lui_str
                cui1_str = '[MASK]'
            if order == 1:
                for cui in cuis:
                    if cui in umls.cui1_rel:
                        if rel in umls.cui1_rel[cui]:
                            answer_cui.update(umls.cui1_rel[cui][rel])
                cui1_str = lui_str
                cui0_str = '[MASK]'
            relation = convert(rel)
            q = " ".join([cui1_str, relation, cui0_str]) + "."
            a = "|".join(list(answer_cui))
            line = "\t".join([q, rel, a])
            f.write(line + "\n")

