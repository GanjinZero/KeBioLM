import numpy as np
from tqdm import tqdm
from load_umls import UMLS
import os


def measure_file(answer_file, umls=None, target_file=None):
    with open(answer_file, "r", encoding="utf-8") as f:
        answer_lines = f.readlines()

    with open(target_file, "r", encoding="utf-8") as f:
        target_lines = f.readlines()

    print(answer_file + "|" + target_file)

    correct = 0
    correct_dict = {}
    count_dict = {}
    non_trivial_correct_dict = {}
    non_trivial_count_dict = {}
    for idx, line in enumerate(answer_lines):
        answer = line.strip().lower().split("|")
        answer = answer + ["".join(ans.split()) for ans in answer]
        target_split = target_lines[idx].split("\t")
        target_sentence = target_split[0].lower()
        target_cui = target_split[-1].strip().split("|")
        if len(target_split) == 3:
            target_type = target_split[1]
            if not target_type in count_dict:
                count_dict[target_type] = 0
                correct_dict[target_type] = 0
                non_trivial_correct_dict[target_type] = 0
                non_trivial_count_dict[target_type] = 0
            count_dict[target_type] += 1

        target_answers = set()
        if umls:
            for cui in target_cui:
                target_answers.update(umls.cui2str[cui])
                target_answers.update([cui])
        else:
            target_answers = target_cui

        target_answers = [w.lower() for w in list(target_answers)]
        trivial_question = False
        for a in target_answers:
            if target_sentence.find(a) >= 0:
                trivial_question = True
                break
        if not trivial_question:
            non_trivial_count_dict[target_type] += 1

        for a in answer:
            if a in target_answers:
                correct += 1
                correct_dict[target_type] += 1
                if not trivial_question:
                    non_trivial_correct_dict[target_type] += 1
                break

    # micro_acc = correct / len(answer_lines) * 100
    macro_list = []
    macro_nontrivial = []
    macro_trivial = []
    print("|".join([answer_file, target_file]) + ":")
    for key, value in correct_dict.items():
        macro_list.append(value / count_dict[key] * 100)

        denom = non_trivial_count_dict[key]
        nom = non_trivial_correct_dict[key]
        if denom > 0:
            macro_nontrivial.append(nom / denom * 100)

        denom = count_dict[key] - non_trivial_count_dict[key]
        nom = value - non_trivial_correct_dict[key]
        if denom > 0:
            macro_trivial.append(nom / denom * 100)

    macro_r = sum(macro_list) / len(macro_list)
    trivial_macro_r = sum(macro_trivial) / len(macro_trivial)
    non_trivial_macro_r = sum(macro_nontrivial) / len(macro_nontrivial)
    print(f"Macro-r:{str(macro_r)}")
    print(f"Trivial macro-r:{str(trivial_macro_r)}")
    print(f"Non-Trivial macro-r:{str(non_trivial_macro_r)}")

    return macro_r, trivial_macro_r, non_trivial_macro_r

if __name__ == "__main__":
    predict_file = sys.argv[1]
    label_file = sys.argv[2]
    umls_path = sys.argv[3]
    umls = UMLS(umls_path, only_load_dict=True)
    measure_file(predict_file, umls, label_file)
