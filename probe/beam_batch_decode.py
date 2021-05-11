import os
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM
from tqdm import tqdm, trange
import copy
import numpy as np
from batch_decode import load_model_tok, read_input, ids2str, output
from batch_decode import get_mask_ids, multi_mask, log_sum, score

import sys
sys.path.append("../")
try:
    from modeling_kebio import KebioForPreTraining
    from entity_indexer import EntityIndexer
except BaseException:
    pass


max_seq_length = 64
batch_size = 96
device = torch.device('cuda:0')


def get_decode_prob(input_ids, model):
    # deal with device
    model.eval()
    input_gpu_0 = torch.LongTensor(input_ids).to(device)
    now_res = model(input_gpu_0)
    if isinstance(now_res, tuple):
        now_res = now_res[0]
    return now_res


def get_mask_ids(input_ids, mask_token_id):
    mask_ids = []
    for input_id in input_ids:
        mask_id = []
        for idx, token_id in enumerate(input_id):
            if token_id == mask_token_id:
                mask_id.append(idx)
        mask_ids.append(mask_id)
    return mask_ids


def init_beam(input_ids, model, tok, mask_ids=None, method='independent', debug=False, beam=2):
    # method in ['independent', 'order', 'confidence']
    # Deal with a batch
    # All sentence are in same length to avoid padding!
    new_input_ids = copy.deepcopy(input_ids)
    output_input_ids = []

    if mask_ids is None:
        mask_ids = get_mask_ids(input_ids, tok.mask_token_id)

    with torch.no_grad():

        count = len(input_ids)
        tqdm_flag = True
        now_count = 0

        if tqdm_flag:
            pbar = tqdm(total=count)
        while now_count < count:

            now_mask_ids = mask_ids[now_count:min(
                now_count + batch_size // beam, count)]
            now_input_ids = new_input_ids[now_count:min(
                now_count + batch_size // beam, count)]

            # For beam
            beam_now_input_ids = []
            beam_now_mask_ids = []
            for tmp in range(len(now_input_ids)):
                for _ in range(beam):
                    beam_now_input_ids.append(now_input_ids[tmp])
                    beam_now_mask_ids.append(now_mask_ids[tmp])

            if method == 'confidence':
                last_input_ids = copy.deepcopy(beam_now_input_ids)
                last_mask_ids = copy.deepcopy(beam_now_mask_ids)
                max_mask_length = max([len(mask_id)
                                       for mask_id in last_mask_ids])

                score_list = [0] * len(beam_now_mask_ids)
                last_score_list = copy.deepcopy(score_list)

                for t in range(max_mask_length):
                    if debug:
                        print(f'Turn {t}')
                    if t > 0:
                        last_input_ids = copy.deepcopy(beam_now_input_ids)
                        last_mask_ids = copy.deepcopy(beam_now_mask_ids)
                        last_score_list = copy.deepcopy(score_list)
                    if debug:
                        print(last_input_ids)
                        print(last_mask_ids)
                    hidden = get_decode_prob(last_input_ids, model)

                    for i in range(0, len(now_input_ids)):
                        if len(last_mask_ids[i * beam]) == 0:
                            continue

                        tmp_dict = {}
                        if t > 0:
                            for j in range(beam):
                                place = i * beam + j
                                # new_mask_ids[place] * vocab
                                log_score = F.log_softmax(
                                    hidden[place][last_mask_ids[place]], dim=-1)
                                shape1 = log_score.shape[1]
                                flat_log_score = log_score.view(-1)
                                topb_log, topb_idx = torch.topk(
                                    flat_log_score, k=beam)
                                for b in range(beam):
                                    x = topb_idx[b] // shape1
                                    y = topb_idx[b] % shape1
                                    tmp_dict[(
                                        place, x, y)] = last_score_list[place] + topb_log[b]
                        if t == 0:
                            place = i * beam
                            # new_mask_ids[place] * vocab
                            log_score = F.log_softmax(
                                hidden[place][last_mask_ids[place]], dim=-1)
                            shape1 = log_score.shape[1]
                            flat_log_score = log_score.view(-1)
                            topb_log, topb_idx = torch.topk(
                                flat_log_score, k=beam)

                            for b in range(beam):
                                x = topb_idx[b] // shape1
                                y = topb_idx[b] % shape1
                                tmp_dict[(place, x, y)
                                         ] = last_score_list[place] + topb_log[b]

                        sort_tmp = sorted(tmp_dict.items(),
                                          key=lambda item: -item[1])[0:beam]
                        if debug:
                            print(tmp_dict)

                        for j in range(beam):
                            (old_place, x, y), new_log = sort_tmp[j]
                            if debug:
                                print(
                                    old_place, last_mask_ids[old_place][x], y.item(), new_log.item())
                            new_place = i * beam + j
                            score_list[new_place] = new_log
                            beam_now_input_ids[new_place] = copy.deepcopy(
                                last_input_ids[old_place])
                            beam_now_input_ids[new_place][last_mask_ids[old_place][x]] = y
                            beam_now_mask_ids[new_place] = copy.deepcopy(
                                last_mask_ids[old_place])
                            beam_now_mask_ids[new_place].remove(
                                last_mask_ids[old_place][x])

                            if debug:
                                print(last_input_ids[old_place],
                                      beam_now_input_ids[new_place])

            for i in range(len(now_input_ids)):
                output_j = []
                for j in range(beam):
                    output_j.append(beam_now_input_ids[i * beam + j])
                output_input_ids.append(output_j)

            if tqdm_flag:
                pbar.update(min(now_count + batch_size //
                                beam, count) - now_count)
            now_count = min(now_count + batch_size // beam, count)

    if tqdm_flag:
        pbar.close()

    return output_input_ids


def get_answers(input_ids, mask_ids, tok):
    decode_res = []
    for i in range(len(input_ids)):
        answer_i = []
        for j in range(len(input_ids[i])):
            ans_ids = np.array(input_ids[i][j])[mask_ids[i]]
            answer_i.append(tok.decode(ans_ids, skip_special_tokens=True))
        decode_res.append(answer_i)
    return decode_res


def multi_mask_decode(sentence_lst, model, tok, mask_count=5, method='confidence', beam=2):
    input_ids = []
    map_idx = {i: [] for i in range(1, max_seq_length + 1)}
    map_overall = {i: [] for i in range(1, max_seq_length + 1)}
    answers_set = {i: [] for i in range(len(sentence_lst))}

    print("Sentence Tokenize")
    overall_idx = 0
    for sentence_idx, sen in tqdm(enumerate(sentence_lst)):
        now_input_id = tok.encode_plus(
            sen, max_length=max_seq_length, add_special_tokens=True)['input_ids']
        multi_mask_now_input_ids = multi_mask(
            now_input_id, tok.mask_token_id, mask_count, max_seq_length)

        for mask_cnt, input_id in enumerate(multi_mask_now_input_ids):
            lth = len(input_id)
            map_idx[lth].append((sentence_idx, mask_cnt))
            map_overall[lth].append(overall_idx)
            input_ids.append(input_id)
            overall_idx += 1

    for l in range(max_seq_length, 0, -1):
        print(l)
        l_input_ids = []
        for idx in map_overall[l]:
            l_input_ids.append(input_ids[idx])
        if not l_input_ids:
            continue
        mask_ids = get_mask_ids(l_input_ids, tok.mask_token_id)
        init0 = init_beam(l_input_ids, model, tok,
                          mask_ids, method, False, beam)
        answers = get_answers(init0, mask_ids, tok)
        for idx in range(len(map_overall[l])):
            answers_set[map_idx[l][idx][0]].extend(answers[idx])

    return answers_set


def output(answers_set, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(answers_set)):
            ans = answers_set[i]
            f.write("|".join(ans) + "\n")
    return None


def main(model_name_or_path, input_file, output_file=None, mask_count=5, method='confidence', beam=2):
    model, tok = load_model_tok(model_name_or_path)
    sentence_lst = read_input(input_file)
    answers_set = multi_mask_decode(
        sentence_lst, model, tok, mask_count, method, beam)
    if output_file is None:
        base_name = os.path.basename(input_file).split(".")[0]
        if model_name_or_path[-1] == "/":
            model_name_or_path = model_name_or_path[:-1]
        try:
            base_path = os.path.basename(model_name_or_path)
        except BaseException:
            base_path = model_name_or_path.split("/")[-1]
        output_file = os.path.join("./probe_result/", "-".join(
            [base_path, base_name, method, str(mask_count), str(beam)]))
    output(answers_set, output_file)
    return


if __name__ == "__main__":
    import sys
    model_name_or_path = sys.argv[1]
    input_file = sys.argv[2]
    mask_count = 10
    method = 'confidence'
    beam = 5
    main(model_name_or_path, input_file,
         mask_count=mask_count, method=method, beam=beam)
