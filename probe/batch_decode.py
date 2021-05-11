import os
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM
from tqdm import tqdm, trange
import copy
import numpy as np

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


def init(input_ids, model, tok, mask_ids=None, method='independent', debug=False):
    # method in ['independent', 'order', 'confidence']
    # Deal with a batch
    # All sentence are in same length to avoid padding!
    new_input_ids = copy.deepcopy(input_ids)

    if mask_ids is None:
        mask_ids = get_mask_ids(input_ids, tok.mask_token_id)

    with torch.no_grad():

        count = len(input_ids)
        tqdm_flag = False
        if count > 1000:
            tqdm_flag = True
        now_count = 0

        if tqdm_flag:
            pbar = tqdm(total=count)
        while now_count < count:
            now_input_ids = input_ids[now_count:min(now_count + batch_size, count)]
            now_mask_ids = mask_ids[now_count:min(now_count + batch_size, count)]

            if method == 'independent':
                hidden = get_decode_prob(now_input_ids, model) # batch_size * seq_length * vocab_size
                for i in range(0, len(now_input_ids)):
                    for idx in now_mask_ids[i]:
                        lm_prob = hidden[i][idx]
                        decode = torch.argmax(lm_prob)
                        new_input_ids[i + now_count][idx] = decode.item()

            if method == 'order':
                max_mask_length = max([len(mask_id) for mask_id in now_mask_ids])
                for t in range(max_mask_length):
                    now_new_input_ids = new_input_ids[now_count:min(now_count + batch_size, count)]
                    if debug:
                        print(f'Turn {t}')
                    hidden = get_decode_prob(now_new_input_ids, model)
                    for i in range(0, len(now_input_ids)):
                        if len(mask_ids[i]) <= t:
                            continue
                        idx = mask_ids[i][t]
                        lm_prob = hidden[i][idx]
                        decode = torch.argmax(lm_prob)
                        new_input_ids[i + now_count][idx] = decode.item()
                        if debug:
                            print(f'{i} {idx} -> {decode}')

            if method == 'confidence':
                new_mask_ids = copy.deepcopy(now_mask_ids)
                max_mask_length = max([len(mask_id) for mask_id in new_mask_ids])
                for t in range(max_mask_length):
                    now_new_input_ids = new_input_ids[now_count:min(now_count + batch_size, count)]
                    if debug:
                        print(f'Turn {t}')
                    hidden = get_decode_prob(now_new_input_ids, model)

                    for i in range(0, len(now_input_ids)):
                        if len(new_mask_ids[i]) == 0:
                            continue
                        idx = new_mask_ids[i][torch.argmax(
                            torch.max(F.softmax(hidden[i][new_mask_ids[i]], dim=-1), dim=1)[0])]
                        lm_prob = hidden[i][idx]
                        decode = torch.argmax(lm_prob)
                        new_input_ids[i + now_count][idx] = decode.item()
                        new_mask_ids[i].remove(idx)
                        if debug:
                            print(f'{i} {idx} -> {decode}')

            if tqdm_flag: 
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)

    if tqdm_flag:
        pbar.close()

    return new_input_ids


def log_sum(l, average=True):
    trans = [np.log(t) for t in l]
    if average:
        return np.mean(trans)
    return sum(trans)


def score(input_ids, model, tok, mask_ids, length_normalize=True):
    score_list = []
    total_length = sum([len(x) for x in mask_ids])
    new_input_ids = [[]] * total_length

    now = 0
    for i in range(len(input_ids)):
        for mask_id in mask_ids[i]:
            new_input_ids[now] = input_ids[i]
            new_input_ids[now][mask_id] = tok.mask_token_id
            now += 1

    prob = get_decode_prob(new_input_ids, model)

    now = 0
    for i in range(len(input_ids)):
        tmp = []
        for mask_id in mask_ids[i]:
            conf = F.softmax(prob[now][mask_id], dim=0)[
                np.asarray(input_ids[i])[mask_id]].item()
            tmp.append(conf)
            now += 1
        score_list.append(log_sum(tmp, length_normalize))
    return score_list


def ids2str(input_ids, tok):
    return tok.decode(input_ids, skip_special_tokens=True)


def get_answers(input_ids, mask_ids, tok):
    decode_res = []
    for i in range(len(input_ids)):
        ans_ids = np.array(input_ids[i])[mask_ids[i]]
        decode_res.append(tok.decode(ans_ids, skip_special_tokens=True))
    return decode_res


def multi_mask(input_id, mask_token_id, mask_count=5, max_seq_length=64):
    output_input_ids = [input_id]

    insert_id = None
    for idx, id in enumerate(input_id):
        if id == mask_token_id:
            insert_id = idx
            break
    if insert_id is None:
        return output_input_ids

    for i in range(2, mask_count + 1):
        tmp_id = input_id[0:insert_id] + [mask_token_id] * i + input_id[insert_id + 1:]
        if len(tmp_id) > max_seq_length:
            tmp_id = tmp_id[0:max_seq_length]
        output_input_ids.append(tmp_id)

    return output_input_ids


def multi_mask_decode(sentence_lst, model, tok, mask_count=5, method='confidence'):
    input_ids = []
    map_idx = {i:[] for i in range(1, max_seq_length + 1)}
    map_overall = {i:[] for i in range(1, max_seq_length + 1)}
    answers_set = {i:[] for i in range(len(sentence_lst))}

    print("Sentence Tokenize")
    overall_idx = 0
    for sentence_idx, sen in tqdm(enumerate(sentence_lst)):
        now_input_id = tok.encode_plus(
            sen, max_length=max_seq_length, add_special_tokens=True)['input_ids']
        multi_mask_now_input_ids = multi_mask(now_input_id, tok.mask_token_id, mask_count, max_seq_length)
        
        for mask_cnt, input_id in enumerate(multi_mask_now_input_ids):
            lth = len(input_id)
            map_idx[lth].append((sentence_idx, mask_cnt))
            map_overall[lth].append(overall_idx)
            input_ids.append(input_id)
            overall_idx += 1

    for l in range(max_seq_length, 0, -1):
        l_input_ids = []
        for idx in map_overall[l]:
            l_input_ids.append(input_ids[idx])
        if not l_input_ids:
            continue
        mask_ids = get_mask_ids(l_input_ids, tok.mask_token_id)
        init0 = init(l_input_ids, model, tok, mask_ids, method)
        answers = get_answers(init0, mask_ids, tok)
        for idx in range(len(map_overall[l])):
            answers_set[map_idx[l][idx][0]].append(answers[idx])

    return answers_set

def output(answers_set, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(answers_set)):
            ans = answers_set[i]
            f.write("|".join(ans) + "\n")
    return None


def load_model_tok(model_name_or_path):
    if not os.path.exists(os.path.join(model_name_or_path, 'entity.jsonl')):
        model = BertForMaskedLM.from_pretrained(model_name_or_path).to(device)
    else:
        #entity_indexer = EntityIndexer(
        #    entity_file=os.path.join(model_name_or_path, 'entity.jsonl'))
        model = KebioForPreTraining.from_pretrained(
            model_name_or_path).to(device)#, entity_indexer=entity_indexer).to(device)
    tok = AutoTokenizer.from_pretrained(model_name_or_path)    
    return model, tok


def read_input(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.split("\t")[0] for line in lines]
    return lines


def main(model_name_or_path, input_file, output_file=None, mask_count=5, method='confidence'):
    model, tok = load_model_tok(model_name_or_path)
    sentence_lst = read_input(input_file)
    answers_set = multi_mask_decode(sentence_lst, model, tok, mask_count, method)
    if output_file is None:
        base_name = os.path.basename(input_file).split(".")[0]
        if model_name_or_path[-1] == "/":
            model_name_or_path = model_name_or_path[:-1]
        try:
            base_path = os.path.basename(model_name_or_path)
        except BaseException:
            base_path = model_name_or_path.split("/")[-1]
        output_file = os.path.join("./probe_result", "-".join([base_path, base_name, method, str(mask_count)]))
    output(answers_set, output_file)
    return

if __name__ == "__main__":
    import sys
    model_name_or_path = sys.argv[1]
    input_file = sys.argv[2]
    mask_count = 10
    method = 'confidence'
    main(model_name_or_path, input_file, mask_count=mask_count, method=method)
