import os, pandas as pd
import pdb

import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from src.utils import config
import torch.utils.data as data
from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import DIA_EMO_MAP as dia_emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
relations = ["oIntent", "oNeed", "oWant", "oEffect", "oReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, item, data_dict):
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)
    return cs_list
    # data_dict["utt_cs"].append(cs_list)


def encode_ctx(vocab, items, data_dict, comet):
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        cs_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)  # pos
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                        w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    e_list.append(w[0])
            # if i == len(ctx) - 1:
            cs_list.append(get_commonsense(comet, item, data_dict))
        data_dict["utt_cs"].append(cs_list)
        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(e_list)


def encode(vocab, file):
    from src.utils.comet import Comet
    import copy
    c = copy.deepcopy
    edos_path = file
    data = {
        "context": [],
        "target": [],
        "emotion": [],
        "dia_emotion": []
    }
    df = pd.read_csv(edos_path, dtype=str)
    context = []
    emotions = []
    dia_emotion = []
    cnt = 0
    # print(df.head)
    for line in df.iterrows():
        info = line[1]
        if info[2] == '1':
            if context:
                data['target'].append(c(context[-1]))
                data['context'].append(c(context[:-1]))
                data['emotion'].append(c(emotions))
                data['dia_emotion'].append(c(dia_emotion))
            else:
                cnt += 1
            context.clear()
            emotions.clear()
            dia_emotion.clear()

        context.append(info[3])
        emotions.append(info[4])
        dia_emotion.append(info[5])
    # the last data
    data['target'].append(c(context[-1]))
    data['context'].append(c(context[:-1]))
    data['emotion'].append(c(emotions))
    data['dia_emotion'].append(c(dia_emotion))

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "dia_emotion": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    comet = Comet("data/Comet", config.device)
    for i, k in enumerate(data.keys()):

        if k == "context":
            encode_ctx(vocab, data[k], data_dict, comet)
        elif k == "emotion" or k == 'dia_emotion':
            data_dict[k] = data[k]
            pass
        else:
            for x, item in tqdm(enumerate(data[k])):
                item = process_sent(item)
                data_dict[k].append(item)
                # data_dict[k][x] = process_sent(item)
                vocab.index_words(item)

        if i == 3:
            break
    assert (
            len(data_dict["context"])
            == len(data_dict["target"])
            == len(data_dict["emotion"])
            == len(data_dict["emotion_context"])
            == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    # files = DATA_FILES(config.data_dir)
    # train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    # dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    # test_files = [np.load(f, allow_pickle=True) for f in files["test"]]
    train_files = 'data/ED/train.csv'
    dev_files = 'data/ED/valid.csv'
    test_files = 'data/ED/test.csv'

    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab


def load_dataset():
    data_dir = config.data_dir
    if config.others:
        data_preproc = 'dataset_o_preproc.p'
    else:
        data_preproc = 'dataset_preproc.p'

    cache_file = f"{data_dir}/{data_preproc}"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_files(
            vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")
    for i in range(20):
        print("[dia_emotion]:", data_tra["dia_emotion"][i][0])
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.dia_emo_map = dia_emo_map
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["dia_emotion_text"] = self.data["dia_emotion"][index][0]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]
        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)
        item["dia_emotion"], item["dia_emotion_label"] = self.preprocess_diaemo(item["dia_emotion_text"],
                                                                                self.dia_emo_map)

        (
            item["emotion_context"],
            item["emotion_context_mask"],
        ) = self.preprocess(item["emotion_context"])

        item["cs_text"] = self.data["utt_cs"][index]
        # item["x_intent_txt"] = item["cs_text"][0]
        # item["cs_text"] = [item["cs_text"][-1]]
        item["x_intent_txt"] = [cs[0] for cs in item["cs_text"]]
        item["x_need_txt"] = [cs[1] for cs in item["cs_text"]]
        item["x_want_txt"] = [cs[2] for cs in item["cs_text"]]
        item["x_effect_txt"] = [cs[3] for cs in item["cs_text"]]
        item["x_react_txt"] = [cs[4] for cs in item["cs_text"]]

        item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
        item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
        item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
        item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
        item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")

        return item

    def preprocess(self, arr, anw=False, cs=None, emo=False):
        """Converts words to ids."""
        if anw:
            sequence = [
                           self.vocab.word2index[word]
                           if word in self.vocab.word2index
                           else config.UNK_idx
                           for word in arr
                       ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:

            res = []
            for utter in arr:
                sequence = [config.CLS_idx] if cs != "react" else []
                for sent in utter:
                    sequence += [
                        self.vocab.word2index[word]
                        for word in sent
                        if word in self.vocab.word2index and word not in ["to", "none"]
                    ]
                res.append(torch.LongTensor(sequence))
            return res
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)

        else:
            # x_dial = [config.CLS_idx]
            # x_mask = [config.CLS_idx]
            # for i, sentence in enumerate(arr):
            #     x_dial += [
            #         self.vocab.word2index[word]
            #         if word in self.vocab.word2index
            #         else config.UNK_idx
            #         for word in sentence
            #     ]
            #     spk = (
            #         self.vocab.word2index["USR"]
            #         if i % 2 == 0
            #         else self.vocab.word2index["SYS"]
            #     )
            #     x_mask += [spk for _ in range(len(sentence))]
            # assert len(x_dial) == len(x_mask)
            x_dial = []
            x_mask = []
            if len(arr) == 0:  # empty utter
                return [torch.LongTensor([config.CLS_idx])], [torch.LongTensor([config.CLS_idx])]
            for i, sentence in enumerate(arr):  # utterances
                dial = [config.CLS_idx] + [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]

                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                mask = [config.CLS_idx] + [spk for _ in range(len(sentence))]
                x_dial.append(torch.LongTensor(dial))
                x_mask.append(torch.LongTensor(mask))
            assert len(x_dial) == len(x_mask)
            return x_dial, x_mask

    def preprocess_diaemo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]

    def preprocess_emo(self, emotion, emo_map):
        programs = []
        re_emo_map = []
        for x, emo in enumerate(emotion):
            program = [0] * len(emo_map)
            if emo in emo_map.keys():
                program[emo_map[emo]] = 1
            else:
                program[emo_map['neutral']] = 1
            programs.append(programs)
            re_emo_map.append(emo_map[emo] if emo in emo_map.keys() else emo_map['neutral'])

        return programs, torch.LongTensor(re_emo_map)


def collate_fn(data):
    def concat(sequences):
        con_seq = []
        for i, seq in enumerate(sequences):
            con_seq.append(torch.cat(seq))
        return con_seq

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def context_merge(sequences):

        lengths = [[len(seq) for seq in ss] for ss in sequences]
        max_len = max([max(s) for s in lengths])

        padded_seqs = []
        for i, seq in enumerate(sequences):
            padded_seq = torch.ones(len(seq), max_len).long()
            for j, ss in enumerate(seq):
                ## padding index 1
                end = lengths[i][j]
                padded_seq[j, :end] = ss[:end]

            padded_seqs.append(padded_seq)

        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = context_merge(item_info["context"])
    mask_input, mask_input_lengths = context_merge(item_info["context_mask"])
    context_batch, context_lengths = merge(concat(item_info["context"]))
    mask_context, mask_context_lengths = merge(concat(item_info["context_mask"]))

    emotion_batch, emotion_lengths = context_merge(item_info["emotion_context"])

    ## Target
    target_batch, target_lengths = merge(item_info["target"])

    input_batch = [data.to(config.device) for data in input_batch]
    mask_input = [data.to(config.device) for data in mask_input]
    target_batch = target_batch.to(config.device)

    context_batch = context_batch.to(config.device)
    mask_context = mask_context.to(config.device)

    d = {}

    d["input_batch"] = input_batch
    d["input_lengths"] = input_lengths
    d["mask_input"] = mask_input
    # dia_history
    d["context_batch"] = context_batch
    d["context_lengths"] = context_lengths
    d["mask_context"] = mask_context

    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["emotion_context_batch"] = [data.to(config.device) for data in emotion_batch]

    ##program
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]
    ## dia_emotion
    d["dia_program"] = item_info["dia_emotion"]
    d["dia_emotion_label"] = item_info["dia_emotion_label"]

    ##text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["dia_emotion_txt"] = item_info["dia_emotion_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
    for r in relations:
        pad_batch, _ = context_merge(item_info[r])
        pad_batch = [data.to(config.device) for data in pad_batch]
        d[r] = pad_batch
        d[f"{r}_txt"] = item_info[f"{r}_txt"]

    return d


def prepare_data_seq(batch_size=32):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map),
    )