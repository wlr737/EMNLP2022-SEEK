### TAKEN FROM https://github.com/kolloldas/torchnlp
import math
import os
from collections import Counter

import numpy as np,pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from src.model.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
    argmax,
    log_sum_exp,
    MultiHeadAttention,
)
from src.utils import config
from src.utils.constants import *
from src.utils.constants import MAP_EMO

START_TAG = -2
STOP_TAG = -1


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1500,
            input_dropout=0.0,
            layer_dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            use_mask=False,
            universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                            .unsqueeze(1)
                            .repeat(1, inputs.shape[1], 1)
                            .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
            self,
            embedding_size,
            hidden_size,
            num_layers,
            num_heads,
            total_key_depth,
            total_value_depth,
            filter_size,
            max_length=1000,
            input_dropout=0.0,
            layer_dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                            .unsqueeze(1)
                            .repeat(1, inputs.shape[1], 1)
                            .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (src_mask, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
            self,
            x,
            attn_dist=None,
            enc_batch_extend_vocab=None,
            extra_zeros=None,
            temp=1,
            beam_search=False,
            attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """pred: size(L, C)
           target: size(1, L)
           mask: (1, L) if <pad> the element equals 0"""
        pred1 = pred.view(-1, config.emotion_cls)
        mask1 = mask.view(-1, 1)  # (L, 1)
        target1 = target.view(-1)  # (L,1)
        if self.weight is None:
            loss = self.loss(pred1 * mask1, target1) / torch.sum(mask1)
        else:
            loss = self.loss(pred1 * mask1, target1) / torch.sum(self.weight[target1] * mask1.squeeze(1))

        return loss


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


class GRUEncoder(nn.Module):
    def __init__(self, emb_dim, rnn_hidden_dim, sent_dim, bigru, dropout=0.3):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(emb_dim, rnn_hidden_dim, num_layers=1, bidirectional=bigru)
        self.proj = nn.Linear(2 * rnn_hidden_dim, sent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sent, sent_len):
        """
        :param sent: torch tensor, N x L x D_in
        :param sent_len: torch tensor, N
        :return:
        """
        # (N, L, D_w) -> (L, N, D_w)
        sent_embs = sent.transpose(0, 1)

        # padding
        # (L, N, D_w) -> (L, N, 2*D_h)
        sent_packed = pack_padded_sequence(sent_embs, sent_len, enforce_sorted=False)

        sent_output, h_n = self.gru(sent_packed)
        sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

        # (L, N, 2*D_h) -> (N, L, 2*D_h)
        sent_output = sent_output.transpose(0, 1)

        # # max poolingsent.size(1)
        # # (N, L, 2*D_h) -> (N, 2*D_h, L) ->
        # # (N, 2*D_h, 1) -> (N, 1, 2*D_h)
        # maxpout = F.max_pool1d(sent_output.transpose(2, 1), sent_output.size(1))
        # maxpout = maxpout.transpose(2, 1)
        #
        # # (N, 1, 2*D_h) -> (N, 1, D_s) -> (N, D_s)
        # sent_rep = self.dropout(F.relu(self.proj(maxpout)))
        # sent_rep = sent_rep.squeeze(1)
        dim1 = h_n.size()[1]
        return sent_output, h_n.transpose(0, 1).reshape(dim1, -1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 4 if config.woEMO else 5
        input_dim = input_num * config.hidden_dim
        hid_num = 2 if config.woEMO else 3
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x


class CRF(nn.Module):

    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        print("build batched crf...")

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.average_batch = False
        self.tagset_size = tagset_size
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size + 2, self.tagset_size + 2)
        # init_transitions = torch.zeros(self.tagset_size+2, self.tagset_size+2)
        self.tag_to_ix = TAG_TO_IX
        # init_transitions[:, self.tag_to_ix[START_TAG]] = -1000.0
        # init_transitions[self.tag_to_ix[STOP_TAG], :] = -1000.0
        # init_transitions[:,0] = -1000.0
        # init_transitions[0,:] = -1000.0
        # self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        # self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000
        init_transitions = init_transitions.to(config.device)
        self.transitions = nn.Parameter(init_transitions)

        # self.transitions = nn.Parameter(torch.Tensor(self.tagset_size+2, self.tagset_size+2))
        # self.transitions.data.zero_()

    def log_sum_exp(self, vec, m_size):
        """
        calculate log of exp sum
        args:
            vec (batch_size, vanishing_dim, hidden_dim) : input tensor
            m_size : hidden_dim
        return:
            batch_size, hidden_dim
        """
        _, idx = torch.max(vec, 1)  # B * 1 * M
        max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
        return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
                                                                                                                    m_size)  # B * M

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        # print feats.view(seq_len, tag_size)
        assert (tag_size == self.tagset_size + 2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size,
                                                            1)  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            cur_partition = self.log_sum_exp(cur_values, tag_size)
            # print cur_partition.data

            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size).to(config.device)

            ## effective updated partition part, only keep the partition value of mask value = 1
            mask_idx = mask_idx.long().eq(0)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
            # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size,
                                                                         tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = self.log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert (tag_size == self.tagset_size + 2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()

        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size
        partition_history.append(partition.reshape(partition.size()[0], partition.size()[1]))
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            ## forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            partition, cur_bp = torch.max(cur_values, 1)
            partition = partition.reshape(partition.size()[0], partition.size()[1])
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size).eq(0), 0)
            back_points.append(cur_bp)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1,
                                                                                                    0).contiguous()  ## (batch_size, seq_len. tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size,
                                                                                                    tag_size).expand(
            batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = Variable(torch.zeros(batch_size, tag_size)).long()
        pad_zero = pad_zero.to(config.device)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        ## select end ids in STOP_TAG
        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last
        back_points.scatter_(1, last_position, insert_last)
        # print "bp:",back_points
        # exit(0)
        back_points = back_points.transpose(1, 0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = Variable(torch.LongTensor(seq_len, batch_size))
        decode_idx = decode_idx.to(config.device)
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.data.squeeze(1)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        ## convert tag value into a new format, recorded label bigram information to index  
        new_tags = Variable(torch.LongTensor(batch_size, seq_len))
        new_tags = new_tags.to(config.device)

        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        ## transition for label to STOP_TAG

        end_transition = self.transitions[:, STOP_TAG].contiguous().view(1, tag_size).expand(batch_size,
                                                                                             tag_size)
        ## length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long().to(config.device)
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ## index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1, 0).contiguous().reshape(seq_len, batch_size, 1)

        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.reshape(seq_len, batch_size, -1), 2, new_tags).reshape(seq_len,
                                                                                               batch_size)  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        mask = mask.long().eq(0)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        # nonegative log likelihood
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        # print "batch, f:", forward_score.data[0], " g:", gold_score.data[0], " dis:", forward_score.data[0] - gold_score.data[0]
        # exit(0)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        else:
            return forward_score - gold_score


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


# the main lab which is superior the baselines
class SEEK(nn.Module):
    def __init__(
            self,
            vocab,
            decoder_number,
            model_file_path=None,
            is_eval=False,
            load_optim=False,
    ):
        super(SEEK, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.word_freq = np.zeros(self.vocab_size)

        self.is_eval = is_eval
        self.rels = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        # my component
        self.bi_gru = True
        self.num_hidd = 2 if self.bi_gru else 1
        self.gru = GRUEncoder(emb_dim=config.hidden_dim, rnn_hidden_dim=config.hidden_dim,
                              sent_dim=config.hidden_dim, bigru=self.bi_gru)
        self.sigmoid = nn.Sigmoid()
        self.kemo_lin = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.tagging = nn.Linear(self.num_hidd * config.hidden_dim, decoder_number)
        self.emo_lin = nn.Linear(config.hidden_dim, len(DIA_EMO_MAP))
        self.tag_loss = MaskedNLLLoss()
        self.softmax = nn.LogSoftmax(dim=-1)

        # tagging
        self.dia_attention = Attention(config.hidden_dim, 2 * config.hidden_dim)
        self.dia_att_lin = nn.Linear(2 * config.hidden_dim, len(DIA_EMO_MAP))
        self.attention = Attention(config.hidden_dim, 2 * config.hidden_dim)
        self.tagset_size = len(TAG_TO_IX)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.num_hidd * config.hidden_dim, self.tagset_size + 2)

        self.attention_size = config.hidden_dim
        self.w_omega = Variable(torch.zeros(self.num_hidd * config.hidden_dim, self.attention_size)).to(config.device)
        self.u_omega = Variable(torch.zeros(self.attention_size)).to(config.device)
        self.emotion_embedding = nn.Linear(decoder_number, config.emb_dim)

        # for decoder layer
        self.emo_know_layer = DecoderLayer(
            hidden_size=config.hidden_dim,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            num_heads=config.heads,
            bias_mask=_gen_bias_mask(1000)
        )
        self.mask = _get_attn_subsequent_mask(1000)

        self.emo_know = MultiHeadAttention(
            input_depth=config.hidden_dim,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            output_depth=config.hidden_dim,
            num_heads=config.heads,
        )
        # self.atten= nn.MultiheadAttention()
        self.encoder = self.make_encoder(config.emb_dim)
        self.know_encoder = self.make_encoder(config.emb_dim)
        self.layer_norm = LayerNorm(config.emb_dim)
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        if not config.woDiv:
            self.criterion.weight = torch.ones(self.vocab_size)
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def mean_pooling(self, encoder_outputs, mask_src):

        mask = mask_src.eq(False).float()

        encode = mask * encoder_outputs  # 把padd部分置0
        summ = encode.sum(dim=-2)  # 在第一维L上求和b,1
        lenth = mask.sum(dim=-2) + 1e-30  # 求出有多长 B,1
        en = summ / lenth
        return en

    def make_encoder(self, emb_dim):
        return Encoder(
            emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "SEEK_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if config.EOS_idx in pred:
                ind = pred.index(config.EOS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == config.SOS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != config.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(config.device)

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        sequence_length = lstm_output.size()[1]
        lstm_output = lstm_output.transpose(0, 1)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, config.hidden_dim * 2])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, batch):
        ## Encode the context (Semantic Knowledge)
        input_lengths = [len(s) for s in batch["input_lengths"]]
        ctx_label = [l[:-1] for l in batch["program_label"]]
        trg_label = [l[-1].unsqueeze(0) for l in batch["program_label"]]
        trg_label = torch.cat(trg_label, dim=0).to(config.device)
        # Encoder utterance
        enc_batch = torch.cat(batch["input_batch"], dim=0)
        mask_emb = self.embedding(torch.cat(batch["mask_input"], dim=0))
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        src_emb = self.embedding(enc_batch) + mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)  # batch_size * seq_len * 300
        split_enc_outputs = torch.split(enc_outputs, input_lengths)
        enc_outputs = pad_sequence(split_enc_outputs, padding_value=1, batch_first=True).to(config.device)
        src_mask = pad_sequence(torch.split(src_mask, input_lengths), padding_value=True, batch_first=True).to(
            config.device)
        # Encoder dia_history

        # Knowledge change here
        know = torch.cat([pad_sequence(batch[r], padding_value=1, batch_first=True) for r in self.rels], -1)
        know_mask = know.data.eq(config.PAD_idx)
        know_mask = know_mask.reshape(know_mask.size()[0], know_mask.size()[1] * know_mask.size()[2]).unsqueeze(1)
        # Batch_size * max_dia_context_length * five knowledge concat tensor
        know_embs = self.embedding(know).to(config.device)
        # Batch_size * max_dia_length * max_seq_length* emb_dim
        know_embs = know_embs.reshape(know_embs.size()[0], know_embs.size()[1] * know_embs.size()[2], -1)
        know_enc_outputs = self.encoder(know_embs, know_mask)  # batch_size * seq_len * 300

        d0, d1, d3 = know_enc_outputs.size()
        d2 = int(d1 / enc_outputs.size()[1])
        d1 = enc_outputs.size()[1]
        know_enc_outputs = know_enc_outputs.reshape(d0, d1, d2, d3)
        know_mean = self.mean_pooling(know_enc_outputs, know_enc_outputs.data.eq(config.PAD_idx))
        lstm_input = enc_outputs[:, :, 0, :]
        lstm_input = self.kemo_lin(torch.cat((lstm_input,know_mean),dim = -1))
        lstm_output, lstm_h_n = self.gru(lstm_input, sent_len=input_lengths)
        lstm_mask = torch.zeros(lstm_output.size()[0], lstm_output.size()[1]).to(config.device)
        for idx, leng in enumerate(input_lengths):
            lstm_mask[idx, :leng] = torch.Tensor([1] * leng)
        a = self.attention(lstm_h_n, lstm_output, lstm_mask).unsqueeze(1)
        # a = [batch size, 1,src len]
        atten_output = torch.bmm(a, lstm_output).squeeze(1)

        cat_emo_label = torch.cat(ctx_label, dim=0)
        split_emo_labels = torch.split(cat_emo_label, input_lengths)
        # utt_emo_labels = pad_sequence(split_emo_labels, padding_value=0, batch_first=True).to(config.device)

        utt_emo_logits = self.tagging(lstm_output)
        emo_pred = self.tagging(atten_output)

        context_emo_logits = torch.bmm(self.dia_attention(lstm_h_n, lstm_output, lstm_mask).unsqueeze(1),
                                       lstm_output).squeeze(1)
        context_emo_logits = self.dia_att_lin(context_emo_logits)
        dia_emo_label = torch.LongTensor(batch["dia_emotion_label"]).to(config.device)
        context_emo_loss = nn.CrossEntropyLoss()(context_emo_logits, dia_emo_label).to(config.device)

        # calc cls_loss

        utt_emo_labels = pad_sequence(split_emo_labels, padding_value=-1, batch_first=True).to(config.device)
        dia_mask = torch.ones(utt_emo_labels.size()).long().to(config.device)
        dia_mask[utt_emo_labels.data.eq(-1)] = 0
        utt_emo_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)(
            utt_emo_logits.reshape(utt_emo_logits.size()[0] * utt_emo_logits.size()[1], -1),
            utt_emo_labels.reshape(-1))
        # utt_emo_loss=0
        trg_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)(emo_pred.squeeze(1), trg_label)

        # emotion and knowledge

        emo_emb = self.emotion_embedding(emo_pred).to(config.device)
        if config.woEMO:
            atten_outputs, attetion_weights = self.emo_know(know_embs, know_embs, know_embs, None)

            know_outputs = self.mean_pooling(atten_outputs, atten_outputs.data.eq(config.PAD_idx))
            sos_emb = know_outputs
        else:
            if not config.woKnow:

                ctx_emo = self.emotion_embedding(utt_emo_logits).to(config.device)
                atten_outputs, attetion_weights = self.emo_know(ctx_emo, know_embs, know_embs, None)
                atten_outputs = self.mean_pooling(atten_outputs, atten_outputs.data.eq(config.PAD_idx)).unsqueeze(1)
                sos_emb = torch.cat((emo_emb.unsqueeze(1), atten_outputs), dim=1)
                sos_emb = self.mean_pooling(sos_emb, sos_emb.data.eq(config.PAD_idx))
            else:
                sos_emb = emo_emb

        return src_mask, enc_outputs, (context_emo_loss, utt_emo_loss, trg_loss), (
            utt_emo_logits, utt_emo_labels, emo_pred, trg_label, context_emo_logits,
            dia_emo_label), dia_mask, sos_emb


    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        src_mask, enc_outputs, cls_loss, logits_and_labels, dia_mask, sos_emb = self.forward(batch)

        (context_emo_loss, cls_loss, trg_loss) = cls_loss
        (utt_emo_logits, utt_emo_labels, trg_emo_logits, trg_label, context_emo_logits, dia_emo_label) = \
            logits_and_labels
        pred_program = np.argmax(context_emo_logits.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["dia_emotion_label"], pred_program)
        utt_pred_emotion = np.argmax(utt_emo_logits.detach().cpu().numpy(), axis=2).flatten()
        utt_emotion_acc = accuracy_score(utt_emo_labels.cpu().numpy().flatten(), utt_pred_emotion,
                                         sample_weight=dia_mask.cpu().numpy().flatten())

        trg_pred_emotion = np.argmax(trg_emo_logits.detach().cpu().numpy(), axis=1).flatten()
        trg_emotion_acc = accuracy_score(trg_label.cpu().numpy(), trg_pred_emotion)

        accs = (program_acc, utt_emotion_acc, trg_emotion_acc)
        if not train:
            confusion_matrix(trg_label.cpu().numpy(), trg_pred_emotion)

        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * dec_batch.size()[0])
                .unsqueeze(1)
                .to(config.device)
        )
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        # batch_size * seq_len * 300 (GloVe)
        dec_emb = self.embedding(dec_batch_shift)
        dec_emb[:, 0] = sos_emb

        dim0_enc, dim1_enc, dim2_enc = enc_outputs.size()[0], enc_outputs.size()[1], enc_outputs.size()[2]

        enc_outputs = enc_outputs.reshape(dim0_enc, dim1_enc * dim2_enc, -1)
        src_mask = src_mask.reshape(dim0_enc, 1, -1)
        # enc_outputs = enc_outputs.transpose(1, 2).reshape(dim0_enc, dim1_enc * dim2_enc, -1)
        # src_mask = src_mask.transpose(1, 2).reshape(dim0_enc, 1, -1)
        pre_logit, attn_dist = self.decoder(dec_emb, enc_outputs, (src_mask, mask_trg))
        # if config.aggFea:
        #     emo_emb_ = emo_emb.repeat([1, pre_logit.size()[1], 1])
        #     # pre_logit = pre_logit * emo_emb_
        #     know_outputs_ = know_outputs.repeat([1, pre_logit.size()[1], 1])
        #     emo_know_cat = torch.cat((emo_emb_, know_outputs_), dim=-1)
        #     g = self.sigmoid(self.layer_norm(self.feat_agg(emo_know_cat)))
        #     pre_logit = pre_logit + g * emo_emb_ + (1 - g) * know_outputs_

        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )
        # Loss acc
        gen_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )

        # print results for testing
        top_preds = ""
        comet_res = {}

        if self.is_eval:
            top_preds = trg_emo_logits.detach().cpu().numpy().argsort()[0][-3:][::-1]
            top_preds = f"{', '.join([MAP_EMO[pred.item()] for pred in top_preds])}"

            for r in self.rels:
                txt = [[" ".join(t) for t in tm] for tm in batch[f"{r}_txt"][0]][0]
                comet_res[r] = txt

        alpha = 0.5
        cls_loss += context_emo_loss
        emo_loss = alpha * cls_loss + (1 - alpha) * trg_loss
        if not (config.woDiv):
            _, preds = logit.max(dim=-1)
            preds = self.clean_preds(preds)
            self.update_frequency(preds)
            self.criterion.weight = self.calc_weight()
            not_pad = dec_batch.ne(config.PAD_idx)
            target_tokens = not_pad.long().sum().item()
            div_loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )
            div_loss /= target_tokens
            loss = emo_loss + 1.5 * div_loss + gen_loss
            c_cls = 1.5 * div_loss + gen_loss
        else:
            loss = emo_loss + gen_loss

        # emo_loss = cls_loss
        # loss = emo_loss + gen_loss

        if train:
            loss.backward()
            self.optimizer.step()
        return gen_loss.item(), gen_loss.item(), emo_loss.item(), accs, top_preds, comet_res
        # return c_cls.item(), math.exp(
        #     min(gen_loss.item(), 100)), emo_loss.item(), accs, top_preds, comet_res

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            _,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, enc_outputs, cls_loss, logits_and_labels, dia_mask, sos_emb = \
            self.forward(batch)


        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        dim0_enc, dim1_enc, dim2_enc = enc_outputs.size()[0], enc_outputs.size()[1], enc_outputs.size()[2]
        ctx_output = enc_outputs.reshape(dim0_enc, dim1_enc * dim2_enc, -1)
        src_mask = src_mask.reshape(dim0_enc, 1, -1)
        # ctx_output = enc_outputs
        ys_embed = sos_emb.unsqueeze(0)
        for i in range(max_dec_step + 1):
            if i != 0:
                ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(ys_embed, ctx_output, (src_mask, mask_trg))

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent
