import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer, BertConfig
import numpy as np
import re

class embeds(nn.Module):

    def __init__(self, model='bert-base-cased', special_tokens=False, device='gpu'):
        super(embeds, self).__init__()
        self.dev = device
        config = BertConfig.from_pretrained(model, output_hidden_states=True)
        self.sptoks = special_tokens
        self.mod = BertModel.from_pretrained(model, config=config).to(self.dev)
        self.tokenizer = BertTokenizer.from_pretrained(model)

    def just_tokenize(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=self.sptoks)
        tokens = np.array(self.tokenizer.convert_ids_to_tokens(ids))
        return ids, tokens

    def E(self, text, level=-1, clip_at=500):
        ids = self.tokenizer.encode(text, add_special_tokens=self.sptoks)
        tokens = np.array(self.tokenizer.convert_ids_to_tokens(ids))

        nSpans = int(len(ids) / clip_at)
        start = [i * clip_at for i in range(nSpans)] + [nSpans * clip_at]
        fins = [(i + 1) * clip_at for i in range(nSpans)] + [len(ids)]

        # outputs = [self.mod(torch.LongTensor(ids[s:e]).unsqueeze(0))[2][level].squeeze(0) for s, e in list(zip(start, fins))]
        steps = list(zip(start, fins))
        outputs = self.mod(torch.LongTensor(ids[steps[0][0]:steps[0][1]]).unsqueeze(0).to(self.dev))[2][level].squeeze(0)

        if len(steps) > 1:
            for step in steps[1:5]:
                outputs = torch.cat([outputs, self.mod(torch.LongTensor(ids[step[0]:step[1]]).unsqueeze(0).to(self.dev))[2][level].squeeze(0)], dim=0)

        return outputs, tokens

    def DELTA(self, w_, tokens, split_indicator='##'):

        outputs = (tokens == w_).nonzero()[0].reshape(-1,1)

        if outputs.sum() < 1:
            outputs = [[0]]
            for i,w in enumerate(tokens):
                if split_indicator in w:
                    if i > outputs[-1][-1]+1:
                        outputs.append([i-1,i])
                    else:
                        outputs[-1].append(i)

            # START VERSION: outputs = [[tok_set[0]] for tok_set in outputs if ''.join(tokens[tok_set]).replace(split_indicator, '') == w_]
            # MID VERSION:
            outputs = [[tok_set[int(len(tok_set)/2)]] for tok_set in outputs if ''.join(tokens[tok_set]).replace(split_indicator, '') == w_]
            # END VERSION: outputs = [[tok_set[-1]] for tok_set in outputs if ''.join(tokens[tok_set]).replace(split_indicator, '') == w_]
            outputs = np.array(outputs)

        return outputs

    def delta(self, w_, sent):
        W = re.compile(r'\b' + re.escape(w_) + r'\b')
        locs = [i.start() for i in re.finditer(W,sent)]
        idx = [len(self.tokenizer.encode(sent[:i], add_special_tokens=self.sptoks)) for i in locs]
        return idx

    def forward(self, w, text, split_indicator='##', level=-1):
        E,tokens = self.E(text,level=level)
        delta = self.delta(w,text)
        
        return E[torch.LongTensor(delta)].detach()

    def multi_w(self, terms, text, split_indicator='##', level=-1):

        Ewi = dict()
        for w_ in terms:
            Ewi[w_] = self.forward(w_, text, split_indicator, level)

        return Ewi






















    # def delta(self, w, tokens, splitter):
    #
    #     fixed_w = w.replace('{', '{ ').replace('}', ' }')
    #
    #     hashed = np.array([splitter in tok for tok in tokens]).nonzero()[0]
    #     words = [[i] for i in range(len(tokens)) if i not in hashed]
    #
    #     for span in words:
    #         for j in hashed:
    #             if j - span[-1] == 1:
    #                 span.append(j)
    #             else:
    #                 continue
    #
    #     W, spans = np.array([''.join(tokens[span]).replace(splitter, '') for span in words]), words
    #
    #     sel = (W == np.array(fixed_w.split()).reshape(-1, 1))
    #     down = np.concatenate([sel[:, 1:], np.array([[0] for _ in range(sel.shape[0])])], axis=-1)
    #     sel = (sel + down).sum(axis=0)
    #     sel += np.array([0] + (sel > 1).astype(int)[:-1].tolist())
    #     sel = (sel > 1).nonzero()[0].reshape(-1, len(fixed_w.split()))
    #
    #     if len(sel) < 1:
    #         sel = (W == np.array(w.split()).reshape(-1, 1)).sum(axis=0).nonzero()[0].reshape(-1, 1)
    #
    #     outputs = []
    #     for s in sel:
    #         a = sum([spans[i] for i in s], [])
    #         outputs.append(a[int(len(a) / 2)])
    #
    #     return np.array(outputs).reshape(-1, 1), {lex: sp for lex, sp in list(zip(W, spans))}