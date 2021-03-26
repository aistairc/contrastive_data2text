from typing import Any, Dict, List

import numpy as np
import torch
from torchtext.data import Field
from torchtext.data.batch import Batch


class NegativeSampling:
    def __init__(self,
                 token_field: Field,
                 neg_enabled: bool,
                 neg_loss_type: str,
                 neg_rules: Dict[str, Any],
                 neg_rule_tag_classes: List[str],
                 neg_samples_ratio: float,
                 neg_loss_ratio: float):

        self.neg_enabled = neg_enabled
        self.neg_loss_type = neg_loss_type
        self.w2i = token_field.vocab.stoi
        self.i2w = token_field.vocab.itos
        self.wLSP = '\u3000'
        self.wSSP = ' '
        self.iLSP = token_field.vocab.stoi[self.wLSP]
        self.iSSP = token_field.vocab.stoi[self.wSSP]
        self.pad_index = token_field.vocab.stoi[token_field.pad_token]
        self.rules = neg_rules
        self.rule_tag_classes = neg_rule_tag_classes
        self.samples_ratio = neg_samples_ratio
        self.neg_loss_ratio = neg_loss_ratio

    def create_negative_sample(self, i_sentence, article_id, random_choice=True):
        pn_tokens = []
        for idx, i_token in enumerate(i_sentence):
            token = self.i2w[i_token]
            for rule_tags in self.rule_tag_classes:
                for k, vs in self.rules[rule_tags].items():
                    if token == k:
                        pn_tokens += [(article_id, rule_tags, idx, token, v) for v in vs]
        pn_tokens = sorted(set(pn_tokens), key=lambda x: (x[2], x[3], x[4]))  # sort by only (idx, token, v)
        if random_choice:
            if pn_tokens:
                np.random.shuffle(pn_tokens)
                sampled_pn_tokens = pn_tokens[0]
                return sampled_pn_tokens
            else:
                return None
        else:
            return pn_tokens  # get all negative samples

    def get_negative_train_tokens(self, batch_tokens, article_ids):
        if self.neg_loss_type == 'sentence':
            return self.__get_negative_train_tokens_sentence_loss(batch_tokens, article_ids)
        elif self.neg_loss_type == 'token' or self.neg_loss_type == 'unlikelihood':
            return self.__get_negative_train_tokens_token_loss(batch_tokens, article_ids)
        else:
            raise NotImplementedError

    def __get_negative_train_tokens_sentence_loss(self, batch_tokens, article_ids):
        neg_train_tokens = torch.zeros(batch_tokens.size(), device=batch_tokens.device).long()
        neg_info_tuples = []
        neg_items = []
        for i in range(batch_tokens.size(1)):
            i_sent = batch_tokens[:, i]
            t = self.create_negative_sample(i_sent, article_ids[i])
            neg_info_tuples += [t]
            if t:
                neg_items += [i]
                _, _, neg_idx, _, neg_token = t
                neg_train_tokens[:, i] = batch_tokens[:, i]
                neg_train_tokens[neg_idx, i] = self.w2i[neg_token]
            else:
                neg_train_tokens[:, i] = batch_tokens[:, i]
        batch_size = neg_train_tokens.size(1)
        n_samples = int(self.samples_ratio * batch_size)
        sample_items = np.random.permutation(neg_items)  # decrease negative batch-size according to the samples_ratio
        sample_items = np.sort(sample_items[:n_samples])
        neg_train_info_tuples = [neg_info_tuples[i] for i in sample_items]
        neg_train_tokens = neg_train_tokens[:, sample_items]
        return neg_train_tokens, neg_train_info_tuples, sample_items

    def __get_negative_train_tokens_token_loss(self, batch_tokens, article_ids):
        neg_info_tuples = []
        neg_train_tokens = []
        for i in range(batch_tokens.size(1)):
            i_sent = batch_tokens[:, i]
            t = self.create_negative_sample(i_sent, article_ids[i], random_choice=False)
            neg_info_tuples += [t]
            negs = {}
            if t:
                for neg_i in t:
                    _, _, neg_idx, _, neg_token = neg_i
                    if neg_idx in negs:
                        negs[neg_idx].append(self.w2i[neg_token])
                    else:
                        negs[neg_idx] = [self.w2i[neg_token]]
            neg_train_tokens.append(negs)
        sample_items = np.arange(0, batch_tokens.size(1))  # all
        return neg_train_tokens, neg_info_tuples, sample_items

    @staticmethod
    def create_negative_train_batch_sentence_loss(lm_batch, neg_train_tokens, sample_items):
        neg_batch = Batch()
        neg_batch.batch_size = len(sample_items)
        neg_batch.dataset = lm_batch.dataset
        neg_batch.fields = lm_batch.fields

        for field in lm_batch.fields:
            v = getattr(lm_batch, field)
            if field == 'article_id':
                # type is List
                varr = np.array(v)
                neg_article_id = varr[sample_items].tolist()
                setattr(neg_batch, field, neg_article_id)
            elif field == 'token':
                setattr(neg_batch, field, neg_train_tokens)
            else:
                setattr(neg_batch, field, v[sample_items])
        return neg_batch

    @staticmethod
    def __fix_neg_batch(pos_tokens, batches, batch_cta, device):
        setattr(batches[batch_cta], 'token', batches[batch_cta].token.transpose(1, 0))
        pos_tokens[batch_cta] = torch.tensor(pos_tokens[batch_cta], device=device).long()
        pos_tokens[batch_cta] = pos_tokens[batch_cta].transpose(1, 0)
        setattr(batches[batch_cta], 'batch_size', len(batches[batch_cta].article_id))

    def get_negative_test_batches(self, lm_batch):
        batches = []
        pos_tokens = []
        batch_cta = 0
        sample_cta = 0

        for i in range(lm_batch.token.size(1)):
            article_id = lm_batch.article_id[i]
            tokens = lm_batch.token[:, i]
            pn_tokens = self.create_negative_sample(tokens, article_id, random_choice=False)
            num_pn_tokens = len(pn_tokens)

            if num_pn_tokens:
                if sample_cta == 0:
                    batches.append(Batch())
                    pos_tokens.append([])
                    batches[batch_cta].dataset = lm_batch.dataset
                    batches[batch_cta].fields = lm_batch.fields

                neg_tokens = []
                for t in pn_tokens:
                    _, _, neg_idx, _, neg_token = t
                    v = tokens.tolist()
                    v[neg_idx] = self.w2i[neg_token]
                    neg_tokens.append(v)
                neg_tokens = torch.tensor(neg_tokens, device=lm_batch.token.device).long()

                for field in lm_batch.fields:
                    if field == 'article_id':
                        # the type of article_id is List (while the others are Tensor)
                        v = [article_id] * num_pn_tokens
                        if sample_cta:
                            setattr(batches[batch_cta], field,
                                    getattr(batches[batch_cta], field) + v)
                        else:
                            setattr(batches[batch_cta], field, v)
                    elif field == 'token':
                        # the batch dimension of token is NOW 0
                        if sample_cta:
                            setattr(batches[batch_cta], field,
                                    torch.cat((getattr(batches[batch_cta], field), neg_tokens), 0))
                        else:
                            setattr(batches[batch_cta], field, neg_tokens)
                    else:
                        # the batch dimension is 0
                        v = getattr(lm_batch, field)[i]
                        v = v.unsqueeze(0)
                        if len(v.size()) == 1:
                            v = v.repeat(num_pn_tokens)  # time
                        else:
                            v = v.repeat(num_pn_tokens, 1)  # the others
                        if sample_cta:
                            setattr(batches[batch_cta], field,
                                    torch.cat((getattr(batches[batch_cta], field), v), 0))
                        else:
                            setattr(batches[batch_cta], field, v)
                pos_tokens[batch_cta].extend([tokens.tolist()] * num_pn_tokens)
                sample_cta += num_pn_tokens
                if lm_batch.batch_size <= sample_cta:
                    self.__fix_neg_batch(pos_tokens, batches, batch_cta, device=lm_batch.token.device)
                    batch_cta += 1
                    sample_cta = 0

        if 0 < sample_cta:
            self.__fix_neg_batch(pos_tokens, batches, batch_cta, device=lm_batch.token.device)

        return pos_tokens, batches

    def remove_id_positive_negative_sentences(self, gold_sents: List[List[str]]) -> List[int]:
        # remove sentences in which both positive and negative words included
        pos_neg_dups = [0] * len(gold_sents)
        for rule_tag_class in self.rule_tag_classes:
            for rule_tag in self.rules[rule_tag_class].keys():
                for i, gold in enumerate(gold_sents):
                    pos_w = [w for w in gold if w == rule_tag]
                    neg_w = [w for w in gold if w in self.rules[rule_tag_class][rule_tag]]
                    if pos_w and neg_w:
                        pos_neg_dups[i] += 1
        eval_ids = [i for i, v in enumerate(pos_neg_dups) if v == 0]
        return eval_ids

    @staticmethod
    def __get_pos_neg_label(pos: bool, neg: bool, subname: str, parname: str) -> str:
        if pos and neg:
            label = 'PN'
        elif pos:
            label = 'P'
        elif neg:
            label = 'N'
        else:
            label = '-other'
        return subname + label + '/' + parname + 'P'

    def evaluate(self, gold_sents: List[List[str]], pred_sents: List[List[str]]) -> Dict[str, Any]:
        eval_result = {}
        for rule_tag_class in self.rules.keys():
            eval_ids = self.remove_id_positive_negative_sentences(gold_sents)
            eval_result[rule_tag_class] = {}
            for rule_tag in self.rules[rule_tag_class].keys():
                eval_result[rule_tag_class][rule_tag] = \
                    {'predP/goldP': 0, 'predN/goldP': 0, 'predPN/goldP': 0, 'pred-other/goldP': 0, 'gold-count': 0,
                     'goldP/predP': 0, 'goldN/predP': 0, 'goldPN/predP': 0, 'gold-other/predP': 0, 'pred-count': 0}
                for i in eval_ids:
                    gold, pred = gold_sents[i], pred_sents[i]
                    pos_pred = rule_tag in pred
                    neg_pred = next((w for w in pred if w in self.rules[rule_tag_class][rule_tag]), None) is not None
                    if rule_tag in gold:
                        label = self.__get_pos_neg_label(pos_pred, neg_pred, 'pred', 'gold')
                        eval_result[rule_tag_class][rule_tag][label] += 1
                        eval_result[rule_tag_class][rule_tag]['gold-count'] += 1
                    pos_gold = rule_tag in gold
                    neg_gold = next((w for w in gold if w in self.rules[rule_tag_class][rule_tag]), None) is not None
                    if rule_tag in pred:
                        label = self.__get_pos_neg_label(pos_gold, neg_gold, 'gold', 'pred')
                        eval_result[rule_tag_class][rule_tag][label] += 1
                        eval_result[rule_tag_class][rule_tag]['pred-count'] += 1
        return eval_result

    @staticmethod
    def evaluate_probs_win(article_ids, gold_probs, neg_probs) -> List[bool]:
        results = {}
        for article_id, gold_prob, neg_prob in zip(article_ids, gold_probs, neg_probs):
            if article_id in results:
                results[article_id]['negs'].append(neg_prob)
            else:
                results[article_id] = {'pos': gold_prob, 'negs': [neg_prob]}

        wins = []
        for article_id in results.keys():
            gold = results[article_id]['pos']
            negs = results[article_id]['negs']
            if max(negs) <= gold:
                wins.append(True)
            else:
                wins.append(False)
        return wins
