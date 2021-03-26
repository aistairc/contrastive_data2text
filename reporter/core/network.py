import itertools
from collections import OrderedDict
from typing import List, Tuple, Dict

import torch
from torch import Tensor, nn
from torchtext.data import Batch

from reporter.util.config import Config
from reporter.util.constant import (
    GENERATION_LIMIT,
    N_LONG_TERM,
    N_SHORT_TERM,
    TIMESLOT_SIZE,
    Phase,
    SeqType
)
from reporter.util.conversion import stringify_ric_seqtype
from reporter.util.tool import log1mexp


class Encoder(nn.Module):
    def __init__(self, config: Config, device: torch.device):

        super(Encoder, self).__init__()
        self.used_seqtypes = [SeqType.NormMovRefLong,
                              SeqType.NormMovRefShort,
                              SeqType.StdLong,
                              SeqType.StdShort] \
            if config.use_standardization \
            else [SeqType.NormMovRefLong,
                  SeqType.NormMovRefShort]
        self.used_rics = config.rics
        self.use_extra_rics = len(self.used_rics) > 1
        self.base_ric = config.base_ric
        self.extra_rics = [ric for ric in self.used_rics if ric != self.base_ric]
        self.base_ric_hidden_size = config.base_ric_hidden_size
        self.ric_hidden_size = config.ric_hidden_size
        self.hidden_size = config.enc_hidden_size
        self.n_layers = config.enc_n_layers
        self.prior_encoding = int(self.base_ric in self.used_rics)
        self.dropout = config.use_dropout
        self.device = device

        self.use_dropout = config.use_dropout
        self.ric_seqtype_to_mlp = nn.ModuleDict()

        for (ric, seqtype) in itertools.product(self.used_rics, self.used_seqtypes):
            input_size = N_LONG_TERM \
                if seqtype.value.endswith('long') \
                else N_SHORT_TERM
            output_size = self.base_ric_hidden_size \
                if ric == self.base_ric \
                else self.ric_hidden_size
            mlp = MLP(input_size,
                      self.hidden_size,
                      output_size,
                      n_layers=self.n_layers).to(self.device)
            ric_seqtype_key = self.__get_ric_seqtype_key(ric, seqtype)
            self.ric_seqtype_to_mlp[ric_seqtype_key] = mlp

        lengths = [N_LONG_TERM if seqtype.value.endswith('long') else N_SHORT_TERM
                   for (_, seqtype) in itertools.product(self.used_rics, self.used_seqtypes)]
        total_length = sum(lengths)
        self.cat_hidden_size = \
            total_length + self.prior_encoding * len(self.used_seqtypes) * self.base_ric_hidden_size \
            if len(self.used_rics) == 1 \
            else self.prior_encoding * len(self.used_seqtypes) * self.base_ric_hidden_size + \
            (len(lengths) - self.prior_encoding * len(self.used_seqtypes)) * self.ric_hidden_size

        self.dense = nn.Linear(self.cat_hidden_size, self.hidden_size)

        if self.use_dropout:
            self.drop = nn.Dropout(p=0.30)

    @staticmethod
    def __get_ric_seqtype_key(ric: str, seqtype: SeqType) -> str:
        if ric[0] == '.':
            return ric[1:] + '___' + seqtype.value
        else:
            return ric + '___' + seqtype.value

    def forward(self,
                batch: Batch,
                mini_batch_size: int) -> Tuple[Tensor, Tensor]:

        L = OrderedDict()  # low-level representation
        H = OrderedDict()  # high-level representation

        attn_vector = []

        for (ric, seqtype) in itertools.product(self.used_rics, self.used_seqtypes):

            vals = getattr(batch, stringify_ric_seqtype(ric, seqtype)).to(self.device)
            ric_seqtype_key = self.__get_ric_seqtype_key(ric, seqtype)

            if seqtype in [SeqType.NormMovRefLong, SeqType.NormMovRefShort]:
                # Switch the source to one which is not normalized
                # to make our implementation compatible with Murakami 2017
                L_seqtype = SeqType.MovRefLong \
                    if seqtype == SeqType.NormMovRefLong \
                    else SeqType.MovRefShort
                L[(ric, seqtype)] = getattr(batch, stringify_ric_seqtype(ric, L_seqtype)).to(self.device)
                H[(ric, seqtype)] = self.ric_seqtype_to_mlp[ric_seqtype_key](vals)
            else:
                L[(ric, seqtype)] = vals
                H[(ric, seqtype)] = self.ric_seqtype_to_mlp[ric_seqtype_key](L[(ric, seqtype)])

        for ric in self.extra_rics:
            attn_vector.extend([H[(ric, seq)] for seq in self.used_seqtypes])

        enc_hidden = torch.cat(list(H.values()), 1) \
            if self.use_extra_rics \
            else torch.cat(list(L.values()) + list(H.values()), 1)  # Murakami model

        enc_hidden = self.dense(enc_hidden)

        if self.use_dropout:
            enc_hidden = self.drop(enc_hidden)

        if len(attn_vector) > 0:
            attn_vector = torch.cat(attn_vector, 1)
            attn_vector = attn_vector.view(mini_batch_size, len(self.extra_rics), -1)

        return (enc_hidden, attn_vector)


class Decoder(nn.Module):
    def __init__(self,
                 config: Config,
                 output_vocab_size: int,
                 device: torch.device):

        super(Decoder, self).__init__()

        self.device = device
        self.dec_hidden_size = config.dec_hidden_size
        self.word_embed_size = config.word_embed_size
        self.time_embed_size = config.time_embed_size

        self.word_embed_layer = nn.Embedding(output_vocab_size, self.word_embed_size, padding_idx=0)
        self.time_embed_layer = nn.Embedding(TIMESLOT_SIZE, self.time_embed_size)
        self.output_layer = nn.Linear(self.dec_hidden_size, output_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dec_hidden_size = self.dec_hidden_size
        self.input_hidden_size = self.time_embed_size + self.word_embed_size
        self.recurrent_layer = nn.LSTMCell(self.input_hidden_size, self.dec_hidden_size)

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        zeros = torch.zeros(batch_size, self.dec_hidden_size, device=self.device)
        self.h_n = zeros
        self.c_n = zeros
        return (self.h_n, self.c_n)

    def forward(self,
                word: Tensor,
                time: Tensor,
                seq_ric_tensor: Tensor,
                batch_size: int) -> Tuple[Tensor, Tensor]:

        word_embed = self.word_embed_layer(word).view(batch_size, self.word_embed_size)
        time_embed = self.time_embed_layer(time).view(batch_size, self.time_embed_size)
        stream = torch.cat((word_embed, time_embed), 1)
        self.h_n, self.c_n = self.recurrent_layer(stream, (self.h_n, self.c_n))
        hidden = self.h_n

        output = self.softmax(self.output_layer(hidden))
        return output


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 mid_size: int,
                 output_size: int,
                 n_layers: int = 3,
                 activation_function: str = 'tanh'):
        '''Multi-Layer Perceptron
        '''

        super(MLP, self).__init__()
        self.n_layers = n_layers

        assert(n_layers >= 1)

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise NotImplementedError

        self.MLP = nn.ModuleList()
        if n_layers == 1:
            self.MLP.append(nn.Linear(input_size, output_size))
        else:
            self.MLP.append(nn.Linear(input_size, mid_size))
            for _ in range(n_layers - 2):
                self.MLP.append(nn.Linear(mid_size, mid_size))
            self.MLP.append(nn.Linear(mid_size, output_size))

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i in range(self.n_layers):
            out = self.MLP[i](out)
            out = self.activation_function(out)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 device: torch.device):
        super(EncoderDecoder, self).__init__()

        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    def get_word_probs(self, token, decoder_output, pad_index):
        bs = decoder_output.size(0)  # batch size
        mask = torch.zeros(decoder_output.size(), device=self.device).scatter_(1, token.unsqueeze(1), 1).bool()
        word_probs = decoder_output.masked_select(mask)
        paddings = torch.zeros(bs).bool()
        paddings[token == pad_index] = True
        word_probs[paddings] = 0.0
        return word_probs

    def forward(self,
                batch: Batch,
                mini_batch_size: int,
                tokens: Tensor,
                time_embedding: Tensor,
                criterion: nn.NLLLoss,
                phase: Phase,
                ) -> Tuple[nn.NLLLoss, Tensor, Tensor]:

        loss = torch.torch.zeros(mini_batch_size, device=self.device)
        n_tokens, _ = tokens.size()
        time_embedding = time_embedding.squeeze()

        pred = []

        self.decoder.init_hidden(mini_batch_size)
        self.decoder.h_n, attn_vector = self.encoder(batch, mini_batch_size)

        if phase == Phase.Train:
            decoder_input = tokens[0]
            pred.append(decoder_input.cpu().numpy())
            for i in range(1, n_tokens):
                decoder_output = self.decoder(decoder_input, time_embedding, attn_vector, mini_batch_size)
                loss += criterion(decoder_output, tokens[i])

                topv, topi = decoder_output.data.topk(1)
                pred.append([t[0] for t in topi.cpu().numpy()])
                decoder_input = tokens[i]
        else:
            decoder_input = tokens[0]
            pred.append(decoder_input.cpu().numpy())
            for i in range(1, GENERATION_LIMIT):
                decoder_output = self.decoder(decoder_input, time_embedding, attn_vector, mini_batch_size)
                if i < n_tokens:
                    loss += criterion(decoder_output, tokens[i])

                topv, topi = decoder_output.detach().topk(1)
                pred.append([t[0] for t in topi.cpu().numpy()])
                decoder_input = topi.squeeze()

        return loss, pred

    def neg_train_sentence_loss(self,
                  batch: Batch,
                  mini_batch_size: int,
                  time_embedding: Tensor,
                  pos_tokens: Tensor,
                  neg_tokens: Tensor,
                  pad_index: int,
                  neg_criterion: nn.MultiMarginLoss
                  ) -> nn.MultiMarginLoss:

        n_tokens, _ = pos_tokens.size()
        time_embedding = time_embedding.squeeze()

        encoder_hidden, attn_vector = self.encoder(batch, mini_batch_size)

        sent_probs = torch.zeros((mini_batch_size, 2), dtype=torch.float32, device=self.device)

        # positive
        self.decoder.init_hidden(mini_batch_size)
        self.decoder.h_n = encoder_hidden
        positive_decoder_input = pos_tokens[0]
        for i in range(1, n_tokens):
            decoder_output = self.decoder(positive_decoder_input, time_embedding, attn_vector, mini_batch_size)
            word_probs = self.get_word_probs(pos_tokens[i], decoder_output, pad_index=pad_index)
            sent_probs[:, 0] += word_probs
            positive_decoder_input = pos_tokens[i]

        # negative
        self.decoder.init_hidden(mini_batch_size)
        self.decoder.h_n = encoder_hidden
        negative_decoder_input = neg_tokens[0]
        for i in range(1, n_tokens):
            decoder_output = self.decoder(negative_decoder_input, time_embedding, attn_vector, mini_batch_size)
            word_probs = self.get_word_probs(neg_tokens[i], decoder_output, pad_index=pad_index)
            sent_probs[:, 1] += word_probs
            negative_decoder_input = neg_tokens[i]
        # margin loss
        target = torch.zeros(batch.batch_size, dtype=torch.long, device=self.device)
        loss = neg_criterion(sent_probs, target)
        loss *= 2.0
        loss = loss.mean()
        return loss

    def neg_train_token_loss(self,
                             batch: Batch,
                             mini_batch_size: int,
                             time_embedding: Tensor,
                             pos_tokens: Tensor,
                             neg_tokens: List[Dict],
                             pad_index: int,
                             criterion: nn.NLLLoss,
                             neg_criterion: nn.MultiLabelMarginLoss,
                             is_unlikelihood_loss: bool = False,
                             ) -> (nn.NLLLoss, nn.MultiLabelMarginLoss):

        if is_unlikelihood_loss:
            def calc_neg_loss(pos_neg_probs):
                neg_probs = pos_neg_probs[:, 1]
                return -log1mexp(neg_probs).sum()
        else:
            def calc_neg_loss(pos_neg_probs):
                target = torch.zeros(pos_neg_probs.size(0), dtype=torch.long, device=self.device)
                loss = neg_criterion(pos_neg_probs, target)
                return loss.sum()

        loss = torch.torch.zeros(mini_batch_size, device=self.device)
        pos_neg_probs = []

        pred = []

        n_tokens, _ = pos_tokens.size()
        time_embedding = time_embedding.squeeze()

        encoder_hidden, attn_vector = self.encoder(batch, mini_batch_size)

        self.decoder.init_hidden(mini_batch_size)
        self.decoder.h_n = encoder_hidden

        decoder_input = pos_tokens[0]
        pred.append(decoder_input.cpu().numpy())
        for i in range(1, n_tokens):
            decoder_output = self.decoder(decoder_input, time_embedding, attn_vector, mini_batch_size)
            loss += criterion(decoder_output, pos_tokens[i])
            pos_word_probs = self.get_word_probs(pos_tokens[i], decoder_output, pad_index=pad_index)
            for bi, ntk in enumerate(neg_tokens):
                if i in ntk:
                    for neg in ntk[i]:
                        neg_word_prob = decoder_output[bi, neg]
                        pos_neg_probs.append(torch.stack([pos_word_probs[bi], neg_word_prob]))
            topv, topi = decoder_output.data.topk(1)
            pred.append([t[0] for t in topi.cpu().numpy()])
            decoder_input = pos_tokens[i]

        pos_neg_probs = torch.stack(pos_neg_probs)
        neg_loss = calc_neg_loss(pos_neg_probs)
        return loss, neg_loss, pred

    def neg_test_probs(self,
                       batch: Batch,
                       mini_batch_size: int,
                       time_embedding: Tensor,
                       pos_tokens: Tensor,
                       neg_tokens: Tensor,
                       pad_index: int
                       ) -> Tuple[List[float], List[float]]:
        n_tokens, _ = pos_tokens.size()
        time_embedding = time_embedding.squeeze()

        encoder_hidden, attn_vector = self.encoder(batch, mini_batch_size)

        # positive
        self.decoder.init_hidden(mini_batch_size)
        self.decoder.h_n = encoder_hidden
        positive_decoder_input = pos_tokens[0]
        positive_sent_probs = torch.zeros(mini_batch_size, dtype=torch.float32, device=self.device)
        for i in range(1, n_tokens):
            decoder_output = self.decoder(positive_decoder_input, time_embedding, attn_vector, mini_batch_size)
            word_probs = self.get_word_probs(pos_tokens[i], decoder_output, pad_index=pad_index)
            positive_sent_probs += word_probs
            positive_decoder_input = pos_tokens[i]

        # negative
        self.decoder.init_hidden(mini_batch_size)
        self.decoder.h_n = encoder_hidden
        negative_decoder_input = neg_tokens[0]
        negative_sent_probs = torch.zeros(mini_batch_size, dtype=torch.float32, device=self.device)
        for i in range(1, n_tokens):
            decoder_output = self.decoder(negative_decoder_input, time_embedding, attn_vector, mini_batch_size)
            word_probs = self.get_word_probs(neg_tokens[i], decoder_output, pad_index=pad_index)
            negative_sent_probs += word_probs
            negative_decoder_input = neg_tokens[i]

        return (positive_sent_probs.tolist(), negative_sent_probs.tolist())
