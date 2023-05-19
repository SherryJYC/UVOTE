import logging
import torch.nn as nn
import numpy as np

from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e

from fds import FDS
from loss import *

def build_model(args, vocab, pretrained_embs, tasks):
    '''
    Build model according to arguments
    '''
    d_word, n_layers_highway = args.d_word, args.n_layers_highway

    # Build embedding layers
    if args.glove:
        word_embs = pretrained_embs
        train_embs = bool(args.train_words)
    else:
        logging.info("\tLearning embeddings from scratch!")
        word_embs = None
        train_embs = True

    print(f'd_word: {d_word}, num_emb: {vocab.get_vocab_size("tokens")}, word_embs.shape: {word_embs.shape}')
    word_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=d_word, weight=word_embs, trainable=train_embs,
                              padding_index=vocab.get_token_index('@@PADDING@@'))
    d_inp_phrase = 0

    token_embedder = {"words": word_embedder}
    d_inp_phrase += d_word
    text_field_embedder = BasicTextFieldEmbedder(token_embedder)
    d_hid_phrase = args.d_hid

    # Build encoders
    phrase_layer = s2s_e.by_name('lstm').from_params(Params({'input_size': d_inp_phrase,
                                                             'hidden_size': d_hid_phrase,
                                                             'num_layers': args.n_layers_enc,
                                                             'bidirectional': True}))
    pair_encoder = HeadlessPairEncoder(vocab, text_field_embedder, n_layers_highway,
                                       phrase_layer, dropout=args.dropout)
    d_pair = 2 * d_hid_phrase

    if args.fds:
        _FDS = FDS(feature_dim=d_pair * 4, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                   start_update=args.start_update, start_smooth=args.start_smooth,
                   kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)

    # Build model and classifiers
    model = MultiTaskModel(args, pair_encoder, _FDS if args.fds else None)
    build_regressor(tasks, model, d_pair)

    if args.cuda >= 0:
        model = model.cuda()

    return model

def build_regressor(tasks, model, d_pair):
    '''
    Build the regressor
    '''
    for task in tasks:
        d_task =  d_pair * 4
        model.build_regressor(task, d_task)
    return

class MultiTaskModel(nn.Module):
    def __init__(self, args, pair_encoder, FDS=None):
        super(MultiTaskModel, self).__init__()
        self.args = args
        self.pair_encoder = pair_encoder

        self.FDS = FDS
        self.start_smooth = args.start_smooth

    def build_regressor(self, task, d_inp):
        mu_layer = nn.ModuleList([nn.Linear(d_inp, 1) for _ in range(self.args.num_branch)])
        logvar_layer = nn.ModuleList([nn.Linear(d_inp, 1) for _ in range(self.args.num_branch)])
        setattr(self, '%s_mu_layer' % task.name, mu_layer)
        setattr(self, '%s_logvar_layer' % task.name, logvar_layer)

    def forward(self, task=None, epoch=None, input1=None, input2=None, mask1=None, mask2=None, label=None, 
                weight_inv=None, weight_sqrt=None):
        
        # pack weights
        weights = [None, weight_inv, weight_sqrt]

        mu_layer = getattr(self, '%s_mu_layer' % task.name)
        logvar_layer = getattr(self, '%s_logvar_layer' % task.name)

        pair_emb = self.pair_encoder(input1, input2, mask1, mask2)
        pair_emb_s = pair_emb
        if self.training and self.FDS is not None:
            if epoch >= self.start_smooth:
                pair_emb_s = self.FDS.smooth(pair_emb_s, label, epoch)

        logits = []
        for i in range(self.args.num_branch):
            mu = mu_layer[i](pair_emb_s)
            logvar = logvar_layer[i](pair_emb_s)
            logits.extend([(mu, logvar)])

        out = {}
        if self.training and self.FDS is not None:
            out['embs'] = pair_emb
            out['labels'] = label

        # compute loss
        loss = None

        if epoch is None:
            # placefolder for val
            alpha = 1
        else:
            alpha = 1 - (epoch / self.args.max_epoch)**2
            alpha = max(alpha, 0)

        mus, logvars = [[] for _ in range(self.args.num_branch)], [[] for _ in range(self.args.num_branch)]
        for i in range(self.args.num_branch):
            mu, logvar = logits[i]
            weight = weights[i]

            mus[i].extend(mu.squeeze(-1).data.cpu().numpy())
            logvars[i].extend(logvar.squeeze(-1).data.cpu().numpy())

            # first branch use no weight
            if loss is None:
                if self.args.dynamic_loss:
                    loss = alpha * globals()[f"weighted_{self.args.loss}_loss"](
                        mu=mu, log_var=logvar, targets=label / torch.tensor(5.).cuda(), weights=weight
                    )
                else:
                    loss = globals()[f"weighted_{self.args.loss}_loss"](
                        mu=mu, log_var=logvar, targets=label / torch.tensor(5.).cuda(), weights=weight
                    )
                # use 1st branch result for logging
                out['logits'] = mu
                task.scorer(mu.squeeze(-1).data.cpu().numpy(), label.squeeze(-1).data.cpu().numpy())
            else:
                # other branches use weights
                if self.args.dynamic_loss:
                    loss += (1 - alpha) * globals()[f"weighted_{self.args.loss}_loss"](
                        mu=mu, log_var=logvar, targets=label / torch.tensor(5.).cuda(), weights=weight
                    )
                else:
                    loss += globals()[f"weighted_{self.args.loss}_loss"](
                        mu=mu, log_var=logvar, targets=label / torch.tensor(5.).cuda(), weights=weight
                    )
        out['mus'] = np.mean(np.array(mus), axis=-1)
        out['logvars'] = np.mean(np.array(logvars), axis=-1)
        # compute different combine by 
        stds = np.sqrt(2) * np.exp(np.array(logvars))
        
        #avg
        finalstds = np.mean(stds, axis=0)
        task.avg_scorer(np.mean(np.array(mus), axis=0), label.squeeze(-1).data.cpu().numpy(), finalstds)

        #avgvar
        w_var = np.power(np.exp(np.array(logvars)), -1) # 1/var as w
        finalpreds = np.sum(np.array(mus)*w_var, axis=0)/np.sum(w_var, axis=0).squeeze()
        finalstds = np.sum(stds*w_var, axis=0)/np.sum(w_var, axis=0).squeeze()
        task.avgvar_scorer(finalpreds, label.squeeze(-1).data.cpu().numpy(), finalstds)

        #minvar
        idxs = np.expand_dims(np.argmin(np.array(logvars), axis=0), axis=0)
        finalpreds = np.take_along_axis(np.array(mus), idxs, axis=0).reshape((-1, ))
        finalstds = np.take_along_axis(stds, idxs, axis=0).reshape((-1, ))
        task.minvar_scorer(finalpreds, label.squeeze(-1).data.cpu().numpy(), finalstds)

        #oracle
        se = np.power(np.array(mus) - label.squeeze(-1).data.cpu().numpy(), 2)
        idxs = np.expand_dims(np.argmin(se, axis=0), axis=0)
        finalstds = np.take_along_axis(stds, idxs, axis=0).reshape((-1, ))
        finalpreds = np.take_along_axis(np.array(mus), idxs, axis=0).reshape((-1, ))
        task.oracle_scorer(finalpreds, label.squeeze(-1).data.cpu().numpy(), finalstds)

        out['loss'] = loss

        return out

class HeadlessPairEncoder(Model):
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 dropout=0.2, mask_lstms=True, initializer=InitializerApplicator()):
        super(HeadlessPairEncoder, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        d_emb = text_field_embedder.get_output_dim()
        self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))

        self._phrase_layer = phrase_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.output_dim = phrase_layer.get_output_dim()

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, s1, s2, m1=None, m2=None):
        s1_embs = self._highway_layer(self._text_field_embedder(s1) if m1 is None else s1)
        s2_embs = self._highway_layer(self._text_field_embedder(s2) if m2 is None else s2)

        s1_embs = self._dropout(s1_embs)
        s2_embs = self._dropout(s2_embs)

        # Set up masks
        s1_mask = util.get_text_field_mask(s1) if m1 is None else m1.long()
        s2_mask = util.get_text_field_mask(s2) if m2 is None else m2.long()

        s1_lstm_mask = s1_mask.float() if self._mask_lstms else None
        s2_lstm_mask = s2_mask.float() if self._mask_lstms else None

        # Sentence encodings with LSTMs
        s1_enc = self._phrase_layer(s1_embs, s1_lstm_mask)
        s2_enc = self._phrase_layer(s2_embs, s2_lstm_mask)

        s1_enc = self._dropout(s1_enc)
        s2_enc = self._dropout(s2_enc)

        # Max pooling
        s1_mask = s1_mask.unsqueeze(dim=-1)
        s2_mask = s2_mask.unsqueeze(dim=-1)
        s1_enc.data.masked_fill_(1 - s1_mask.byte().data, -float('inf'))
        s2_enc.data.masked_fill_(1 - s2_mask.byte().data, -float('inf'))
        s1_enc, _ = s1_enc.max(dim=1)
        s2_enc, _ = s2_enc.max(dim=1)

        return torch.cat([s1_enc, s2_enc, torch.abs(s1_enc - s2_enc), s1_enc * s2_enc], 1)
