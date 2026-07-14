"""ATLOP relation-extraction model.

Re-implemented from wzhouad/ATLOP (model.py), described in Zhou et al. 2021.
The repo's license is unspecified, so this is a clean re-implementation with the
three ATLOP ingredients:

  1. Entity representation via log-sum-exp pooling over each mention's `*`
     start-marker hidden state (get_hrt).
  2. Localized Context Pooling: for a (head, tail) pair, multiply the two
     entities' token-attention distributions to focus a context vector on the
     tokens both attend to.
  3. Grouped bilinear classifier over [head; context] and [tail; context],
     trained with Adaptive Thresholding Loss (losses.ATLoss).

The encoder itself (BERT/RoBERTa/...) is passed in, so the same model works for
any HF encoder and can be handed a tiny random encoder for CPU smoke tests.
"""

import torch
import torch.nn as nn

from .losses import ATLoss
from .long_input import process_long_input


class DocREModel(nn.Module):
    def __init__(self, config, encoder, emb_size: int = 768, block_size: int = 64,
                 num_labels: int = 97, offset: int = 1):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        assert emb_size % block_size == 0, "emb_size must be divisible by block_size"
        self.bilinear = nn.Linear(emb_size * block_size, num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        # +offset to skip the leading special token ([CLS]) when indexing markers.
        self.offset = offset

    def encode(self, input_ids, attention_mask):
        start_tokens = [self.config.cls_token_id]
        end_tokens = [self.config.sep_token_id]
        sequence_output, attention = process_long_input(
            self.encoder, input_ids, attention_mask, start_tokens, end_tokens
        )
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        """Pool per-entity embeddings (log-sum-exp over mention markers) and
        per-pair localized-context vectors. Returns (hs, rs, ts), each
        (total_pairs, hidden), concatenated across the batch in hts order."""
        offset = self.offset
        _, num_heads, _, c = attention.size()
        hss, tss, rss = [], [], []

        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for mentions in entity_pos[i]:
                if len(mentions) > 1:
                    m_emb, m_att = [], []
                    for start, _end in mentions:
                        if start + offset < c:
                            m_emb.append(sequence_output[i, start + offset])
                            m_att.append(attention[i, :, start + offset])
                    if m_emb:
                        e_emb = torch.logsumexp(torch.stack(m_emb, dim=0), dim=0)
                        e_att = torch.stack(m_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(num_heads, c).to(attention)
                else:
                    start, _end = mentions[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(num_heads, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)   # (n_ent, hidden)
            entity_atts = torch.stack(entity_atts, dim=0)   # (n_ent, heads, seq)

            # reshape(-1, 2) keeps shape (0, 2) for docs with no valid pairs.
            ht_i = torch.as_tensor(hts[i], dtype=torch.long,
                                   device=sequence_output.device).reshape(-1, 2)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            # Localized Context Pooling: element-wise product of head/tail
            # attention, averaged over heads, renormalized to a distribution.
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])   # (n_pair, heads, seq)
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)                          # (n_pair, seq)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            rs = torch.matmul(ht_att, sequence_output[i])            # (n_pair, hidden)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self, input_ids, attention_mask, entity_pos, hts, labels=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))

        # Grouped bilinear: split each vector into blocks and take outer products.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        preds = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        output = (preds,)
        if labels is not None:
            # collate_fn hands us a dense (pairs, num_labels) float tensor.
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels, dtype=torch.float)
            labels = labels.to(dtype=torch.float, device=logits.device)
            loss = self.loss_fnt(logits, labels)
            output = (loss,) + output
        return output
