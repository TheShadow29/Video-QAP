"""
QA Models
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import argparse
from argparse import Namespace as NS
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerModel,
    EncoderOut,
    PositionalEmbedding,
    TransformerEncoderLayer,
    LayerNorm,
)
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

import random
from vidqa_code.transformer_code import Transformer as TxFormer
from typing import Dict


def gather_from_index(inp1, dim1, index1):
    index1_reshaped = index1.unsqueeze(-1).expand(*index1.shape, inp1.size(-1))
    return torch.gather(inp1, dim1, index1_reshaped)


def update_namespace(inp_ns, **kwargs):
    for name in kwargs:
        setattr(inp_ns, name, kwargs[name])


def get_encoder_out(
    encoder_out,
    encoder_padding_mask,
    encoder_embedding,
    encoder_states,
    src_tokens,
    src_lengths,
):
    return EncoderOut(
        encoder_out=encoder_out,  # T x B x C
        encoder_padding_mask=encoder_padding_mask,  # B x T
        encoder_embedding=encoder_embedding,  # B x T x C
        encoder_states=encoder_states,  # List[T x B x C]
        # src_tokens=src_tokens,  # B x T
        # src_lengths=src_lengths,  # B x 1
    )


class TxEncSimple(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, embed_dim):
        super().__init__()

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop
        self.max_source_positions = args.max_source_positions

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        assert args.no_token_positional_embeddings

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward_embedding(self, src_embed):
        """
        src_embed: B x seqlen x embed_dim
        """
        # embed tokens and positions
        x = embed = self.embed_scale * src_embed
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths=None,
        cls_input=None,
        return_all_hiddens=False,
        **unused
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # NOTE: src_tokens should already be embedded!
        assert len(src_tokens.shape) > 2

        assert "encoder_padding_mask" in unused

        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_padding_mask = unused["encoder_padding_mask"]

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return get_encoder_out(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        # return TransformerEncoder.max_positions(self)
        return self.max_source_positions

    def buffered_future_mask(self, tensor):
        return TransformerEncoder.buffered_future_mask(self, tensor)


class BaseLangTx(nn.Module):
    def build_lang_model(self):
        """
        How to encode/decode the input sentence
        """
        self.bid = self.cfg.mdl.rnn.bidirectional

        parser = argparse.ArgumentParser(allow_abbrev=False)
        xargs = parser.parse_known_args()[0]

        update_namespace(xargs, **self.cfg.mdl.lang_tx)
        if self.cfg.mdl.use_phr_clf:
            target_dct = self.comm.awvoc
        else:
            target_dct = self.comm.qwvoc
        task = NS(
            **{"source_dictionary": self.comm.qwvoc, "target_dictionary": target_dct}
        )
        lang_tx = TransformerModel.build_model(xargs, task)

        self.encoder = lang_tx.encoder
        self.decoder = lang_tx.decoder

        self.max_decoder_positions = self.decoder.max_positions
        self.get_normalized_probs = lang_tx.get_normalized_probs
        self.max_decoder_positions = lang_tx.max_decoder_positions

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def forward_decoder(self, prev_out_tokens, encoder_out, incremental_state=None):
        # try:
        decoder_out = self.decoder(
            prev_output_tokens=prev_out_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        return decoder_out

    def forward(self, inp):
        """
        Main difference is that prop feats/seg features
        have an extra dimension
        """
        encoder_out = self.forward_encoder(inp)
        prev_output_tokens = self.prepare_prev_toks(inp)
        # prev_output_tokens = inp2["prev_out_toks"]

        decoder_out = self.forward_decoder(prev_output_tokens, encoder_out)
        lprobs = F.log_softmax(decoder_out[0], dim=-1)

        return {"net_out": decoder_out, "lprobs": lprobs}


class LangQA(BaseLangTx):
    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.build_lang_model()
        # NOTE: Redefine self.encoder

    def prepare_inputs(self, inp):
        out_dict = {
            "src_tokens": inp["question_toks"].squeeze(1),
            "src_lens": inp["question_tok_len"].squeeze(1),
        }
        return out_dict

    def prepare_prev_toks(self, inp):
        if self.cfg.mdl.use_phr_clf:
            return inp["prev_out_answer_clf"].squeeze(1)
        return inp["prev_out_answer_toks"].squeeze(1)

    def forward_encoder(self, inp, prep_inp=True):
        if prep_inp:
            inp2 = self.prepare_inputs(inp)
            inp2.update(inp)
        else:
            inp2 = inp

        return self.encoder(src_tokens=inp2["src_tokens"], src_lengths=inp2["src_lens"])


class MTxVidSimple(BaseLangTx):
    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        super().__init__()
        parser = argparse.ArgumentParser(allow_abbrev=False)
        xargs = parser.parse_known_args()[0]

        update_namespace(xargs, **self.cfg.mdl.mtx_simple_enc)
        self.set_args_mdl()
        self.build_lang_model()
        self.build_embedding_models()

        # self.encoder = TxEncSimple(xargs, xargs.embed_dim)
        self.vis_lang_encoder = TxFormer(
            d_model=self.prop_encode_dim,
            n_vocab_src=0,
            vocab_trg=0,
            d_hidden=(self.prop_encode_dim) // 2,
            n_layers=2,
            n_heads=3,
            drop_ratio=0.2,
            pe=False,
        )

    def set_args_mdl(self):
        # proposal dimension
        self.prop_dim = self.cfg.mdl.prop_feat_dim
        # Encoded dimension of the region features
        self.prop_encode_dim = self.cfg.mdl.vsrl.prop_encode_size

        # Segment features (2048+1024)
        self.seg_feat_dim = self.cfg.mdl.seg_feat_dim
        # Encoded dimension of the segment features
        self.seg_feat_encode_dim = self.cfg.mdl.vsrl.seg_encode_size

        assert self.prop_encode_dim == self.seg_feat_encode_dim

    def build_embedding_models(self):
        self.prop_encoder = nn.Sequential(
            *[nn.Linear(self.prop_dim, self.prop_encode_dim), nn.ReLU()]
        )
        self.prop_pos_encoder = nn.Sequential(
            *[nn.Linear(5, self.prop_encode_dim), nn.ReLU()]
        )

        self.seg_encoder = nn.Sequential(
            *[nn.Linear(self.seg_feat_dim, self.seg_feat_encode_dim), nn.ReLU()]
        )
        self.seg_pos_encoder = PositionalEmbedding(
            num_embeddings=self.cfg.ds.num_frms_to_select,
            embedding_dim=512,
            padding_idx=0,
            learned=False,
        )

        self.lang_embedding = nn.Embedding(
            num_embeddings=len(self.comm.qwvoc),
            embedding_dim=512,
            padding_idx=self.comm.qwvoc.pad(),
        )
        self.lang_pos_encoder = PositionalEmbedding(
            num_embeddings=50,
            embedding_dim=512,
            padding_idx=self.comm.qwvoc.pad(),
            learned=False,
        )

        # Frame = 0, Box = 1, Txt = 2, Pad=3
        self.inp_type_embedding = nn.Embedding(
            num_embeddings=4, embedding_dim=512, padding_idx=3
        )

        return

    def prepare_inputs(self, inp):
        seg_emb = self.seg_encoder(inp["seg_feature_for_frms_glob"].squeeze(1))
        seg_msk = inp["seg_feature_for_frms_glob"].squeeze(1).mean(-1).ne(0)
        seg_pos = self.seg_pos_encoder(seg_msk)
        seg_emb_pos = seg_emb + seg_pos
        lang_emb = self.lang_embedding(inp["question_toks"].squeeze(1))
        lang_pos = self.lang_pos_encoder(inp["question_toks"].squeeze(1))
        lang_msk = inp["question_toks"].squeeze(1).ne(self.comm.qwvoc.pad())
        lang_emb_pos = lang_emb + lang_pos
        if self.cfg.mdl.mtx_simple_enc.add_props:
            prop_emb = self.prop_encoder(inp["pad_region_feature"].squeeze(1))
            prop_msk = inp["pad_region_feature"].squeeze(1).mean(-1).ne(0)
            inp_type = lang_emb.new_tensor(
                [0] * seg_emb.size(1) + [1] * prop_emb.size(1) + [2] * lang_emb.size(1)
            ).long()
            assert prop_emb.size(-1) == seg_emb.size(-1)
            src_emb = torch.cat([seg_emb_pos, prop_emb, lang_emb_pos], dim=1)
            encoder_padding_mask = torch.cat([seg_msk, prop_msk, lang_msk], dim=1)
        else:
            inp_type = lang_emb.new_tensor(
                [0] * seg_emb.size(1) + [2] * lang_emb.size(1)
            ).long()
            src_emb = torch.cat([seg_emb_pos, lang_emb_pos], dim=1)
            encoder_padding_mask = torch.cat([seg_msk, lang_msk], dim=1)

        inp_type_exp = inp_type.view(1, inp_type.size(-1)).expand(
            seg_emb.size(0), inp_type.size(-1)
        )

        assert seg_emb.size(-1) == lang_emb.size(-1)

        src_inp_type_emb = self.inp_type_embedding(inp_type_exp)

        src_emb_inp = src_emb + src_inp_type_emb

        out_dict = {
            "src_tokens": src_emb_inp,
            "src_lens": encoder_padding_mask.sum(dim=-1),
            "kwargs": {"encoder_padding_mask": ~encoder_padding_mask},
        }
        return out_dict

    def prepare_prev_toks(self, inp):
        if self.cfg.mdl.use_phr_clf:
            return inp["prev_out_answer_clf"].squeeze(1)
        return inp["prev_out_answer_toks"].squeeze(1)

    def forward_encoder(self, inp, prep_inp=True):
        if prep_inp:
            inp2 = self.prepare_inputs(inp)
            inp2.update(inp)
        else:
            inp2 = inp
        enc_out = self.vis_lang_encoder(inp2["src_tokens"])

        return get_encoder_out(
            encoder_out=enc_out.transpose(0, 1),
            encoder_padding_mask=None,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,  # B x T
            src_lengths=None,  # B x 1
        )


class BUTD_Simple(BaseLangTx):
    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        super().__init__()
        parser = argparse.ArgumentParser(allow_abbrev=False)
        xargs = parser.parse_known_args()[0]

        update_namespace(xargs, **self.cfg.mdl.mtx_simple_enc)
        self.set_args_mdl()
        self.build_lang_model()
        self.build_embedding_models()
        self.lang_encoder = self.encoder

        self.vis_objtx_encoder = TxFormer(
            d_model=self.prop_encode_dim,
            n_vocab_src=0,
            vocab_trg=0,
            d_hidden=int(self.prop_dim // 2),
            n_layers=2,
            n_heads=3,
            drop_ratio=0.2,
        )

    def set_args_mdl(self):
        MTxVidSimple.set_args_mdl(self)

    def build_embedding_models(self):
        self.prop_encoder = nn.Sequential(
            *[nn.Linear(self.prop_dim, self.prop_encode_dim), nn.ReLU()]
        )

        self.seg_encoder = nn.Sequential(
            *[nn.Linear(self.seg_feat_dim, self.seg_feat_encode_dim), nn.ReLU()]
        )
        self.lin_attn = nn.Sequential(
            *[
                nn.Linear(self.prop_encode_dim, self.prop_encode_dim // 2),
                nn.ReLU(),
                nn.Linear(self.prop_encode_dim // 2, 1),
            ]
        )
        self.vis_lang_enc = nn.Sequential(
            *[nn.Linear(self.prop_encode_dim, self.prop_encode_dim), nn.ReLU()]
        )

    def prepare_inputs(self, inp):
        return {
            "src_tokens": inp["question_toks"].squeeze(1),
            "src_lens": inp["question_tok_len"].squeeze(1),
        }

    def forward_encoder(self, inp, prep_inp=True):
        assert prep_inp

        seg_emb = self.seg_encoder(inp["seg_feature_for_frms_glob"].squeeze(1))
        # B x T x vdim
        if self.cfg.mdl.butd.add_props:
            prop_emb = self.prop_encoder(inp["pad_region_feature"].squeeze(1))
            vis_src_emb = torch.cat([seg_emb, prop_emb], dim=1)
        else:
            vis_src_emb = seg_emb

        if self.cfg.mdl.butd.add_obj_tx:
            # B x T x vdim -> B x T x vdim
            vis_src_emb = self.vis_objtx_encoder(vis_src_emb)
        lang_enc = self.lang_encoder(
            src_tokens=inp["question_toks"].squeeze(1),
            src_lengths=inp["question_tok_len"].squeeze(1),
        )
        # B x ldim
        lang_enc1 = lang_enc.encoder_out[0]
        assert len(lang_enc1.shape) == 2

        # ldim == vdim
        assert lang_enc1.size(-1) == vis_src_emb.size(-1)
        B, ldim = lang_enc1.shape
        lang_enc1_exp = lang_enc1.view(B, 1, ldim).expand(B, vis_src_emb.size(1), ldim)
        attn = self.lin_attn(vis_src_emb * lang_enc1_exp)
        # .mean(dim=-1)
        # B x T
        attn_norm = F.softmax(attn, dim=1)
        attn_norm_exp = attn_norm.view(B, attn_norm.size(1), 1).expand(
            B, attn_norm.size(1), ldim
        )
        # B x T x C -> B x C
        vis_src_emb_out = (vis_src_emb * attn_norm_exp).sum(dim=1)
        assert vis_src_emb_out.size(-1) == lang_enc1.size(-1)

        vis_lang_out = self.vis_lang_enc(vis_src_emb_out * lang_enc1)

        return get_encoder_out(
            encoder_out=vis_lang_out.unsqueeze(0),  # T x B x C
            encoder_padding_mask=None,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None,  # B x T
            src_lengths=None,  # B x 1
        )

    def prepare_prev_toks(self, inp):
        return inp["prev_out_answer_toks"].squeeze(1)


class VOGSimple(BaseLangTx):
    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        super().__init__()
        parser = argparse.ArgumentParser(allow_abbrev=False)
        xargs = parser.parse_known_args()[0]

        update_namespace(xargs, **self.cfg.mdl.mtx_simple_enc)
        self.set_args_mdl()
        self.build_lang_model()
        self.build_embedding_models()
        self.lang_encoder = self.encoder
        self.vis_lang_txf = TxFormer(
            d_model=self.prop_encode_dim,
            n_vocab_src=0,
            vocab_trg=0,
            d_hidden=(self.prop_encode_dim) // 2,
            n_layers=2,
            n_heads=3,
            drop_ratio=0.2,
            pe=False,
        )

    def set_args_mdl(self):
        MTxVidSimple.set_args_mdl(self)

    def build_embedding_models(self):
        self.prop_encoder = nn.Sequential(
            *[nn.Linear(self.prop_dim, self.prop_encode_dim), nn.ReLU()]
        )

        self.seg_encoder = nn.Sequential(
            *[nn.Linear(self.seg_feat_dim, self.seg_feat_encode_dim), nn.ReLU()]
        )
        self.lang_srl_encoder = nn.Sequential(
            *[nn.Linear(self.prop_encode_dim * 2, self.prop_encode_dim), nn.ReLU()]
        )

        self.vis_lang_enc = nn.Sequential(
            *[nn.Linear(self.prop_encode_dim * 2, self.prop_encode_dim), nn.ReLU()]
        )

    def prepare_inputs(self, inp):
        return {
            "src_tokens": inp["question_toks"].squeeze(1),
            "src_lens": inp["question_tok_len"].squeeze(1),
        }

    def forward_encoder(self, inp, prep_inp=True):
        assert prep_inp

        seg_emb = self.seg_encoder(inp["seg_feature_for_frms_glob"].squeeze(1))
        # B x T x vdim
        if self.cfg.mdl.vog.add_props:
            prop_emb = self.prop_encoder(inp["pad_region_feature"].squeeze(1))
            vis_src_emb = torch.cat([seg_emb, prop_emb], dim=1)
        else:
            vis_src_emb = seg_emb

        if self.cfg.mdl.vog.add_obj_tx:
            # B x T x vdim -> B x T x vdim
            vis_src_emb = self.vis_objtx_encoder(vis_src_emb)

        lang_enc = self.lang_encoder(
            src_tokens=inp["question_toks"].squeeze(1),
            src_lengths=inp["question_tok_len"].squeeze(1),
        )
        # B x T x ldim
        lang_enc1 = lang_enc.encoder_out.transpose(0, 1)
        assert len(lang_enc1.shape) == 3

        # ldim == vdim

        assert lang_enc1.size(-1) == vis_src_emb.size(-1)
        B, qwords, ldim = lang_enc1.shape
        num_srls = 5
        # B x num_srls x ldim
        lang_enc2_st = gather_from_index(
            lang_enc1, 1, inp["question_srl_bounds_idxs"].squeeze(1)[..., 0]
        )
        # B x num_srls x ldim
        lang_enc2_en = gather_from_index(
            lang_enc1, 1, inp["question_srl_bounds_idxs"].squeeze(1)[..., 1]
        )
        # B x num_srls x 2*ldim -> B x num_srls x ldim
        lang_enc2 = self.lang_srl_encoder(
            torch.cat([lang_enc2_st, lang_enc2_en], dim=2)
        )
        lang_enc2_msk = (
            inp["num_srls_used_msk"]
            .squeeze(1)
            .view(B, num_srls, 1)
            .expand(B, num_srls, ldim)
        )

        lang_enc2 = lang_enc2 * lang_enc2_msk
        nprop = vis_src_emb.size(1)
        vdim = vis_src_emb.size(2)
        lang_enc2_reshape = lang_enc2.view(B, num_srls, 1, ldim).expand(
            B, num_srls, nprop, ldim
        )
        vis_src_reshape = vis_src_emb.view(B, 1, nprop, vdim).expand(
            B, num_srls, nprop, vdim
        )
        vis_lang_conc = self.vis_lang_enc(
            torch.cat([lang_enc2_reshape, vis_src_reshape], dim=3)
        )
        assert vis_lang_conc.shape == (B, num_srls, nprop, vdim)
        vis_lang_conc_reshape = vis_lang_conc.view(B, num_srls * nprop, vdim)
        # B x num_srls*nprop x vdim
        vis_lang_out = self.vis_lang_txf(vis_lang_conc_reshape)
        return get_encoder_out(
            encoder_out=vis_lang_out.transpose(0, 1),  # T x B x C
            encoder_padding_mask=None,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
            src_tokens=None,  # B x T
            src_lengths=None,  # B x 1
        )

    def prepare_prev_toks(self, inp):
        return inp["prev_out_answer_toks"].squeeze(1)


# class LossB_Box(nn.Module):
#     def __init__(self, cfg, comm):
#         super().__init__()
#         self.cfg = cfg
#         self.comm = comm
#         self.loss_keys = ["loss"]
#         self.tgt_tok_key_dct = get_tgt_keydct_from_cfg(self.cfg)

#     def forward(self, out, inp):
#         lprobs = out["lprobs"]
#         tgt_tok_key = self.tgt_tok_key_dct["toks"]
#         tgt_tokens = inp[tgt_tok_key].squeeze(1)

#         loss = label_smoothed_nll_loss(
#             lprobs,
#             tgt_tokens.clone(),
#             epsilon=0,
#             ignore_index=self.comm.wtoi.pad_index,
#             reduce=True,
#         )

#         B = lprobs.size(0)
#         loss = [loss1 / B for loss1 in loss]

#         return {"loss": loss[0]}


def get_tgt_keydct_from_cfg(cfg) -> Dict[str, str]:
    assert cfg.task == "vid_qa"
    tgt_tok_key = "answer_toks"
    tgt_tok_len_key = "answer_tok_lens"
    if cfg.mdl.use_phr_clf:
        tgt_tok_key = "answer_clf"
        tgt_tok_len_key = "answer_clf_lens"

    return {"toks": tgt_tok_key, "lens": tgt_tok_len_key}


class LossB(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.loss_keys = ["loss"]
        self.tgt_tok_key_dct = get_tgt_keydct_from_cfg(self.cfg)

    def forward(self, out, inp):
        lprobs = out["lprobs"]
        tgt_tok_key = self.tgt_tok_key_dct["toks"]
        tgt_tokens = inp[tgt_tok_key].squeeze(1)

        loss = label_smoothed_nll_loss(
            lprobs,
            tgt_tokens.clone(),
            epsilon=0,
            ignore_index=self.comm.qwvoc.pad_index,
            reduce=True,
        )

        B = lprobs.size(0)
        loss = [loss1 / B for loss1 in loss]

        return {"loss": loss[0]}
