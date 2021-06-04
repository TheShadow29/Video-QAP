"""
Basically, write outputs to a file
Use fairseq/vizseq to compute the score
"""

import torch
from typing import Dict
from torch import nn
from fastprogress.fastprogress import progress_bar
from pathlib import Path
import pickle
from utils.trn_utils import (
    synchronize,
    is_main_process,
    get_world_size,
    compute_avg_dict,
    move_to,
)
from vidqa_code.eval_fn_vidqap import EvalFnCap
from vidqa_code.seq_gen import SequenceGenerator
from typing import List
from vidqa_code.mdl_qa import get_tgt_keydct_from_cfg


def get_met_keys_(in_met_keys: List[str]) -> List[str]:
    out_list = []
    for k in in_met_keys:
        if k == "bleu":
            # out_list += [f"bleu@{j}" for j in range(1, 5)]
            out_list += [f"bleu@{j}" for j in range(1, 3)]
        else:
            out_list.append(k)

    return out_list


def get_enc_keydct_from_cfg(cfg) -> Dict[str, str]:
    enc_key = None
    enc_key_len = None

    if cfg.task == "vid_qa":
        if cfg.mdl.name == "lqa":
            enc_key = "question_toks"
            enc_key_len = "question_tok_len"
        elif (
            cfg.mdl.name == "mtx_qa"
            or cfg.mdl.name == "butd_qa"
            or cfg.mdl.name == "stage_qa"
            or cfg.mdl.name == "lqa_rob"
            or cfg.mdl.name == "vog_qa"
        ):
            enc_key = "src_tokens"
            enc_key_len = "src_lens"
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return {"enc_key": enc_key, "enc_key_len": enc_key_len}


class EvalB(nn.Module):
    def __init__(self, cfg, comm, device):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.device = device
        # self.in_met_keys = ["bleu", "meteor", "rouge", "cider", "bert_score"]
        self.in_met_keys = ["bert_score", "bleu", "meteor", "rouge", "cider"]
        # self.in_met_keys = ['bleu']
        self.met_keys = get_met_keys_(self.in_met_keys)
        gen_params = self.cfg.gen
        if self.cfg.mdl.use_phr_clf:
            self.word_dct = self.comm.awvoc
        else:
            self.word_dct = self.comm.qwvoc
        self.gen = SequenceGenerator(
            self.word_dct,
            beam_size=gen_params.beam,
            max_len_a=gen_params.max_len_a,
            max_len_b=gen_params.max_len_b,
            min_len=gen_params.min_len,
            normalize_scores=not gen_params.unnormalized,
            len_penalty=gen_params.lenpen,
            unk_penalty=gen_params.unkpen,
            retain_dropout=gen_params.retain_dropout,
            temperature=gen_params.temperature,
            match_source_len=gen_params.match_source_len,
            no_repeat_ngram_size=gen_params.no_repeat_ngram_size,
            cfg=cfg,
            comm=comm,
        )
        self.cap_eval = EvalFnCap(cfg, comm, self.in_met_keys)
        # self.
        self.enc_key_dct = get_enc_keydct_from_cfg(self.cfg)
        self.tgt_tok_key_dct = get_tgt_keydct_from_cfg(self.cfg)

    def decode(self, x):
        if isinstance(x[0], int):
            x = torch.tensor(x).long()
        x = self.word_dct.string(x)
        return x

    def forward_one_batch(self, mdl, inp):
        """
        Should reutrn List[Dict]
        Dict = {
           'idx_sent', 'pred_tokens'
        }
        """
        eos_index = (
            self.word_dct.bos_index
            if self.cfg.misc.use_bos_decoding
            else self.word_dct.eos_index
        )
        if isinstance(mdl, torch.nn.parallel.DistributedDataParallel):
            inp2 = mdl.module.prepare_inputs(inp)
        else:
            inp2 = mdl.prepare_inputs(inp)
        inp2.update(inp)

        gen_sent = self.gen.generate(
            [mdl],
            inp2,
            bos_token=eos_index,
            encoder_key=self.enc_key_dct["enc_key"],
            encoder_key_len=self.enc_key_dct["enc_key_len"],
        )
        tgt_tok_key = self.tgt_tok_key_dct["toks"]
        tgt_tok_len_key = self.tgt_tok_key_dct["lens"]
        net_inp = {
            "tgt_tokens": inp[tgt_tok_key].squeeze(1),
            "tgt_lengths": inp[tgt_tok_len_key].squeeze(1),
        }

        out_dict_list = [
            {
                "tgt_tokens": self.decode(tgt_tok[:tgt_len]),
                "pred_tokens": [self.decode(p["tokens"]) for p in pred],
                "pred_scores": [p["score"] for p in pred],
                "idx_sent": sent_ind,
                "idx_vid_seg": vs_ind,
                "idx_qsrl": qsrl_ind,
                "qtype": [self.comm.qwvoc.symbols[qt] for qt in qtype],
            }
            for tgt_tok, tgt_len, pred, sent_ind, vs_ind, qsrl_ind, qtype in zip(
                net_inp["tgt_tokens"].detach().cpu().tolist(),
                net_inp["tgt_lengths"].detach().cpu().tolist(),
                gen_sent,
                inp["sent_idx"].detach().cpu().tolist(),
                inp["ann_idx"].detach().cpu().tolist(),
                inp["qsrl_idx"].detach().cpu().tolist(),
                inp["question_type"].detach().cpu().tolist(),
            )
        ]

        return out_dict_list

    def forward(self, model, loss_fn, dl, dl_name, rank=0, pred_path=None, mb=None):

        fname = Path(pred_path) / f"{dl_name}_{rank}.pkl"
        model.eval()
        model.to(self.device)
        loss_keys = loss_fn.loss_keys
        val_losses = {k: [] for k in loss_keys}
        nums = []
        results = []
        for batch in progress_bar(dl, parent=mb):
            batch = move_to(batch, self.device)
            b = next(iter(batch.keys()))
            nums.append(batch[b].size(0))
            torch.cuda.empty_cache()
            with torch.no_grad():
                out = model(batch)
                out_loss = loss_fn(out, batch)

            for k in out_loss:
                val_losses[k].append(out_loss[k].detach().cpu())
            results += self.forward_one_batch(model, batch)
        pickle.dump(results, open(fname, "wb"))
        nums = torch.tensor(nums).float()
        val_loss = compute_avg_dict(val_losses, nums)

        synchronize()
        if is_main_process():
            curr_results = results
            world_size = get_world_size()
            for w in range(1, world_size):
                tmp_file = Path(pred_path) / f"{dl_name}_{w}.pkl"
                with open(tmp_file, "rb") as f:
                    tmp_results = pickle.load(f)
                curr_results += tmp_results
                tmp_file.unlink
            with open(fname, "wb") as f:
                pickle.dump(curr_results, f)
            out_acc = self.cap_eval.eval_vidqap_met(
                fname, split_type="val", do_rescaling=self.cfg.evl.do_rescale
            )
            val_acc = {
                k: torch.tensor(v).to(self.device)
                for k, v in out_acc.items()
                if k in self.met_keys
            }
            # return val_loss, val_acc
        synchronize()
        if is_main_process():
            return val_loss, val_acc
        else:
            return (
                {k: torch.tensor(0.0).to(self.device) for k in loss_keys},
                {k: torch.tensor(0.0).to(self.device) for k in self.met_keys},
            )
