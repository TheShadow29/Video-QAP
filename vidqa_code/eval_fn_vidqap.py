"""
Evaluating Video-QAP
Use eval metrics from Pycocoevalcap

Can be used standalone

Requires Bertscore
"""
from pathlib import Path
import fire
from yacs.config import CfgNode as CN
import yaml
import pickle
import numpy as np
from collections import namedtuple
import json
import bert_score as bs
import time
from collections import defaultdict
from tqdm import tqdm
from typing import List

from bert_score.utils import get_bert_embedding, pad_sequence, greedy_cos_idf
import torch

import sys

sys.path.append("./coco-caption")


def remove_nonascii(text):
    return "".join([i if ord(i) < 128 else " " for i in text])


def bert_cos_score_idf(
    model,
    refs,
    hyps,
    tokenizer,
    idf_dict,
    verbose=False,
    batch_size=64,
    device="cuda:0",
    all_layers=False,
):
    """
    Compute BERTScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    preds = []

    def dedup_and_sort(l1):
        return sorted(list(set(l1)), key=lambda x: len(x.split(" ")), reverse=True)

    sentences = dedup_and_sort(refs + hyps)
    embs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print("computing bert embedding.")
        iter_range = tqdm(iter_range)
    stats_dict = dict()
    for batch_start in iter_range:
        sen_batch = sentences[batch_start : batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict, device=device, all_layers=all_layers
        )
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            stats_dict[sen] = (emb, idf)

    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(device) for e in emb]
        idf = [i.to(device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad

    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)
    if verbose:
        print("computing greedy matching.")
        iter_range = tqdm(iter_range)

    with torch.no_grad():
        for batch_start in iter_range:
            batch_refs = refs[batch_start : batch_start + batch_size]
            batch_hyps = hyps[batch_start : batch_start + batch_size]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)

            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds


class BertScoreOrig(bs.BERTScorer):
    def score(self, cands, refs, verbose=False, batch_size=64, return_hash=False):
        ref_group_boundaries = None
        if not isinstance(refs[0], str):
            ref_group_boundaries = []
            ori_cands, ori_refs = cands, refs
            cands, refs = [], []
            count = 0
            for cand, ref_group in zip(ori_cands, ori_refs):
                cands += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)

        if verbose:
            print("calculating scores...")
            start = time.perf_counter()

        if self.idf:
            assert self._idf_dict, "IDF weights are not computed"
            idf_dict = self._idf_dict
        else:
            idf_dict = defaultdict(lambda: 1.0)
            idf_dict[self._tokenizer.sep_token_id] = 0
            idf_dict[self._tokenizer.cls_token_id] = 0

        all_preds = bert_cos_score_idf(
            self._model,
            refs,
            cands,
            self._tokenizer,
            idf_dict,
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
            all_layers=self.all_layers,
        ).cpu()

        if ref_group_boundaries is not None:
            max_preds = []
            for start, end in ref_group_boundaries:
                max_preds.append(all_preds[start:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)

        if self.rescale_with_baseline:
            all_preds = (all_preds - self.baseline_vals) / (1 - self.baseline_vals)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

        if verbose:
            time_diff = time.perf_counter() - start
            print(
                f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec"
            )

        if return_hash:
            out = tuple([out, self.hash])

        return out


class BertScoreSimple:
    def __init__(
        self, lang="en", verbose=False, rescale_baseline: bool = True, idf: bool = True
    ):

        import logging

        logging.getLogger("pytorch_pretrained_bert").setLevel(logging.WARNING)
        logging.getLogger("langid").setLevel(logging.WARNING)
        self.verbose = verbose
        self.bert_score_oop = BertScoreOrig(
            lang=lang, rescale_with_baseline=rescale_baseline, nthreads=4, idf=idf
        )

    def compute_score(
        self, gts, res, force_recompute_idf: bool = False, return_prf: bool = False
    ):
        assert gts.keys() == res.keys()
        imgIds = sorted(list(gts.keys()))
        # scores = []
        assert all([len(res[i]) == 1 for i in imgIds])
        assert all(
            [len(gts[i]) == 1 for i in imgIds]
        ), "Only single references supported in bert score atm"

        hypothesis = [res[i][0] for i in imgIds]
        references = [gts[i] for i in imgIds]

        if self.bert_score_oop._idf:
            if self.bert_score_oop._idf_dict is None or force_recompute_idf:
                refs_idf = [r for ref in references for r in ref]
                self.bert_score_oop.compute_idf(sents=refs_idf)
        if return_prf:
            return self.bert_score_oop.score(
                hypothesis, references, verbose=self.verbose
            )
        sent_scores = self.bert_score_oop.score(
            hypothesis, references, verbose=self.verbose
        )[2].tolist()

        corpus_scores = np.mean(sent_scores)
        return corpus_scores, sent_scores


class EvalFnQAP:
    def __init__(self, cfg, comm, met_keys, read_val_file: bool = True):
        self.cfg = cfg
        self.comm = comm
        self.met_keys = met_keys
        self.get_scorers()
        self.scorers = {}
        ScorerE = namedtuple("ScorerE", ["fn", "out_str"])
        for k in self.met_keys:
            scorer_tuple = self.scorer_dict[k]
            if scorer_tuple.to_init:
                scorer = scorer_tuple.cls_fn()
            else:
                scorer = scorer_tuple.cls_fn
            self.scorers[k] = ScorerE(scorer, scorer_tuple.out_str)
        # if read_val_file:
        #     if not self.cfg.ds.val_full_bal:
        #         val_set = json.load(open(self.cfg.ds.val_qa_flat_bal_trim))
        #     else:
        #         val_set = json.load(open(self.cfg.ds.val_qa_flat_full_bal_trim))
        #     self.val_srl_annots = {x["qsrl_ind"]: x for x in val_set}

    def read_gt(self, gt_file):
        val_set = json.load(open(gt_file))
        self.val_srl_annots = {x["qsrl_ind"]: x for x in val_set}

    def get_scorers(self):
        # from pycoco_scorers_vizseq import BLEUScorerAll
        from pycocoevalcap.bleu.bleu import Bleu

        # from pycocoevalcap.spice.spice import Spice
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        import logging
        import transformers

        transformers.tokenization_utils.logger.setLevel(logging.ERROR)
        transformers.configuration_utils.logger.setLevel(logging.ERROR)
        transformers.modeling_utils.logger.setLevel(logging.ERROR)
        Scorer_ = namedtuple("Scorer_", ["cls_fn", "to_init", "out_str"])
        self.scorer_dict = {
            "bleu": Scorer_(
                Bleu(4, verbose=0), False, ["bleu@1", "bleu@2", "bleu@3", "bleu@4"]
            ),
            "meteor": Scorer_(Meteor(), False, ["meteor"]),
            "cider": Scorer_(Cider("corpus"), False, ["cider"]),
            "rouge": Scorer_(Rouge(), False, ["rouge"]),
            # "spice": Scorer_(Spice(), False, ["spice"]),
            "bert_score": Scorer_(BertScoreSimple, True, ["bert_score"]),
        }
        self.tokenizer = PTBTokenizer()

    def get_fin_scores(self, tok_hypo_dct, tok_gts_ref_dct, met_keys: List[str] = None):
        out_score_dict = {}
        if met_keys is None:
            met_keys = self.met_keys
        for k in met_keys:
            scorer = self.scorers[k]
            corpus_score, sent_score = scorer.fn.compute_score(
                tok_gts_ref_dct, tok_hypo_dct
            )

            if isinstance(corpus_score, float):
                assert len(scorer.out_str) == 1
                out_score_dict[scorer.out_str[0]] = corpus_score
            else:
                for oi, ostr in enumerate(scorer.out_str):
                    out_score_dict[ostr] = corpus_score[oi]

        return out_score_dict

    def get_fin_scores_rescaled(self, tok_hypo_dct, tok_gts_ref_dct, full_bal=False):
        def retokenize(dct):
            return {k: [v] for k, v in dct.items()}

        def get_ref_hyp_bas(ref_dct, hyp_dct):
            qsrl_inds = [k for k in ref_dct]
            qa_pair1 = {k: self.val_srl_annots[k]["qa_pair"] for k in qsrl_inds}
            ref_new_dct = {
                k: qa_pair1[k]["question"].replace(
                    qa_pair1[k]["question_type"], qa_pair1[k]["answer"]
                )
                for k in qa_pair1
            }
            bas_new_dct = {
                k: qa_pair1[k]["question"].replace(qa_pair1[k]["question_type"], "")
                for k in qa_pair1
            }
            hyp_new_dct = {
                k: qa_pair1[k]["question"].replace(
                    qa_pair1[k]["question_type"], " ".join(hyp_dct[k])
                )
                for k in qa_pair1
            }
            return (
                retokenize(ref_new_dct),
                retokenize(bas_new_dct),
                retokenize(hyp_new_dct),
            )

        def rescore_one(
            ref_base_score1, ref_hyp_score1, refref_score1=1, sc_key: str = None
        ):
            if sc_key is not None:
                if sc_key != "cider":
                    assert abs(refref_score1 - 1) < 0.01
                    refref_score1 = 1
            if ref_base_score1 < 0.98 * refref_score1:
                return (ref_hyp_score1 - ref_base_score1) / (
                    refref_score1 - ref_base_score1
                )
            else:
                print("Pain")
                return 1

        def get_score_from_cs(score1, score2):
            if score1 < 0.1 or score2 < 0.1:
                return 0
            else:
                return (score1 + score2) / 2

        def get_cons(score1, score2, cons_thresh=0.1):
            if score1 < cons_thresh and score2 < cons_thresh:
                return 1
            elif score1 > cons_thresh and score2 > cons_thresh:
                return 1
            else:
                return 0

        def get_out_dct_scores_from_ref_hyp_base(
            scorer_key: str,
            sent_score_only_phr,
            sent_score_ref_bas,
            sent_score_ref_hyp,
            sent_score_ref_ref,
        ):

            assert len(sent_score_ref_bas) == len(sent_score_ref_hyp)
            sent_adj_scores = [
                rescore_one(rb_score1, rh_score1, rrscore1, scorer_key)
                for rb_score1, rh_score1, rrscore1 in zip(
                    sent_score_ref_bas, sent_score_ref_hyp, sent_score_ref_ref
                )
            ]
            sent_adj_scores_dct = {
                k: sent_adj_scores[i] for i, k in enumerate(qsrl_ids)
            }
            sent_only_phr_dct = {
                k: sent_score_only_phr[i] for i, k in enumerate(qsrl_ids)
            }
            sent_new_adj_scores = {}
            sent_only_phr_bal_scores = {}
            sent_fin_adj_scores = {}
            cons_adj_score = {}
            for k in sent_adj_scores_dct:
                if len(self.val_srl_annots[k]["cs_qsrl_inds"]) > 0:
                    if not full_bal:
                        other_ind = self.val_srl_annots[k]["cs_qsrl_inds"][0]
                    else:
                        assert "matched_ind" in self.val_srl_annots[k]
                        other_ind = self.val_srl_annots[k]["matched_ind"]
                    new_score_adj = get_score_from_cs(
                        sent_adj_scores_dct[k], sent_adj_scores_dct[other_ind]
                    )
                    cons_adj_score[k] = get_cons(
                        sent_adj_scores_dct[k], sent_adj_scores_dct[other_ind]
                    )
                    sent_new_adj_scores[k] = new_score_adj

                    new_score_simp = get_score_from_cs(
                        sent_only_phr_dct[k], sent_only_phr_dct[other_ind]
                    )
                    sent_only_phr_bal_scores[k] = new_score_simp

                    sent_fin_adj_scores[k] = min(new_score_adj, new_score_simp)

            corpus_adj_scores = sum(sent_adj_scores) / len(sent_adj_scores)
            corpus_only_phr_scores = sum(sent_score_only_phr) / len(sent_score_only_phr)
            corpus_adj_scores_new = sum(
                [sent_new_adj_scores[k] for k in sent_new_adj_scores]
            ) / len(sent_new_adj_scores)
            corpus_fin_adj_scores = sum(
                [sent_fin_adj_scores[k] for k in sent_fin_adj_scores]
            ) / len(sent_fin_adj_scores)

            corpush_bal_only_phr = sum(
                [sent_only_phr_bal_scores[k] for k in sent_only_phr_bal_scores]
            ) / len(sent_only_phr_bal_scores)

            corpus_cons = sum([cons_adj_score[k] for k in cons_adj_score]) / len(
                cons_adj_score
            )

            sc_str = scorer_key

            out_dct = {}
            out_dct[sc_str] = corpus_fin_adj_scores

            out_dct[f"{sc_str}_cons_sent"] = cons_adj_score
            out_dct[f"{sc_str}_cons_corpus"] = corpus_cons

            out_dct[f"{sc_str}_corpus_fin_adj_scores"] = corpus_fin_adj_scores
            out_dct[f"{sc_str}_sent_fin_adj_scores"] = sent_fin_adj_scores

            out_dct[f"{sc_str}_corpus_adj_bal"] = corpus_adj_scores_new
            out_dct[f"{sc_str}_sent_adj_bal"] = sent_new_adj_scores

            out_dct[f"{sc_str}_corpus_onlyphr_bal"] = corpush_bal_only_phr
            out_dct[f"{sc_str}_sent_onlyphr_bal"] = sent_only_phr_bal_scores

            out_dct[f"{sc_str}_corpus_adj_notbal"] = corpus_adj_scores
            out_dct[f"{sc_str}_sent_adj_scores_not_bal"] = sent_adj_scores_dct

            out_dct[f"{sc_str}_corpus_only_phr_scores_not_bal"] = corpus_only_phr_scores
            out_dct[f"{sc_str}_sent_only_phr_scores_not_bal"] = sent_only_phr_dct

            out_dct[f"{sc_str}_sent_ref_bas"] = sent_score_ref_bas
            out_dct[f"{sc_str}_sent_ref_hyp"] = sent_score_ref_hyp
            out_dct[f"{sc_str}_sent_ref_hyp_adj"] = sent_adj_scores
            out_dct[f"{sc_str}_sent_ref_ref"] = sent_score_ref_ref

            return out_dct

        out_score_dict = {}
        refnew, basnew, hypnew = get_ref_hyp_bas(tok_gts_ref_dct, tok_hypo_dct)
        qsrl_ids = sorted(list(tok_gts_ref_dct.keys()))
        for k in self.met_keys:

            print("Starting Scorer", k)
            scorer = self.scorers[k]
            corpus_score_only_phr, sent_score_only_phr = scorer.fn.compute_score(
                tok_gts_ref_dct, tok_hypo_dct
            )

            corpus_score_ref_bas, sent_score_ref_bas = scorer.fn.compute_score(
                refnew, basnew
            )
            corpus_score_ref_hyp, sent_score_ref_hyp = scorer.fn.compute_score(
                refnew, hypnew
            )
            corpus_score_ref_ref, sent_score_ref_ref = scorer.fn.compute_score(
                refnew, refnew
            )
            if k != "bleu":
                out_dct = get_out_dct_scores_from_ref_hyp_base(
                    scorer.out_str[0],
                    sent_score_only_phr,
                    sent_score_ref_bas,
                    sent_score_ref_hyp,
                    sent_score_ref_ref,
                )
                out_score_dict.update(out_dct)
            else:
                for bl_ix, scstr in enumerate(scorer.out_str):
                    if bl_ix >= 2:
                        continue
                    out_dct = get_out_dct_scores_from_ref_hyp_base(
                        scstr,
                        sent_score_only_phr[bl_ix],
                        sent_score_ref_bas[bl_ix],
                        sent_score_ref_hyp[bl_ix],
                        sent_score_ref_ref[bl_ix],
                    )
                    out_score_dict.update(out_dct)

        return out_score_dict

    def eval_vidqap_met(
        self,
        fname: str,
        split_type: str = "valid",
        do_rescaling: bool = False,
        gt_file=None,
    ):
        assert split_type in ["valid", "test"]
        assert Path(fname).exists()
        pred_file = pickle.load(open(fname, "rb"))

        if gt_file is None:
            gt_file = self.cfg.ds.val_qa_trim

        self.read_gt(gt_file=gt_file)

        hypo_dct = {}
        gts_ref_dct = {}

        for i, p in enumerate(pred_file):
            qsrl_inds = p["idx_qsrl"]
            assert len(qsrl_inds) == 1
            qsrl_ind = qsrl_inds[0]
            hypo_dct[qsrl_ind] = [{"caption": remove_nonascii(p["pred_tokens"][0])}]
            gts_ref_dct[qsrl_ind] = [{"caption": remove_nonascii(p["tgt_tokens"])}]

        assert len(hypo_dct) == len(gts_ref_dct)
        print("Num vids", len(hypo_dct))
        tok_hypo_dct = self.tokenizer.tokenize(hypo_dct)
        tok_gts_ref_dct = self.tokenizer.tokenize(gts_ref_dct)
        if not do_rescaling:
            out_score_dict = self.get_fin_scores(
                tok_hypo_dct=tok_hypo_dct, tok_gts_ref_dct=tok_gts_ref_dct
            )
        else:
            # if not self.cfg.ds.val_full_bal:
            full_bal = False
            # else:
            #     full_bal = True
            out_score_dict_by_qtype = {}

            qtype_lst = ["<Q-ARG0>", "<Q-V>", "<Q-ARG1>", "<Q-ARG2>", "<Q-ARGM-LOC>"]
            if self.cfg.ds_to_use == "ch":
                qtype_lst = qtype_lst[1:]
            for qtype in qtype_lst:
                key_lst = [
                    k
                    for k, x in self.val_srl_annots.items()
                    if (x["qa_pair"]["question_type"] == qtype) and (k in tok_hypo_dct)
                ]
                hypo_dct = {k: tok_hypo_dct[k] for k in key_lst}
                refo_dct = {k: tok_gts_ref_dct[k] for k in key_lst}
                out_score_dict_by_qtype[qtype] = self.get_fin_scores_rescaled(
                    tok_hypo_dct=hypo_dct, tok_gts_ref_dct=refo_dct, full_bal=full_bal
                )
            out_score_dict = self.get_fin_scores_rescaled(
                tok_hypo_dct=tok_hypo_dct,
                tok_gts_ref_dct=tok_gts_ref_dct,
                full_bal=full_bal,
            )
            out_score_dict_by_qtype["full_dct"] = out_score_dict
            fname = Path(fname)
            out_file = fname.parent / f"{fname.stem}_evl_outs.pkl"
            pickle.dump(out_score_dict_by_qtype, open(out_file, "wb"))

        out_score_dict.update({"hypo_dct": tok_hypo_dct, "refo_dct": tok_gts_ref_dct})
        return out_score_dict


def main(pred_file, split_type="valid"):
    from vidqa_code.eval_vidqap import get_met_keys_

    cfg = CN(yaml.safe_load(open("./configs/ivd_asrl_cfg.yml")))
    cfg.ds.val_full_bal = False
    met_keys = ["meteor", "rouge", "bert_score"]
    evl_fn = EvalFnQAP(cfg, None, met_keys)
    out = evl_fn.eval_vidqap_met(pred_file, do_rescaling=True)
    out_met_keys = get_met_keys_(met_keys)
    print({k: v for k, v in out.items() if k in out_met_keys})
    return evl_fn


if __name__ == "__main__":
    evl_fn = fire.Fire(main)
