"""
Use DataLoader for ASRL-QA and Charades-SRL-QA
"""

from torch.utils.data import Dataset
import torch
from torch import Tensor
from pathlib import Path
from yacs.config import CfgNode as CN
import yaml
import pandas as pd
import h5py

import numpy as np
import json
import copy

from typing import Dict, List, Tuple
from utils.type_hint_utils import Pstr

from munch import Munch
from utils.trn_utils import DataWrap


import pickle
from utils.trn_utils import get_dataloader

from fairseq.data import Dictionary
from fairseq.tokenizer import tokenize_line


def coalesce_dicts(dct_list: List[Dict]) -> Dict:
    """
    Convert list of dicts with different keys
    to a single dict
    """
    out_dict = {}
    for dct in dct_list:
        for k in dct:
            if k in out_dict:
                assert torch.all(out_dict[k] == dct[k])

        out_dict.update(dct)
    return out_dict


def truncate_batch(
    inp_dict: Dict[str, Tensor], key: str, max_len: int, dim: int
) -> Dict[str, Tensor]:
    """
    Truncate the value for the dictionary key
    with max len and wrt dim
    """
    assert len(inp_dict[key].shape) > dim
    if dim == 1:
        inp_dict[key] = inp_dict[key][:, :max_len].contiguous()
    elif dim == 2:
        inp_dict[key] = inp_dict[key][:, :, :max_len].contiguous()
    elif dim == 3:
        inp_dict[key] = inp_dict[key][:, :, :, :max_len].contiguous()
    else:
        raise NotImplementedError

    return


def pad_words(
    word_list: List, max_len: int, pad_index, eos_index=None, append_eos=False
) -> Tuple[List, int]:
    if append_eos:
        assert eos_index is not None
        cur_len = len(word_list)
        if cur_len >= max_len:
            return word_list[: max_len - 1] + [eos_index], max_len
        out_word_list = word_list + [eos_index] + [pad_index] * (max_len - 1 - cur_len)
        return out_word_list, cur_len + 1
    else:
        cur_len = len(word_list)
        if cur_len > max_len:
            return word_list[:max_len], max_len
        out_word_list = word_list + [pad_index] * (max_len - cur_len)
        return out_word_list, cur_len


def add_prev_tokens(
    inp_dict: Dict[str, Tensor], key: str, pad_token: int, bos_token: int
) -> Dict[str, Tensor]:
    """
    Create prev tokens for the given dictionary key
    """
    src_toks = inp_dict[key]
    prev_output_tokens = src_toks.new_full(src_toks.shape, fill_value=pad_token)
    prev_output_tokens[..., 0] = bos_token
    prev_output_tokens[..., 1:] = src_toks[..., :-1].clone()
    out_key = f"prev_out_{key}"
    inp_dict[out_key] = prev_output_tokens
    return


def simple_collate_dct_list(
    batch: List[Dict], stack_or_cat: str = "stack", cat_dim: int = None
) -> Dict[str, List]:
    """
    Convert List[Dict[k, tensor]] -> Dict[k, Stacked Tensor]
    """
    assert stack_or_cat in ["stack", "cat"]
    if stack_or_cat == "cat":
        assert cat_dim is not None
    out_dict = {}
    # nothing needs to be done
    all_keys = list(batch[0].keys())
    if stack_or_cat == "stack":
        batch_size = len(batch)
    else:
        batch_size = len(batch) * batch[0][all_keys[0]].shape[0]
    for k in all_keys:

        if stack_or_cat == "stack":
            out_dict[k] = torch.stack([b[k] for b in batch])
        elif stack_or_cat == "cat":
            out_dict[k] = torch.cat([b[k] for b in batch], cat_dim)
        else:
            raise NotImplementedError
    assert all([len(v) == batch_size for k, v in out_dict.items()])
    return out_dict


class AnetEntDataset(Dataset):
    """
    Dataset class adopted from
    https://github.com/facebookresearch/grounded-video-description
    /blob/master/misc/dataloader_anet.py#L27
    This is basically AE loader.
    """

    def __init__(
        self, cfg: CN, ann_file: Pstr, split_type: str = "train", comm: Dict = None
    ):
        self.cfg = cfg

        # Common stuff that needs to be passed around
        if comm is not None:
            # assert isinstance(comm, (dict, Munch))
            self.comm = comm
        else:
            self.comm = Munch()

        self.split_type = split_type
        self.ann_file = Path(ann_file)
        assert self.ann_file.suffix == ".csv"
        self.set_args()
        self.load_annotations()

        with h5py.File(self.proposal_h5, "r") as h5_proposal_file:
            self.num_proposals = h5_proposal_file["dets_num"][:]
            self.label_proposals = h5_proposal_file["dets_labels"][:]

        self.itemgetter = getattr(self, "simple_item_getter")
        self.test_mode = split_type != "test"

        self.after_init()

    def after_init(self):
        pass

    def set_args(self):
        """
        Define the arguments to be used from the cfg
        """
        # NOTE: These are changed at extended_config/post_proc_config
        dct = self.cfg.ds[f"{self.cfg.ds.exp_setting}"]
        self.proposal_h5 = Path(dct["proposal_h5"])
        self.feature_root = Path(dct["feature_root"])

        # Assert raw caption file (from activity net captions) exists
        self.raw_caption_file = Path(self.cfg.ds.anet_cap_file)
        assert self.raw_caption_file.exists(), f"{self.raw_caption_file}"

        # Assert act ent caption file with bbox exists
        self.anet_ent_annot_file = Path(self.cfg.ds.anet_ent_annot_file)
        assert self.anet_ent_annot_file.exists(), f"{self.anet_ent_annot_file}"

        # Max proposals to be considered
        # By default it is 10 * 100
        self.num_frms = self.cfg.ds.num_sampled_frm
        self.num_prop_per_frm = dct["num_prop_per_frm"]
        self.comm.num_prop_per_frm = self.num_prop_per_frm
        self.max_proposals = self.num_prop_per_frm * self.num_frms

        # Assert h5 file to read from exists
        assert self.proposal_h5.exists(), str(self.proposal_h5)

        # Assert region features exists
        assert self.feature_root.exists()

        # Assert rgb, motion features exists
        self.seg_feature_root = Path(self.cfg.ds.seg_feature_root)
        assert self.seg_feature_root.exists()

        # Which proposals to be included
        self.prop_thresh = self.cfg.misc.prop_thresh
        self.exclude_bgd_det = self.cfg.misc.exclude_bgd_det

        # Assert word vocab files exist
        # self.dic_anet_file = Path(self.cfg.ds.anet_ent_split_file)
        # assert self.dic_anet_file.exists()

        # Max gt box to consider
        # should consider all, set high
        self.max_gt_box = self.cfg.ds.max_gt_box

        # temporal attention size
        self.t_attn_size = self.cfg.ds.t_attn_size

        # Sequence length
        self.seq_length = self.cfg.ds.max_seq_length

    def load_annotations(self):
        """
        Process the annotation file.
        """
        # Load annotation files
        self.annots = pd.read_csv(self.ann_file)
        # Needs to exported as well

        # Load raw captions
        with open(self.raw_caption_file) as f:
            self.raw_caption = json.load(f)

        # Load anet bbox
        with open(self.anet_ent_annot_file) as f:
            self.anet_ent_captions = json.load(f)

        # Load dictionaries
        # with open(self.dic_anet_file) as f:
        #     self.comm.dic_anet = json.load(f)

        # # Get detections to index
        # self.comm.dtoi = {w: i + 1 for w, i in self.comm.dic_anet["wtod"].items()}
        # self.comm.itod = {i: w for w, i in self.comm.dtoi.items()}
        # self.comm.itow = self.comm.dic_anet["ix_to_word"]
        # wtoi = {w: i for i, w in self.comm.itow.items()}
        # wtoi_dict = Dictionary()
        # del wtoi['UNK']
        # for w in wtoi:
        #     wtoi_dict.add_symbol(w)
        wtoi_dict = pickle.load(open(self.cfg.ds.word_vocab_file, "rb"))
        self.comm.wtoi = wtoi_dict
        self.comm.vocab_size = len(self.comm.wtoi)
        # self.comm.detect_size = len(self.comm.itod)

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx: int):
        return self.itemgetter(idx)

    def pad_words_with_vocab(self, out_list, voc=None, pad_len=-1, defm=[1]):
        """
        Input is a list.
        If curr_len < pad_len: pad remaining with default value
        Instead, if cur_len > pad_len: trim the input
        """
        curr_len = len(out_list)
        if pad_len == -1 or curr_len == pad_len:
            return out_list
        else:
            if curr_len > pad_len:
                return out_list[:pad_len]
            else:
                if voc is not None and hasattr(voc, "itos"):
                    assert voc.itos[1] == "<pad>"
                out_list += defm * (pad_len - curr_len)
                return out_list

    def get_props(self, index: int):
        """
        Returns the padded proposals, padded mask, number of proposals
        by reading the h5 files
        """
        num_proposals = int(self.num_proposals[index])
        label_proposals = self.label_proposals[index]
        proposals = copy.deepcopy(label_proposals[:num_proposals, :])

        assert num_proposals > 0
        # proposal mask to filter out low-confidence proposals or backgrounds
        # mask is 1 if proposal is included
        pnt_mask = proposals[:, 6] >= self.prop_thresh
        if self.exclude_bgd_det:
            pnt_mask &= proposals[:, 5] != 0

        num_props = min(proposals.shape[0], self.max_proposals)

        padded_props = self.pad_words_with_vocab(
            proposals.tolist(), pad_len=self.max_proposals, defm=[[0] * 7]
        )
        padded_mask = self.pad_words_with_vocab(
            pnt_mask.tolist(), pad_len=self.max_proposals, defm=[0]
        )
        return np.array(padded_props), np.array(padded_mask), num_props

    def get_features(self, vid_seg_id: str, num_proposals: int, props):
        """
        Returns the region features, rgb-motion features
        """

        vid_id_ix, seg_id_ix = vid_seg_id.split("_segment_")
        seg_id_ix = str(int(seg_id_ix))

        region_feature_file = self.feature_root / f"{vid_seg_id}.npy"
        region_feature = np.load(region_feature_file)
        region_feature = region_feature.reshape(-1, region_feature.shape[2]).copy()
        assert num_proposals == region_feature.shape[0]
        if self.cfg.misc.add_prop_to_region:
            region_feature = np.concatenate(
                [region_feature, props[:num_proposals, :5]], axis=1
            )

        # load the frame-wise segment feature
        seg_rgb_file = self.seg_feature_root / f"{vid_id_ix[2:]}_resnet.npy"
        seg_motion_file = self.seg_feature_root / f"{vid_id_ix[2:]}_bn.npy"

        assert seg_rgb_file.exists() and seg_motion_file.exists()

        seg_rgb_feature = np.load(seg_rgb_file)
        seg_motion_feature = np.load(seg_motion_file)
        seg_feature_raw = np.concatenate((seg_rgb_feature, seg_motion_feature), axis=1)

        return region_feature, seg_feature_raw

    def get_frm_mask(self, proposals, gt_bboxs):
        """
        1 where proposals and gt box don't match
        0 where they match
        We are basically matching the frame indices,
        that is 1 where they belong to different frames
        0 where they belong to same frame.

        In mdl_bbox_utils.py -> bbox_overlaps_batch
        frm_mask ~= frm_mask is used.
        (We have been tricked, we have been backstabbed,
        quite possibly bamboozled)
        """
        # proposals: num_pps
        # gt_bboxs: num_box
        num_pps = proposals.shape[0]
        num_box = gt_bboxs.shape[0]
        return np.tile(proposals.reshape(-1, 1), (1, num_box)) != np.tile(
            gt_bboxs, (num_pps, 1)
        )

    def get_seg_feat_for_frms(self, seg_feats, timestamps, duration, idx=None):
        """
        Given seg features of shape num_frms x 3072
        converts to 10 x 3072
        Here 10 is the number of frames used by the mdl
        timestamps contains the start and end time of the clip
        duration is the total length of the video
        note that end-st != dur, since one is for the clip
        other is for the video

        Additionally returns average over the timestamps
        """
        # ctx is the context of the optical flow used
        # 10 means 5 seconds previous, to 5 seconds after
        # This is because optical flow is calculated at
        # 2fps
        ctx = self.cfg.misc.ctx_for_seg_feats
        if timestamps[0] > timestamps[1]:
            # something is wrong in AnetCaptions dataset
            # since only 2 have problems, ignore
            # print(idx, 'why')
            timestamps = timestamps[1], timestamps[0]
        st_time = timestamps[0]
        end_time = timestamps[1]
        duration_clip = end_time - st_time

        num_frms = seg_feats.shape[0]
        # usually select 10 frms, but can select more
        num_frms_to_select_orig = self.cfg.ds.num_frms_to_select
        num_frms_to_select = self.cfg.ds.num_frms_to_select

        if not num_frms >= num_frms_to_select:
            num_frms_to_select = num_frms

        frm_ind = np.arange(0, num_frms_to_select)
        frm_time = st_time + (duration_clip / num_frms_to_select) * (frm_ind + 0.5)
        # *2 because of sampling at 2fps
        frm_index_in_seg_feat = np.minimum(
            np.maximum((frm_time * 2).astype(np.int_) - 1, 0), num_frms - 1
        )

        st_indices = np.maximum(frm_index_in_seg_feat - ctx - 1, 0)
        end_indices = np.minimum(frm_index_in_seg_feat + ctx + 1, num_frms)

        if not st_indices[0] == end_indices[-1]:
            try:
                seg_feats_frms_glob = np.zeros(
                    (num_frms_to_select_orig, seg_feats.shape[-1])
                )
                seg_feats_frms_glob[: len(st_indices)] = seg_feats[st_indices]
                # st_ind = st_indices[0]
                # end_ind = end_indices[-1]
                # tot_ind = end_ind - st_ind
                # seg_feats_frms_glob[:tot_ind] = seg_feats[st_ind:end_ind]

            except RuntimeWarning:
                import pdb

                pdb.set_trace()
        else:
            print(f"clip duration: {duration_clip}")
            seg_feats_frms_glob = seg_feats[st_indices[0]]

        assert np.all(end_indices - st_indices > 0)
        try:
            if ctx != 0:
                # seg_feats_frms = np.vstack([
                #     seg_feats[sti:endi, :].mean(axis=0)
                #     for sti, endi in zip(st_indices, end_indices)])
                raise NotImplementedError
            else:
                frm_ind_orig = np.arange(0, 10)
                frm_time_orig = st_time + (duration_clip / num_frms) * (
                    frm_ind_orig + 0.5
                )
                frm_index_in_seg_feat_orig = np.minimum(
                    np.maximum((frm_time_orig * 2).astype(np.int_) - 1, 0), num_frms - 1
                )

                seg_feats_frms = np.zeros((10, seg_feats.shape[-1]))
                assert len(frm_index_in_seg_feat_orig) == 10
                seg_feats_frms[:10] = seg_feats[frm_index_in_seg_feat_orig]
        except RuntimeWarning:
            import pdb

            pdb.set_trace()
            pass
        return seg_feats_frms, seg_feats_frms_glob

    def get_gt_annots(self, caption_dct: Dict, idx: int):
        gt_bboxs = torch.tensor(caption_dct["bbox"]).float()
        gt_frms = torch.tensor(caption_dct["frm_idx"]).unsqueeze(-1).float()
        assert len(gt_bboxs) == len(gt_frms)
        num_box = len(gt_bboxs)
        gt_bboxs_t = torch.cat([gt_bboxs, gt_frms], dim=-1)

        padded_gt_bboxs = self.pad_words_with_vocab(
            gt_bboxs_t.tolist(), pad_len=self.max_gt_box, defm=[[0] * 5]
        )
        padded_gt_bboxs_mask_list = [1] * num_box
        padded_gt_box_mask = self.pad_words_with_vocab(
            padded_gt_bboxs_mask_list, pad_len=self.max_gt_box, defm=[0]
        )
        return {
            "padded_gt_bboxs": np.array(padded_gt_bboxs),
            "padded_gt_box_mask": np.array(padded_gt_box_mask),
            "num_box": num_box,
        }

    def simple_item_getter(self, idx: int):
        """
        Basically, this returns stuff for the
        vid_seg_id obtained from the idx
        """
        row = self.annots.iloc[idx]

        vid_id = row["vid_id"]
        seg_id = str(row["seg_id"])
        vid_seg_id = row["id"]
        ix = row["Index"]

        # Get the padded proposals, proposal masks and the number of proposals
        padded_props, pad_pnt_mask, num_props = self.get_props(ix)

        # Get the region features and the segment features
        # Region features are for spatial stuff
        # Segment features are for temporal stuff
        region_feature, seg_feature_raw = self.get_features(
            vid_seg_id, num_proposals=num_props, props=padded_props
        )

        # not accurate, with minor misalignments
        # Get the time stamp information for each segment
        timestamps = self.raw_caption[vid_id]["timestamps"][int(seg_id)]

        # Get the durations for each time stamp
        dur = self.raw_caption[vid_id]["duration"]

        # Get the number of frames in the segment
        num_frm = seg_feature_raw.shape[0]

        # basically time stamps.
        # Not really used, kept for legacy reasons
        sample_idx = np.array(
            [
                np.round(num_frm * timestamps[0] * 1.0 / dur),
                np.round(num_frm * timestamps[1] * 1.0 / dur),
            ]
        )

        sample_idx = np.clip(np.round(sample_idx), 0, self.t_attn_size).astype(int)

        # Get segment features based on the number of frames used
        seg_feature = np.zeros((self.t_attn_size, seg_feature_raw.shape[1]))
        seg_feature[: min(self.t_attn_size, num_frm)] = seg_feature_raw[
            : self.t_attn_size
        ]

        # gives both local and global features.
        # In model can choose either one
        seg_feature_for_frms, seg_feature_for_frms_glob = self.get_seg_feat_for_frms(
            seg_feature_raw, timestamps, dur, idx
        )

        # get gt annotations
        # Get the a AE annotations
        caption_dct = self.anet_ent_captions[vid_id]["segments"][seg_id]
        vid_sent_toks = [self.comm.wtoi.index(w) for w in caption_dct["caption"]]
        vid_sent_toks_pad, vid_sent_toks_len = pad_words(
            vid_sent_toks,
            max_len=self.seq_length,
            pad_index=self.comm.wtoi.pad_index,
            eos_index=self.comm.wtoi.eos_index,
            append_eos=True,
        )

        # get the groundtruth_box annotations
        gt_annot_dict = self.get_gt_annots(caption_dct, idx)
        # extract the padded gt boxes
        pad_gt_bboxs = gt_annot_dict["padded_gt_bboxs"]
        # store the number of gt boxes
        num_box = gt_annot_dict["num_box"]

        # frame mask is NxM matrix of which proposals
        # lie in the same frame of groundtruth
        frm_mask = self.get_frm_mask(
            padded_props[:num_props, 4], pad_gt_bboxs[:num_box, 4]
        )
        # pad it
        pad_frm_mask = np.ones((self.max_proposals, self.max_gt_box))
        pad_frm_mask[:num_props, :num_box] = frm_mask

        pad_pnt_mask = torch.tensor(pad_pnt_mask).long()

        # pad region features
        pad_region_feature = np.zeros((self.max_proposals, region_feature.shape[1]))
        pad_region_feature[:num_props] = region_feature[:num_props]

        out_dict = {
            # segment features
            "seg_feature": torch.from_numpy(seg_feature).float(),
            # local segment features
            "seg_feature_for_frms": torch.from_numpy(seg_feature_for_frms).float(),
            # global segment features
            "seg_feature_for_frms_glob": torch.from_numpy(
                seg_feature_for_frms_glob
            ).float(),
            # number of proposals
            "num_props": torch.tensor(num_props).long(),
            # number of groundtruth boxes
            "num_box": torch.tensor(num_box).long(),
            # padded proposals
            "pad_proposals": torch.tensor(padded_props).float(),
            # padded groundtruth boxes
            "pad_gt_bboxs": torch.tensor(pad_gt_bboxs).float(),
            # padded groundtruth mask, not used, kept for legacy
            "pad_gt_box_mask": torch.tensor(gt_annot_dict["padded_gt_box_mask"]).byte(),
            # segment id, not used, kept for legacy
            "seg_id": torch.tensor(int(seg_id)).long(),
            # idx, ann_idx are same correspond to
            # it is the index of vid_seg in the ann_file
            "idx": torch.tensor(idx).long(),
            "ann_idx": torch.tensor(idx).long(),
            # padded region features
            "pad_region_feature": torch.tensor(pad_region_feature).float(),
            # padded frame mask
            "pad_frm_mask": torch.tensor(pad_frm_mask).byte(),
            # padded proposal mask
            "pad_pnt_mask": pad_pnt_mask.byte(),
            # sample number, not used, legacy
            "sample_idx": torch.tensor(sample_idx).long(),
            "vid_sent_toks": torch.tensor(vid_sent_toks_pad).long(),
            "vid_sent_lens": torch.tensor(vid_sent_toks_len).long(),
        }

        return out_dict


class Anet_Ent_(AnetEntDataset):
    def after_init(self):
        # basically, unsqueeze(0) for all
        self.itemgetter = getattr(self, "simple_item_getter_usq")

    def simple_item_getter_usq(self, idx):
        out_dict_1 = self.simple_item_getter(idx)
        out_dict_col = simple_collate_dct_list([out_dict_1])
        out_dict_col["sent_idx"] = torch.tensor(0).long()
        return out_dict_col


class AnetSRL_VidQA_:
    def after_init(self):
        if self.split_type == "train":
            srl_annot_file = self.cfg.ds.trn_qa_trim
        elif self.split_type == "valid" or self.split_type == "test":
            srl_annot_file = self.cfg.ds.val_qa_trim
        else:
            raise NotImplementedError

        # Read the file
        self.srl_annots = json.load(open(srl_annot_file))
        assert hasattr(self, "srl_annots")

        self.srl_annots_by_qsrl_idx = {v["qsrl_ind"]: v for v in self.srl_annots}

        # DON'T INCLUDE Y/N questions
        self.qsrl_idx_lst = [
            v["qsrl_ind"]
            for v in self.srl_annots
            if v["qa_pair"]["question_type"] != "<Q-Y/N>"
        ]

        self.max_srl_in_sent = 1

        with open(self.cfg.ds.qword_vocab_file, "rb") as f:
            qword_dct = pickle.load(f)

        if self.cfg.mdl.use_phr_clf:
            if self.cfg.mdl.aword_type == "with_sw":
                afile = self.cfg.ds.ans_vocab_file
            elif self.cfg.mdl.aword_type == "no_sw":
                afile = self.cfg.ds.ans_nosw_vocab_file
            else:
                raise NotImplementedError
            with open(afile, "rb") as f:
                print("Loaded Answer Dict")
                self.comm.awvoc = pickle.load(f)

        self.comm.qwvoc = qword_dct

        # Only used for box expt
        self.box_per_srl_arg = self.cfg.misc.box_per_srl_arg

        self.itemgetter = getattr(self, "qa_item_getter2")

    def __len__(self) -> int:
        # return 100
        return len(self.qsrl_idx_lst)

    def process_srl_row_simple(self, srl_row, word_dct: Dictionary):
        qa_pair = srl_row["qa_pair"]

        def get_padded_toks_and_lens(inp, max_l):
            return pad_words(
                word_list=word_dct.encode_line(
                    inp, add_if_not_exist=False, append_eos=False
                ).tolist(),
                max_len=max_l,
                append_eos=True,
                eos_index=word_dct.eos_index,
                pad_index=word_dct.pad_index,
            )

        max_l = 20
        if self.split_type == "train":
            qarg_lemma_out_lst = qa_pair["qarg_lemma"]
            qarg_x = qa_pair["question_type"]
            mapping = {
                "<Q-V>": ["ARG0", "ARG1", "ARG2", "ARGM-LOC"],
                "<Q-ARG0>": ["V", "ARG1", "ARG2", "ARGM-LOC"],
                "<Q-ARG1>": ["V", "ARG0", "ARG2", "ARGM-LOC"],
                "<Q-ARG2>": ["V", "ARG0", "ARG1", "ARGM-LOC"],
                "<Q-ARGM-LOC>": ["V", "ARG0", "ARG1", "ARG2"],
            }
            qphrase_lst = []
            qphrase_ix_lst = []
            for qarg_lemm1_ix, qarg_lemm1 in enumerate(qarg_lemma_out_lst):
                if qarg_x == qarg_lemm1[0]:
                    qphrase_lst.append(qarg_lemm1[3])
                    qphrase_ix_lst.append(qarg_lemm1_ix)
                else:
                    if (
                        qarg_lemm1[0]
                        in mapping[qarg_x]
                        # and len(qphrase_lst) < 3
                    ):
                        qphrase_lst.append(qarg_lemm1[3])
                        qphrase_ix_lst.append(qarg_lemm1_ix)

            if len(qphrase_lst) < 3:
                question = qa_pair["question"]
                qphrase_ix_lst = [ix for ix in range(len(qarg_lemma_out_lst))]
            else:
                question = " ".join(qphrase_lst)
                qphrase_ix_lst = qphrase_ix_lst
        else:
            question = qa_pair["question"]
            qphrase_ix_lst = qa_pair["qphrase_ix_lst"]

        question_toks, question_tok_lens = get_padded_toks_and_lens(
            question, max_l=max_l
        )

        answer_toks, answer_tok_lens = get_padded_toks_and_lens(
            qa_pair["answer"], max_l=max_l
        )
        question_type = qa_pair["question_type"]

        out_dct = {
            "question_toks": torch.tensor(question_toks).long(),
            "question_tok_len": torch.tensor(question_tok_lens).long(),
            "question_type": torch.tensor(word_dct.indices[question_type]).long(),
            "answer_toks": torch.tensor(answer_toks).long(),
            "answer_tok_lens": torch.tensor(answer_tok_lens).long(),
        }

        if self.cfg.mdl.use_phr_clf:
            ans = qa_pair["answer"]
            if ans in self.comm.awvoc.indices:
                answer_tok1 = self.comm.awvoc.indices[ans]
            else:
                answer_tok1 = self.comm.awvoc.unk_index
            aeos_ind = self.comm.awvoc.eos_index
            out_dct["answer_clf"] = torch.tensor([answer_tok1, aeos_ind]).long()
            out_dct["answer_clf_lens"] = torch.tensor(2).long()

        if self.cfg.mdl.use_srl_bounds:
            # only for VOGNET-QAP
            question_srl_bounds = []
            cur_ix = 0
            num_srls_max = 5
            num_box_per_srl = 4
            if self.cfg.ds_name == "anet":
                vid_seg_gt_box = srl_row["gt_bboxes"]
                vid_seg_gt_frms = srl_row["gt_frms"]
            gt_bbox_lst = []

            req_cls_pats_mask = srl_row["req_cls_pats_mask"]
            to_break = False
            for qarg_le1_ix in qphrase_ix_lst:
                qarg_lemma1 = qa_pair["qarg_lemma"][qarg_le1_ix]
                tok_str = qarg_lemma1[3]
                assert isinstance(tok_str, str)
                tok_out = tokenize_line(tok_str)
                tok_len = len(tok_out)
                en_ix = cur_ix + tok_len - 1
                is_groundable = qarg_lemma1[2]
                if is_groundable and self.cfg.ds_name == "anet":
                    gt_info = req_cls_pats_mask[qarg_le1_ix]
                    assert gt_info[1] == 1
                    gbox_frm = []
                    for z in gt_info[2]:
                        gbx = copy.deepcopy(vid_seg_gt_box[z])
                        gfrm = copy.deepcopy(vid_seg_gt_frms[z])
                        gbx.append(gfrm)
                        gbox_frm += [gbx]
                else:
                    gbox_frm = [[0] * 5] * num_box_per_srl

                if len(gbox_frm) < num_box_per_srl:
                    gbox_frm += [[0] * 5] * (num_box_per_srl - len(gbox_frm))
                else:
                    gbox_frm = gbox_frm[:num_box_per_srl]
                gt_bbox_lst.append(gbox_frm)
                if en_ix < max_l - 1:
                    question_srl_bounds.append([cur_ix, en_ix])
                else:
                    question_srl_bounds.append([cur_ix, max_l - 1])
                    to_break = True
                if to_break:
                    break
                cur_ix += tok_len
            num_srls_used = min(len(question_srl_bounds), num_srls_max)

            if len(question_srl_bounds) > num_srls_max:
                question_srl_bounds = question_srl_bounds[:num_srls_max]
                gt_bbox_lst = gt_bbox_lst[:num_srls_max]
            else:
                to_add_srls = num_srls_max - len(question_srl_bounds)
                question_srl_bounds += [[0, 0]] * to_add_srls
                gt_bbox_lst += [[[0] * 5] * num_box_per_srl] * to_add_srls

            assert len(question_srl_bounds) == num_srls_max

            out_dct["question_srl_bounds_idxs"] = torch.tensor(
                question_srl_bounds
            ).long()
            out_dct["num_srls_used_msk"] = torch.tensor(
                [1] * num_srls_used + [0] * (num_srls_max - num_srls_used)
            ).long()
            out_dct["num_srls_used"] = torch.tensor(num_srls_used).long()
            out_dct["gt_bbox_for_srls"] = torch.tensor(gt_bbox_lst).float()
            out_dct["gt_bbox_for_srls_msk"] = (
                out_dct["gt_bbox_for_srls"].sum(dim=-1).ne(0)
            )
        return out_dct

    def qa_item_getter(self, idx):
        qsrl_ind = self.qsrl_idx_lst[idx]
        srl_row = self.srl_annots_by_qsrl_idx[qsrl_ind]
        # srl_row = self.srl_annots[idx]
        out_dict_srl = self.process_srl_row_simple(srl_row, self.comm.qwvoc)
        out_dict_srl["srl_idx"] = torch.tensor(idx).long()
        out_dict_srl["qsrl_idx"] = torch.tensor(qsrl_ind).long()
        out_dict_srl_col = simple_collate_dct_list([out_dict_srl])
        out_dict_vid = self.simple_item_getter_usq(srl_row["ann_ind"])
        out_dict_vid["sent_idx"] = torch.tensor(idx).long()
        out_dict = coalesce_dicts([out_dict_vid, out_dict_srl_col])

        return out_dict

    def qa_item_getter2(self, idx):
        qsrl_ind = self.qsrl_idx_lst[idx]
        srl_row = self.srl_annots_by_qsrl_idx[qsrl_ind]
        out_dict = self.qa_item_getter(idx)
        if self.split_type == "train":
            if self.cfg.ds.trn_bal:
                other_ind = None
                choose_rand = True
                if "cs_qsrl_inds" in srl_row:
                    if len(srl_row["cs_qsrl_inds"]) > 0:
                        other_ind = int(np.random.choice(srl_row["cs_qsrl_inds"]))
                        choose_rand = False
                if choose_rand:
                    other_ind = int(np.random.choice(srl_row["rand_qsrl_inds"]))

                assert other_ind is not None
                out_dict2 = self.qa_item_getter(other_ind)
                out_dict = simple_collate_dct_list([out_dict, out_dict2])
        return out_dict


class AnetSRL_VidQA(AnetSRL_VidQA_, Anet_Ent_):
    pass


class Charades_Ent_(Dataset):
    def __init__(self, cfg, split_type: str, comm=None):
        self.cfg = cfg
        if comm is not None:
            self.comm = comm
        else:
            self.comm = Munch()

        assert split_type in ["train", "valid", "test"]
        self.split_type = split_type
        self.seg_feat_dir = Path(self.cfg.ds.seg_feats_mixed_5c_out)
        assert self.seg_feat_dir.exists()
        self.after_init()

    def simple_item_getter_usq(self, vid_name: str):
        seg_feats_file = self.seg_feat_dir / f"{vid_name}_mixed_feat.npy"
        assert seg_feats_file.exists()
        seg_feats = np.load(seg_feats_file)
        nfrms = self.cfg.ds.num_frms_to_select
        seg_feats_frms_glob = np.zeros((nfrms, seg_feats.shape[1]))
        nfrms_choose = min(nfrms, seg_feats.shape[0])
        seg_feats_frms_glob[:nfrms_choose] = seg_feats[:nfrms_choose]

        return {"seg_feature_for_frms_glob": torch.tensor(seg_feats_frms_glob).float()}

    def __getitem__(self, idx: int):
        return self.itemgetter(idx)


class Charades_VidQA_(AnetSRL_VidQA_):
    def after_init(self):
        if self.split_type == "train":
            srl_annot_file = self.cfg.ds.trn_qa_trim
        elif self.split_type == "valid" or self.split_type == "test":
            srl_annot_file = self.cfg.ds.val_qa_trim
        else:
            raise NotImplementedError

        # Read the file
        self.srl_annots = json.load(open(srl_annot_file))
        assert hasattr(self, "srl_annots")

        self.srl_annots_by_qsrl_idx = {v["qsrl_ind"]: v for v in self.srl_annots}
        self.fps1_frm_lst = pd.read_csv(self.cfg.ds.fps1_frm_list, index_col=0)
        self.vid_ids_lst = set(self.fps1_frm_lst.vid_name.unique())
        self.qsrl_idx_lst = [v["qsrl_ind"] for v in self.srl_annots]

        self.max_srl_in_sent = 1
        self.srl_arg_len = self.cfg.misc.srl_arg_length
        with open(self.cfg.ds.qword_vocab_file, "rb") as f:
            qword_dct = pickle.load(f)

        if self.cfg.mdl.use_phr_clf:
            if self.cfg.mdl.aword_type == "with_sw":
                afile = self.cfg.ds.ans_vocab_file
            elif self.cfg.mdl.aword_type == "no_sw":
                afile = self.cfg.ds.ans_nosw_vocab_file
            else:
                raise NotImplementedError
            with open(afile, "rb") as f:
                print("Loaded Answer Dict")
                self.comm.awvoc = pickle.load(f)

        self.comm.qwvoc = qword_dct

        self.box_per_srl_arg = self.cfg.misc.box_per_srl_arg

        self.itemgetter = getattr(self, "qa_item_getter2")

    def qa_item_getter(self, idx):
        qsrl_ind = self.qsrl_idx_lst[idx]
        srl_row = self.srl_annots_by_qsrl_idx[qsrl_ind]

        out_dict_srl = self.process_srl_row_simple(srl_row, self.comm.qwvoc)
        out_dict_srl["srl_idx"] = torch.tensor(idx).long()
        out_dict_srl["qsrl_idx"] = torch.tensor(qsrl_ind).long()
        out_dict_srl_col = simple_collate_dct_list([out_dict_srl])
        vid_id = srl_row["vid_id"]
        out_dict1 = self.simple_item_getter_usq(vid_id)
        out_dict1["ann_idx"] = torch.tensor(idx).long()
        out_dict_vid = simple_collate_dct_list([out_dict1])
        out_dict_vid["sent_idx"] = torch.tensor(idx).long()

        out_dict = coalesce_dicts([out_dict_vid, out_dict_srl_col])

        return out_dict


class Charades_SRL_QA(Charades_VidQA_, Charades_Ent_):
    pass


class BatchCollator_QA:
    def __init__(self, cfg, comm, split_type="train"):
        self.cfg = cfg
        self.comm = comm
        self.batch_collator_fn = getattr(self, "batch_collator_qa")
        self.after_init()
        self.split_type = split_type

    def after_init(self):
        pass

    def __call__(self, batch):
        return self.batch_collator_fn(batch)

    def batch_collator_qa(self, batch):
        if self.cfg.ds.trn_bal and self.split_type == "train":
            stack_or_cat = "cat"
            cat_dim = 0
        else:
            stack_or_cat = "stack"
            cat_dim = None
        out_dict = simple_collate_dct_list(
            batch, stack_or_cat=stack_or_cat, cat_dim=cat_dim
        )
        max_qlen = out_dict["question_toks"].max()
        truncate_batch(out_dict, key="question_toks", max_len=max_qlen, dim=2)

        max_alen = out_dict["answer_toks"].max()
        truncate_batch(out_dict, key="answer_toks", max_len=max_alen, dim=2)

        wvoc = self.comm.qwvoc
        add_prev_tokens(
            out_dict,
            key="answer_toks",
            pad_token=wvoc.pad_index,
            bos_token=wvoc.eos_index,
        )
        if "answer_clf" in out_dict:
            awvoc = self.comm.awvoc
            add_prev_tokens(
                out_dict,
                key="answer_clf",
                pad_token=awvoc.pad_index,
                bos_token=awvoc.eos_index,
            )

        return out_dict


def get_data(cfg):
    task = cfg.task
    assert task == "vid_qa", f"Only supported task {task}"
    if cfg.ds_name == "anet":
        DS = AnetSRL_VidQA
        BC = BatchCollator_QA
    elif cfg.ds_name == "ch":
        DS = Charades_SRL_QA
        BC = BatchCollator_QA
    else:
        raise NotImplementedError

    # Training file
    if cfg.ds_name == "anet":
        # Train file
        trn_ann_file = cfg.ds["trn_ann_file"]
        trn_ds = DS(cfg=cfg, ann_file=trn_ann_file, split_type="train")
        comm = trn_ds.comm
        # Validation file
        val_ann_file = cfg.ds["val_ann_file"]
        val_ds = DS(cfg=cfg, ann_file=val_ann_file, split_type="valid", comm=comm)
    elif cfg.ds_name == "ch":
        trn_ds = DS(cfg=cfg, split_type="train")
        comm = trn_ds.comm
        val_ds = DS(cfg=cfg, split_type="valid", comm=comm)
    else:
        raise NotImplementedError
    # collate_fn = BatchCollator(cfg, comm)
    collate_fn_trn = BC(cfg, comm, split_type="train")
    trn_dl = get_dataloader(cfg, trn_ds, is_train=True, collate_fn=collate_fn_trn)

    collate_fn_val = BC(cfg, comm, split_type="valid")
    val_dl = get_dataloader(cfg, val_ds, is_train=False, collate_fn=collate_fn_val)

    data = DataWrap(
        path=cfg.misc.tmp_path, train_dl=trn_dl, valid_dl=val_dl, test_dl=None
    )
    print("Done DataLoader")
    return data


if __name__ == "__main__":
    cfg = CN(yaml.safe_load(open("./configs/asrl_qa_cfg.yml")))
    # cfg.train.nw = 0
    cfg.train.bs = 16
    cfg.train.nw = 4
    data = get_data(cfg)

    diter = iter(data.train_dl)
    batch = next(diter)
    from tqdm import tqdm

    for _ in tqdm(data.train_dl):
        pass
