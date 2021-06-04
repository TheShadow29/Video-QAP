# Data setup

## Part-1: Directly using ActivityNet-SRL-QA and Charades-SRL-QA

1. Download the relevant dataset files + vocabs used from drive:

    https://drive.google.com/file/d/1PqHYty4D71dakC1a95p4PbAr6WXYYCYq/view?usp=sharing

1. For Anet features:

    File 1: https://drive.google.com/file/d/14GTjt3wuifK6GhaTsRYxMVXhBc4rREk2/view?usp=sharing

    File 2: For RGB-Motion feats See dwn_datasets.sh

1. For Charades SRL Features:

    File 1: https://drive.google.com/file/d/1bJgzyVue8GhbaImjiVhfNgHk3O9d8Apc/view?usp=sharing

After unzipping all files you should have the following structure:

```
data
├── anet
│   ├── anet_detection_vg_fc6_feat_100rois_resized.h5
│   ├── anet_detection_vg_fc6_feat_gt5_rois.h5
│   ├── csv_dir
│   ├── fc6_feat_5rois
│   └── rgb_motion_1d
├── anet_vocab
│   ├── aphr_nosw_vocab_10k.pkl
│   ├── aphr_nosw_vocab.pkl
│   ├── aphr_vocab_10k.pkl
│   ├── aphr_vocab.pkl
│   ├── arg_vocab.pkl
│   ├── qarg_vocab.pkl
│   ├── qword_vocab_old.pkl
│   ├── qword_vocab.pkl
│   └── word_vocab.pkl
├── asrl_vidqap_files
│   ├── anet_captions_all_splits.json
│   ├── anet_ent_cls_bbox_trainval.json
│   ├── trn_srlqa_trim.json
│   └── val_srlqa_trim.json
├── charades
│   ├── s3d_charades_fps1_frm_list.csv
│   └── s3d_charades_rgb_mix5c_ftdir
├── charades_vidqap_files
│   ├── trn_srlqa_trim.json
│   └── val_srlqa_trim.json
└── charades_vocab
    ├── arg_vocab.pkl
    ├── ch_aphr_nosw_vocab_10k.pkl
    ├── ch_aphr_nosw_vocab.pkl
    ├── ch_aphr_vocab_10k.pkl
    ├── ch_aphr_vocab.pkl
    ├── qarg_vocab.pkl
    ├── qword_vocab.pkl
    └── word_vocab.pkl
```


## Part 2 - Create dataset from scratch, and likely for a new captioning dataset

<WIP: Needs some heavy code clean up and refactorization>