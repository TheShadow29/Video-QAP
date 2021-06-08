# Video-QAP (NAACL21)
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/TheShadow29/Video-QAP/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.7-yellow)
[![Arxiv](https://img.shields.io/badge/Arxiv-2104.03762-purple)](https://arxiv.org/abs/2104.03762)


[**Video Question Answering with Phrases via Semantic Roles**](https://arxiv.org/abs/2104.03762)<br>
[Arka Sadhu](https://theshadow29.github.io/), [Kan Chen](https://kanchen.info/) [Ram Nevatia](https://sites.usc.edu/iris-cvlab/professor-ram-nevatia/)<br>
[NAACL 2021](https://2021.naacl.org/)

Video Question Answering has been studied through the lens of N-way phrase classification. While this eases evaluation, it severely limits its application in the wild. Here, we require the model to generate the answer and we propose a novel evaluation metric using relative scoring and contrastive scoring. We further create ActivityNet-SRL-QA and Charades-SRL-QA. 

## Quickstart

## Quick Start
1. Clone repo:
    ```
    git clone https://github.com/TheShadow29/Video-QAP
    cd Video-QAP
    export ROOT=$(pwd)
    ```
    
1. Setup a new conda environment using the file [vidqap_env.yml](vidqap_env.yml) file provided.
Please refer to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for details on installing conda.

    ```
    MINICONDA_ROOT=[to your Miniconda/Anaconda root directory]
    conda env create -f vidqap_env.yml --prefix $MINICONDA_ROOT/envs/vidqap_pyt
    conda activate vidqap_pyt
    ```
1. See instructions to install fairseq [INSTALL.md](INSTALL.md)

1. To download the datasets ActivityNet-SRL-QA and Charades-SRL-QA see [DATA.md](dcode/README.md)


## Training 

1.  Configuration files are insider [configs](./configs) 
    ```
    cd $ROOT
    python code/main_dist.py "vogqap_asrlqa" --ds_to_use='asrl_qa' --mdl.name='vog_qa' --train.bs=4 --train.epochs=10 --train.lr=1e-4
    ```
    Use one of the models `lqa, mtx_qa, butd_qa, vog_qa`
    
## Evaluation

1. Main evaluation file is `vidqa_code/eval_fn_vidqap.py`. You can use this as a stand-alone file for a separate dataset as well. 
  ```
  cd $ROOT
  python vidqa_code/eval_fn_vidqap.py --pred_file=... --ds_to_use='asrl_qa' --split_type='valid' --met_keys='meteor,rouge,bert_score'
  ```


ToDo:

- [ ] Add more documentation on how to run the models
- [ ] Add pre-trained model weights.
- [ ] Support dataset creation for new caption dataset.


## Acknowledgements:

We thank:
1. @LuoweiZhou: for their codebase on GVD (https://github.com/facebookresearch/grounded-video-description) along with the extracted features for ActivityNet.
2. @antoine77340 for their codebase on S3D pretrained on Howto100M (https://github.com/antoine77340/S3D_HowTo100M) used for feature extraction on Charades.
3. [allennlp](https://github.com/allenai/allennlp) for providing [demo](https://demo.allennlp.org/semantic-role-labeling) and pre-trained model for SRL.
4. [fairseq](https://github.com/pytorch/fairseq) for sequence generation implementation and transformer encoder decoder models.


## Citation
```
@inproceedings{Sadhu2021VideoQA,
  title={Video Question Answering with Phrases via Semantic Roles},
  author={Arka Sadhu and Kan Chen and R. Nevatia},
  booktitle={NAACL},
  year={2021}
}
```



