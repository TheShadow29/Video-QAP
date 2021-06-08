# VidQAP code

## Using the code

1. `main_dist.py` is the main file used to run the models. Which model to use is specified with `--mdl.name` which can be `lqa, mtx_qa, butd_qa, vog_qa` as specified in `mdl_selector.py`
2. `eval_fn_vidqap.py` is the main file for evaluation. It implements relative-scoring and contrastive-scoring as described in the paper.


## Using evaluation for a separate dataset

1. Currently, we support for argument-based fill-in-the-phrase. But it should be relatively straightforward to extend to any fill-in-the-phrase task
2. You can choose to use only relative-scoring or only contrastive scoring. 
   + The dataset should have question-answer pairs with the position information of the question.
   + Contrastive IDs for validation and test sets are required for contrtastive scoring

