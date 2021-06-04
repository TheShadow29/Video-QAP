"""
Select the model, loss, eval_fn
"""


from vidqa_code.eval_ivd import EvalB
from vidqa_code.mdl_qa import (
    LangQA,
    MTxVidSimple,
    BUTD_Simple,
    VOGSimple,
    LossB,
)


def get_mdl_loss_eval(cfg):
    task = cfg.task
    mdl = None
    assert task == "vid_qa", task
    mdl_type = cfg.mdl.name
    loss = LossB
    evl = EvalB
    if mdl_type == "lqa":
        mdl = LangQA
    elif mdl_type == "mtx_qa":
        mdl = MTxVidSimple
    elif mdl_type == "butd_qa":
        mdl = BUTD_Simple
    elif mdl_type == "vog_qa":
        mdl = VOGSimple
    else:
        raise NotImplementedError

    assert mdl is not None
    return {"mdl": mdl, "loss": loss, "eval": evl}
