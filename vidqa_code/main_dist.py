"""
Main file for distributed training
"""

import sys


import torch
import fire
from functools import partial

from vidqa_code.extended_config import (
    get_default_cfg,
    get_ch_cfg,
    key_maps,
    CN,
    update_from_dict,
    post_proc_config,
)
from vidqa_code.dat_loader import get_data
from vidqa_code.mdl_selector import get_mdl_loss_eval
from utils.trn_utils import Learner, synchronize

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def get_name_from_inst(inst):
    return inst.__class__.__name__


def learner_init(uid: str, cfg: CN) -> Learner:
    mdl_loss_eval = get_mdl_loss_eval(cfg)
    get_default_net = mdl_loss_eval["mdl"]
    get_default_loss = mdl_loss_eval["loss"]
    get_default_eval = mdl_loss_eval["eval"]

    device = torch.device("cuda")
    data = get_data(cfg)
    comm = data.train_dl.dataset.comm

    mdl = get_default_net(cfg=cfg, comm=comm)

    loss_fn = get_default_loss(cfg, comm)
    loss_fn.to(device)

    eval_fn = get_default_eval(cfg, comm, device)
    eval_fn.to(device)
    opt_fn = partial(torch.optim.Adam, betas=(0.9, 0.99))

    # unfreeze cfg to save the names
    cfg.defrost()
    module_name = mdl
    cfg.mdl_data_names = CN(
        {
            "trn_data": get_name_from_inst(data.train_dl.dataset),
            "val_data": get_name_from_inst(data.valid_dl.dataset),
            "trn_collator": get_name_from_inst(data.train_dl.collate_fn),
            "val_collator": get_name_from_inst(data.valid_dl.collate_fn),
            "mdl_name": get_name_from_inst(module_name),
            "loss_name": get_name_from_inst(loss_fn),
            "eval_name": get_name_from_inst(eval_fn),
            "opt_name": opt_fn.func.__name__,
        }
    )
    cfg.freeze()

    learn = Learner(
        uid=uid,
        data=data,
        mdl=mdl,
        loss_fn=loss_fn,
        opt_fn=opt_fn,
        eval_fn=eval_fn,
        device=device,
        cfg=cfg,
    )
    if cfg.do_dist:
        mdl.to(device)
        mdl = torch.nn.parallel.DistributedDataParallel(
            mdl,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            broadcast_buffers=True,
            find_unused_parameters=True,
        )
    elif cfg.do_dp:
        mdl = torch.nn.DataParallel(mdl)

    mdl = mdl.to(device)

    return learn


def run_job(local_rank, num_proc, func, init_method, backend, cfg):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Initialize the process group.
    world_size = num_proc
    rank = num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend, init_method=init_method, world_size=world_size, rank=rank,
        )
    except Exception as e:
        raise e

    torch.cuda.set_device(local_rank)
    func(cfg)


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            run_job,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)


def main_dist(uid: str, **kwargs):
    """
    uid is a unique identifier for the experiment name
    Can be kept same as a previous run, by default will start executing
    from latest saved model
    **kwargs: allows arbit arguments of cfg to be changed
    """
    # cfg = conf
    assert "ds_to_use" in kwargs
    ds_to_use = kwargs["ds_to_use"]
    assert ds_to_use in ["asrl_qa", "ch_qa"]
    if ds_to_use == "asrl_qa":
        cfg = get_default_cfg()
    elif ds_to_use == "ch_qa":
        cfg = get_ch_cfg()
    else:
        raise NotImplementedError

    num_gpus = torch.cuda.device_count()
    cfg.num_gpus = num_gpus
    cfg.uid = uid
    cfg.cmd = sys.argv
    if num_gpus > 1:
        if "local_rank" in kwargs:
            # We are doing distributed parallel
            cfg.do_dist = True
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(kwargs["local_rank"])
            synchronize()
        else:
            # We are doing data parallel
            cfg.do_dist = False
            # cfg.do_dp = True
    # Update the config file depending on the command line args
    cfg = update_from_dict(cfg, kwargs, key_maps)
    cfg = post_proc_config(cfg)
    # Freeze the cfg, can no longer be changed
    cfg.freeze()
    # print(cfg)
    # Initialize learner
    learn = learner_init(uid, cfg)
    # Train or Test
    if not (cfg.only_val or cfg.only_test or cfg.overfit_batch):
        learn.fit(epochs=cfg.train.epochs, lr=cfg.train.lr)
        if cfg.run_final_val:
            print("Running Final Validation using best model")
            learn.load_model_dict(resume_path=learn.model_file, load_opt=False)
            val_loss, val_acc, _ = learn.validate(
                db={"valid": learn.data.valid_dl}, write_to_file=True
            )
            print(val_loss)
            print(val_acc)
        else:
            pass
    else:
        if cfg.overfit_batch:
            learn.overfit_batch(cfg.train.epochs, 1e-4)
        if cfg.only_val:
            # learn.load_model_dict(resume_path=learn.model_file, load_opt=False)
            if cfg.train.resume_path != "":
                resume_path = cfg.train.resume_path
            else:
                resume_path = learn.model_file
            learn.load_model_dict(resume_path=resume_path)
            val_loss, val_acc, _ = learn.validate(
                db={"valid": learn.data.valid_dl}, write_to_file=True
            )
            print(val_loss)
            print(val_acc)
            # learn.testing(learn.data.valid_dl)
            pass
        if cfg.only_test:
            # learn.testing(learn.data.test_dl)
            learn.load_model_dict(resume_path=learn.model_file, load_opt=False)
            test_loss, test_acc, _ = learn.validate(db=learn.data.test_dl)
            print(test_loss)
            print(test_acc)

    return


if __name__ == "__main__":
    fire.Fire(main_dist)
