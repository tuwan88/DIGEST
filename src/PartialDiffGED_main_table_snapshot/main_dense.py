import os
import random
import traceback
from contextlib import contextmanager

import numpy as np
import torch

from param_parser import parameter_parser
from trainer_dense import TrainerDense
from utils import tab_printer


def seed_everything(torch_seed):
    random.seed(torch_seed)
    os.environ["PYTHONHASHSEED"] = str(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _imdb_split_test_k_requested(args):
    return int(getattr(args, "test_k_small", 0) or 0) > 0 or int(getattr(args, "test_k_large", 0) or 0) > 0


def _resolve_imdb_split_test_k(args, split_name):
    if split_name == "small":
        split_k = int(getattr(args, "test_k_small", 0) or 0)
    elif split_name == "large":
        split_k = int(getattr(args, "test_k_large", 0) or 0)
    else:
        split_k = 0
    return split_k if split_k > 0 else int(args.test_k)


@contextmanager
def _temporary_args(args, **updates):
    original = {}
    for key, value in updates.items():
        original[key] = getattr(args, key)
        setattr(args, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(args, key, value)


def _run_training_validation(trainer, args, completed_epochs):
    if not bool(getattr(args, "validation_enable", False)):
        return
    val_interval = int(getattr(args, "validation_every_epochs", 1) or 0)
    if val_interval <= 0 or completed_epochs % val_interval != 0:
        return
    val_size = len(getattr(trainer, "val_graphs", []))
    if val_size <= 0:
        print(
            "\n[Validation] Skip epoch {} because the val split is empty.\n".format(completed_epochs),
            flush=True,
        )
        return

    print(
        "\n[Validation] Run val split after epoch {} with reverse decoding and test_k=1.\n".format(completed_epochs),
        flush=True,
    )
    with _temporary_args(
        args,
        reverse_decode_mode="constrained",
        test_k=1,
        testset="val",
    ):
        trainer.score(
            testing_graph_set="val",
            test_k=1,
            top_k_approach=args.topk_approach,
        )


def main():
    trainer = None
    try:
        seed_everything(0)
        args = parameter_parser()
        tab_printer(args)
        trainer = TrainerDense(args)

        if args.model_epoch_start > 0:
            trainer.load(args.model_epoch_start)

        if args.model_train == 1:
            for epoch in range(args.model_epoch_start, args.model_epoch_end):
                trainer.cur_epoch = epoch
                trainer.fit()
                completed_epochs = epoch + 1
                _run_training_validation(trainer, args, completed_epochs)
                if args.save_every_epochs > 0 and completed_epochs % args.save_every_epochs == 0:
                    trainer.save(completed_epochs)
            if args.model_epoch_end == 0 or args.save_every_epochs <= 0 or args.model_epoch_end % args.save_every_epochs != 0:
                trainer.save(args.model_epoch_end)
        else:
            if args.experiment != "test":
                raise ValueError("main_dense.py currently supports --experiment test only.")
            if _imdb_split_test_k_requested(args):
                if str(args.dataset).upper() != "IMDB":
                    raise ValueError("--test-k-small/--test-k-large are only supported for --dataset IMDB.")
                if args.testset == "test":
                    for split_name in ("small", "large"):
                        split_k = _resolve_imdb_split_test_k(args, split_name)
                        print(
                            "\n[IMDB split test-k] Run {} split with test_k={}.".format(
                                split_name,
                                split_k,
                            ),
                            flush=True,
                        )
                        trainer.score(
                            testing_graph_set=split_name,
                            test_k=split_k,
                            top_k_approach=args.topk_approach,
                        )
                else:
                    split_k = _resolve_imdb_split_test_k(args, args.testset)
                    print(
                        "\n[IMDB split test-k] Run {} split with test_k={}.".format(
                            args.testset,
                            split_k,
                        ),
                        flush=True,
                    )
                    trainer.score(
                        testing_graph_set=args.testset,
                        test_k=split_k,
                        top_k_approach=args.topk_approach,
                    )
            else:
                trainer.score(
                    testing_graph_set=args.testset,
                    test_k=args.test_k,
                    top_k_approach=args.topk_approach,
                )
    except Exception as exc:
        print(f"[Fatal][dense] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
