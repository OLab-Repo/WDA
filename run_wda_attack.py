import argparse
import logging
import time

import numpy as np
import torch
from adaptive_auto_attack.Adaptive_Auto_Attack_main import Adaptive_Auto_white_box_attack
from robustbench.data import load_cifar10, load_cifar100, load_imagenet
from robustbench.utils import clean_accuracy, load_model

from utils import setup_logger
from wda import run_wda, run_wda_pp

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def budget_audit(x_orig, x_adv, eps, norm="Linf", tol=1e-6):
    if norm == "Linf":
        d = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
    elif norm == "L2":
        d = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(1).sqrt()
    elif norm == "L1":
        d = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(1)
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    max_d = float(d.max().item())
    num_over = int((d > eps + tol).sum().item())
    if num_over > 0:
        logging.warning(
            f"Budget violation: {num_over} / {x_orig.shape[0]} exceeded eps={eps:.6f} (max {max_d:.6f})"
        )
    else:
        logging.info(f"Budget OK: max distance {max_d:.6f} within eps={eps:.6f}")
    return max_d, num_over


def main():
    parser = argparse.ArgumentParser(description="Run WDA attacks")
    parser.add_argument("--eps", type=float, default=8/255, help="perturbation budget")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--slicing", type=int, default=None, help="number of test examples")
    parser.add_argument("--budget_tol", type=float, default=1e-6, help="tolerance for budget audit")
    parser.add_argument("--debug_budget", action="store_true", help="toggle epsilon budget checks")
    parser.add_argument("--step_size_coeff", type=float, default=0.02, help="step-size coeff for WDA attacks")
    parser.add_argument("--steps", type=int, default=20, help="number of attack iterations")
    parser.add_argument("--srchiter", type=int, default=10, help="bisection steps for radial refinement")
    parser.add_argument(
        "--wasserstein_p",
        type=str,
        default="1",
        help="Wasserstein order for WDA++ budget (e.g., 1, 2, inf)",
    )
    parser.add_argument(
        "--select_metric",
        type=str,
        default="logit_raw",
        choices=["logit_raw", "logit", "ce", "dlr"],
        help="candidate selection metric",
    )
    parser.add_argument(
        "--loss_metric",
        type=str,
        default="dlr",
        choices=["ce", "dlr"],
        help="logging loss metric",
    )
    parser.add_argument("--norm", type=str, default="Linf", choices=["Linf", "L2", "L1"], help="attack norm")
    parser.add_argument("--wda", action="store_true", help="run WDA attack")
    parser.add_argument("--wda_pp", action="store_true", help="run WDA++ attack")
    parser.add_argument("--topk", type=int, default=9, help="top-k competitor classes for WDA")
    parser.add_argument("--aaa", action="store_true", help="run Adaptive Auto Attack instead")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--model_name", type=str, default="Sehwag2021Proxy")
    args = parser.parse_args()

    eps = float(args.eps)
    setup_logger("WDA")
    torch.manual_seed(32)
    np.random.seed(32)
    logging.info("Using norm: %s, eps: %.6f", args.norm, eps)

    start = time.time()
    logging.info("Loading model and data...")

    model = load_model(model_name=args.model_name, threat_model=args.norm, dataset=args.dataset)
    if args.dataset == "cifar10":
        x_test, y_test = load_cifar10()
    elif args.dataset == "imagenet":
        x_test, y_test = load_imagenet()
    else:
        x_test, y_test = load_cifar100()

    if args.slicing > 0:
        x_test = x_test[: args.slicing].to(DEVICE)
        y_test = y_test[: args.slicing].to(DEVICE)
    else:
        x_test = x_test.to(DEVICE)
        y_test = y_test.to(DEVICE)

    logging.info("Length of test set: %d", x_test.shape[0])
    model.to(DEVICE).eval()

    clean = clean_accuracy(model, x_test, y_test, batch_size=128, device=DEVICE)
    logging.info("Clean accuracy: %.2f%%", clean * 100.0)

    x_adv = None
    if args.aaa:
        logging.info("Running Adaptive Auto Attack (A$^3$-WDA)")
        Adaptive_Auto_white_box_attack(
            model=model,
            device=torch.device(DEVICE),
            eps=eps,
            is_random=True,
            batch_size=args.bs,
            average_num=1000,
            model_name=args.model_name,
            data_set=args.dataset,
            Lnorm=args.norm,
            attack_method="wda",
            wda_options={
                "restarts": 13,
                "no_probe": False,
                "bs": args.bs,
                "topk": args.topk,
                "loss_metric": "ce",
                "verbose": True,
            },
            slicing=None,
        )
    elif args.wda:
        logging.info("Running WDA")
        x_adv, _, _ = run_wda(
            model,
            x_test,
            y_test,
            eps=eps,
            norm=args.norm,
            steps=args.steps,
            bs=args.bs,
            device=DEVICE,
            verbose=True,
            return_history=True,
            topk=args.topk,
            step_size=args.step_size_coeff,
            probe_steps=10,
            loss_metric=args.loss_metric,
            select_metric=args.select_metric,
        )
    elif args.wda_pp:
        logging.info("Running WDA++ attack")
        x_adv, _, _ = run_wda_pp(
            model,
            x_test,
            y_test,
            eps=eps,
            norm=args.norm,
            wasserstein_p=args.wasserstein_p,
            steps=args.steps,
            bs=args.bs,
            device=DEVICE,
            verbose=True,
            return_history=True,
            topk=args.topk,
            srchiter=args.srchiter,
            step_size=args.step_size_coeff,
            select_metric=args.select_metric,
            loss_metric=args.loss_metric,
        )
    else:
        raise NotImplementedError("Specify either --wda_pp, --wda or --aaa.")

    if x_adv is not None:
        if args.debug_budget:
            max_d, num_over = budget_audit(x_test, x_adv, eps, norm=args.norm, tol=args.budget_tol)
            logging.info("Budget audit: max=%.6f, violations=%d", max_d, num_over)

        acc_adv = clean_accuracy(model, x_adv, y_test, batch_size=256, device=DEVICE)
        logging.info("Adversarial accuracy: %.2f%%", acc_adv * 100.0)

    logging.info("Total runtime: %.2f seconds", time.time() - start)


if __name__ == "__main__":
    main()
