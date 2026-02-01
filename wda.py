import math
from typing import Iterable, List, Optional, Sequence, Tuple
import time
import logging

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------- small utils ----------------------
def clamp(img: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(img, lo, hi)


def lp_project(x: torch.Tensor, x0: torch.Tensor, eps: float, p: str = "Linf") -> torch.Tensor:
    p = p.lower()
    if p == "linf":
        return x0 + (x - x0).clamp_(-eps, eps)
    elif p == "l2":
        d = (x - x0).view(x.size(0), -1)
        n = d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(n), eps / n)
        d = (d * scale).view_as(x)
        return x0 + d
    elif p == "l1":
        d = (x - x0).view(x.size(0), -1)
        n = d.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(n), eps / n)
        d = (d * scale).view_as(x)
        return x0 + d
    else:
        raise ValueError("p must be 'Linf', 'L2', or 'L1'")


def lp_normalize(v: torch.Tensor, p: str = "Linf", eps: float = 1e-12) -> torch.Tensor:
    p = p.lower()
    if p == "linf":
        return v.sign()
    elif p == "l2":
        flat = v.view(v.size(0), -1)
        n = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        return (flat / n).view_as(v)
    elif p == "l1":
        flat = v.view(v.size(0), -1)
        n = flat.abs().sum(dim=1, keepdim=True).clamp_min(eps)
        return (flat / n).view_as(v)
    else:
        raise ValueError("p must be 'Linf', 'L2', or 'L1'")


def dlr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the per-sample Difference of Logits Ratio (DLR) loss.

    Matches the definition used in AutoAttack/APGD. Falls back gracefully when the
    number of classes is < 3 by using the smallest available logit for the
    denominator.
    """
    if logits.ndim != 2:
        raise ValueError("dlr_loss expects [B, C] logits tensor")
    x_sorted, ind_sorted = logits.sort(dim=1)
    idx = torch.arange(logits.size(0), device=logits.device)
    top1 = x_sorted[:, -1]
    top2 = x_sorted[:, -2]
    if logits.size(1) >= 3:
        top3 = x_sorted[:, -3]
    else:
        top3 = x_sorted[:, 0]
    correct_on_top = (ind_sorted[:, -1] == labels).float()
    num = logits[idx, labels] - top2 * correct_on_top - top1 * (1.0 - correct_on_top)
    den = (top1 - top3).clamp_min(1e-12)
    return -(num / den)


def attack_loss(logits: torch.Tensor, labels: torch.Tensor, metric: str) -> torch.Tensor:
    """Return per-sample attack loss according to the chosen metric."""
    m = metric.lower()
    if m == "ce":
        return F.cross_entropy(logits, labels, reduction='none')
    if m == "dlr":
        return dlr_loss(logits, labels)
    raise ValueError(f"Unsupported attack loss metric '{metric}' (expected 'ce' or 'dlr')")


@torch.no_grad()
def _topk_nontrue_classes(logits: torch.Tensor, y: torch.Tensor, k: int = 5) -> torch.Tensor:
    B, C = logits.shape
    scores = logits.clone()
    scores[torch.arange(B, device=logits.device), y] = -1e9
    k = int(min(k, max(1, C - 1)))
    return scores.topk(k, dim=1).indices


def _ensure_u_shape(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape/broadcast a flat direction u to image-shaped tensor matching x.

    Accepts:
      - [D] universal vector
      - [B, D] per-sample vectors
      - [B, C, H, W] already image-shaped
    Returns: tensor shaped like x
    """
    if u.dim() == x.dim():
        return u
    B = x.size(0)
    D = int(torch.tensor(x.shape[1:]).prod().item())
    if u.dim() == 1:
        if u.numel() != D:
            raise ValueError(f"u has shape {tuple(u.shape)}; expected flat dim {D}")
        u_img = u.view(1, *x.shape[1:]).expand_as(x)
        return u_img
    elif u.dim() == 2:
        if u.shape[1] != D:
            raise ValueError(f"u has shape {tuple(u.shape)}; expected [B,{D}] with B in [1,{B}]")
        if u.shape[0] == 1:
            u_img = u.view(1, *x.shape[1:]).expand_as(x)
        elif u.shape[0] == B:
            u_img = u.view(B, *x.shape[1:])
        else:
            raise ValueError(f"u batch {u.shape[0]} incompatible with x batch {B}")
        return u_img
    else:
        raise ValueError("Unsupported u rank; expected 1D, 2D, or image-shaped tensor")


# ---------------------- Per-sample closed-form variant ----------------------
def _best_direction_from_grad(g: torch.Tensor, *, p: str = "Linf") -> torch.Tensor:
    """Closed-form maximizer of g^T u s.t. ||u||_p ≤ 1, per sample."""
    if p.lower() == "linf":
        return g.sign()
    elif p.lower() == "l2":
        flat = g.view(g.size(0), -1)
        n = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return (flat / n).view_as(g)
    elif p.lower() == "l1":
        B = g.size(0)
        flat = g.view(B, -1)
        idx = flat.abs().argmax(dim=1)
        out = torch.zeros_like(flat)
        out[torch.arange(B, device=g.device), idx] = flat[torch.arange(B, device=g.device), idx].sign()
        return out.view_as(g)
    else:
        raise ValueError("p must be 'Linf', 'L2', or 'L1'")


# ---------------------- WDA++ ----------------------
def _lp_distance(x: torch.Tensor, y: torch.Tensor, *, p: str = "Linf") -> torch.Tensor:
    if p.lower() == "linf":
        return (x - y).abs().view(x.size(0), -1).max(dim=1).values
    if p.lower() == "l2":
        return (x - y).view(x.size(0), -1).norm(p=2, dim=1)
    return (x - y).view(x.size(0), -1).abs().sum(dim=1)


def _parse_wasserstein_p(p: float | str | None) -> float:
    if p is None:
        return 1.0
    if isinstance(p, str):
        s = p.strip().lower()
        if s in {"inf", "infty", "infinity"}:
            return float("inf")
        try:
            p_val = float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid Wasserstein order '{p}'") from exc
    else:
        p_val = float(p)
    if math.isinf(p_val):
        return float("inf")
    if p_val < 1:
        raise ValueError("Wasserstein order p must be >= 1")
    return p_val


def _allocate_budget(distances: torch.Tensor, eps: float, *, p: float = 1.0) -> torch.Tensor:
    """Fractional knapsack allocation for alpha_i with uniform mu_i = 1/N.

    For finite p: enforce sum_i (1/N) * alpha_i * d_i^p <= eps^p.
    For p = inf: allow alpha_i > 0 only when d_i <= eps.
    """
    N = distances.numel()
    alpha = torch.zeros_like(distances)
    finite_mask = torch.isfinite(distances)
    if not finite_mask.any():
        return alpha
    if math.isinf(p):
        eligible = finite_mask & (distances <= eps)
        alpha[eligible] = 1.0
        return alpha
    idx_all = torch.nonzero(finite_mask, as_tuple=False).squeeze(-1)
    d_sorted, order = torch.sort(distances[idx_all])
    remaining = float(eps) ** float(p)
    for rel_idx, d_i in zip(order, d_sorted):
        if remaining <= 0:
            break
        if d_i <= 0:
            alpha[idx_all[rel_idx]] = 1.0
            continue
        cost_full = (float(d_i.item()) ** float(p)) / float(N)
        if remaining >= cost_full:
            alpha[idx_all[rel_idx]] = 1.0
            remaining -= cost_full
        else:
            alpha[idx_all[rel_idx]] = max(0.0, remaining / cost_full)
            remaining = 0.0
            break
    return alpha


def run_wda_pp(
    model,
    x_orig,
    y_orig,
    *,
    eps: float,
    norm: str = "Linf",
    wasserstein_p: float | str | None = 1,
    steps: int = 50,
    bs: int = 64,
    topk: int | None = None,
    srchiter: int = 10,
    step_size: float | None = None,
    device: str | torch.device = "cuda",
    verbose: bool = True,
    return_history: bool = True,
    select_metric: str | None = "logit_raw",
    loss_metric: str = "ce",
    random_start: bool = False,
):
    """Run WDA++ attack (closest-flip oracle + knapsack allocation)."""
    dev = torch.device(device)
    x = x_orig.to(dev)
    y = y_orig.to(dev)
    p_order = _parse_wasserstein_p(wasserstein_p)

    with torch.no_grad():
        preds = []
        for i in range(0, x.size(0), bs):
            preds.append(model(x[i:i + bs]).argmax(1))
        y_pred = torch.cat(preds, dim=0)
    clean_acc = float((y_pred == y).float().mean().item())
    if verbose:
        logging.info(
            "--------------------------  running %s-attack (wda++) with epsilon %.5f --------------------------",
            norm,
            eps,
        )
        logging.info("WDA++ Wasserstein order: %s", "inf" if math.isinf(p_order) else f"{p_order:g}")
        logging.info("initial accuracy: %.2f%%", clean_acc * 100)

    robust_flags = (y_pred == y).detach().clone()
    num_robust = int(robust_flags.sum().item())
    x_adv = x.detach().clone()
    global_acc_steps = (
        torch.zeros(steps + 1, num_robust, dtype=torch.float32, device=x.device)
        if return_history and num_robust > 0
        else None
    )
    d_all = torch.full((x.shape[0],), float("inf"), device=x.device)

    robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
    if num_robust > 1:
        robust_lin_idcs.squeeze_()

    n_batches = int(np.ceil(max(1, num_robust) / bs))
    for batch_idx in range(n_batches):
        start = batch_idx * bs
        end = min((batch_idx + 1) * bs, num_robust)
        ids = robust_lin_idcs[start:end]
        if ids.numel() == 0:
            continue
        if len(ids.shape) > 1:
            ids = ids.squeeze(-1)
        xb = x[ids].clone()
        yb = y[ids].clone()

        effective_select_metric = (select_metric or "logit_raw")
        atk = WDAPP(
            model,
            norm=norm,
            steps=steps,
            alpha=step_size if step_size is not None else (eps / max(1, steps)),
            topk=topk if topk is not None else 0,
            srchiter=srchiter,
            select_metric=effective_select_metric,
        )
        xb_adv, d_b, _, curve = atk.perturb(xb, yb, return_history=return_history)
        if return_history and curve is not None:
            curve_t = torch.tensor(curve, device=x.device, dtype=torch.float32)
            if curve_t.numel() < steps + 1:
                pad_val = curve_t[-1] if curve_t.numel() > 0 else torch.tensor(1.0, device=x.device)
                pad = pad_val.repeat(steps + 1 - curve_t.numel())
                curve_t = torch.cat([curve_t, pad], dim=0)
            elif curve_t.numel() > steps + 1:
                curve_t = curve_t[: steps + 1]
            global_acc_steps[:, start:end] = curve_t.view(-1, 1).expand(steps + 1, end - start)

        with torch.no_grad():
            out = model(xb_adv).argmax(1)
            false_batch = (out != yb)
            x_adv[ids[false_batch]] = xb_adv[false_batch].detach().to(x_adv.device)
            robust_flags[ids[false_batch]] = False
            d_all[ids] = d_b

        if verbose:
            logging.info(
                "wda++ - %d/%d - %d out of %d successfully perturbed",
                batch_idx + 1,
                n_batches,
                int(false_batch.sum().item()),
                xb.shape[0],
            )

    alpha_all = _allocate_budget(d_all, eps, p=p_order)
    if math.isinf(p_order):
        x_adv = clamp(lp_project(x_adv, x, eps, p=norm), 0.0, 1.0)
    robust_accuracy = float(robust_flags.float().mean().item())
    with torch.no_grad():
        clean_correct = y_pred.eq(y).float()
        adv_preds = []
        for i in range(0, x_adv.size(0), bs):
            adv_preds.append(model(x_adv[i:i + bs]).argmax(1))
        adv_correct = torch.cat(adv_preds, dim=0).eq(y).float()
        mixture_robust = float(((1.0 - alpha_all) * clean_correct + alpha_all * adv_correct).mean().item())
        if math.isinf(p_order):
            robust_accuracy = float(adv_correct.mean().item())
        active_mask = torch.isfinite(d_all) & (alpha_all > 0)
        if math.isinf(p_order):
            spent = float(d_all[active_mask].max().item()) if active_mask.any() else 0.0
        else:
            if active_mask.any():
                cost_sum = (alpha_all[active_mask] * d_all[active_mask].pow(p_order)).sum().item() / max(1, x.shape[0])
                spent = float(cost_sum ** (1.0 / p_order))
            else:
                spent = 0.0
    if verbose:
        logging.info("robust accuracy after wda++: %.2f%%", robust_accuracy * 100)
        logging.info("WDA++ mixture robust accuracy: %.2f%%", mixture_robust * 100)
        logging.info("WDA++ budget spent (W_p): %.6f (eps=%.6f)", spent, eps)

    if return_history and global_acc_steps is not None:
        curve_out = global_acc_steps.mean(dim=1).detach().cpu().tolist()
    else:
        curve_out = None
    aux = {
        "alpha": alpha_all.detach().cpu(),
        "distances": d_all.detach().cpu(),
        "mixture_robust": mixture_robust,
        "spent": spent,
        "wasserstein_p": p_order,
    }
    return x_adv, curve_out, aux


class WDAPP:
    """WDA++"""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        norm: str = "Linf",
        steps: int = 50,
        alpha: float = 0.01,
        topk: int = 5,
        srchiter: int = 10,
        select_metric: str = "logit_raw",
        clip: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.model = model
        self.norm = norm
        self.steps = int(steps)
        self.alpha = float(alpha)
        self.topk = int(topk)
        self.srchiter = int(max(1, srchiter))
        metric = (select_metric or "logit_raw").lower()
        if metric not in {"ce", "dlr", "logit", "logit_raw"}:
            raise ValueError("select_metric must be 'ce', 'dlr', 'logit', or 'logit_raw'")
        self.select_metric = metric
        self.clip = clip

    def _score_candidates(self, logits: torch.Tensor, y: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        if self.select_metric == "logit_raw":
            return logits[torch.arange(logits.size(0), device=logits.device), j]
        if self.select_metric == "logit":
            idx = torch.arange(logits.size(0), device=logits.device)
            return logits[idx, j] - logits[idx, y]
        if self.select_metric == "dlr":
            return dlr_loss(logits, y)
        return F.cross_entropy(logits, y, reduction="none")

    def perturb(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        return_history: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[float]]]:
        """Return (x_adv, d_i, flipped_mask, curve)."""
        self.model.eval()
        x0 = x.detach()
        B = x0.size(0)
        x_iter = x0.clone()

        with torch.no_grad():
            logits0 = self.model(x0)
        k_use = self.topk if self.topk > 0 else (logits0.size(1) - 1)
        cand_js = _topk_nontrue_classes(logits0, y, k=k_use)
        k_eff = int(cand_js.size(1))

        active = torch.ones(B, dtype=torch.bool, device=x0.device)
        flipped = torch.zeros(B, dtype=torch.bool, device=x0.device)
        x_pre = x0.clone()
        x_flip = x0.clone()

        curve = []
        if return_history:
            with torch.no_grad():
                pred0 = logits0.argmax(1)
                curve.append(float((pred0 == y).float().mean().item()))

        for _ in range(self.steps):
            if not active.any():
                break
            x_iter = x_iter.detach()
            x_iter.requires_grad_(True)
            logits = self.model(x_iter)

            grads = []
            for t in range(k_eff):
                j_t = cand_js[:, t]
                v = torch.zeros_like(logits)
                idx = torch.arange(B, device=logits.device)
                v[idx, j_t] = 1.0
                v[idx, y] -= 1.0
                g_t = torch.autograd.grad(
                    outputs=logits,
                    inputs=x_iter,
                    grad_outputs=v,
                    retain_graph=True if t < k_eff - 1 else False,
                    create_graph=False,
                )[0]
                grads.append(g_t.detach())

            trial_list = []
            for g_t in grads:
                u_t = _best_direction_from_grad(g_t, p=self.norm)
                x_trial = clamp(x_iter + self.alpha * u_t, *self.clip)
                trial_list.append(x_trial)

            with torch.no_grad():
                trial_logits = self.model(torch.cat(trial_list, dim=0))
            scores = []
            for t in range(k_eff):
                logits_t = trial_logits[t * B:(t + 1) * B]
                scores_t = self._score_candidates(logits_t, y, cand_js[:, t])
                scores.append(scores_t.unsqueeze(1))
            scores = torch.cat(scores, dim=1)
            best_idx = scores.argmax(dim=1)

            x_next = x_iter.detach().clone()
            for t in range(k_eff):
                mask = best_idx == t
                if mask.any():
                    x_next[mask] = trial_list[t][mask]

            with torch.no_grad():
                preds = self.model(x_next).argmax(1)
            newly_flipped = active & preds.ne(y)
            if newly_flipped.any():
                x_pre[newly_flipped] = x_iter.detach()[newly_flipped]
                x_flip[newly_flipped] = x_next.detach()[newly_flipped]
                flipped = flipped | newly_flipped
                active = active & ~newly_flipped

            if active.any():
                x_iter = x_iter.detach()
                x_iter[active] = x_next.detach()[active]

            if return_history:
                curve.append(float((preds == y).float().mean().item()))

        x_adv = x0.clone()
        d_i = torch.full((B,), float("inf"), device=x0.device)
        if flipped.any():
            idxs = torch.nonzero(flipped, as_tuple=False).squeeze(-1)
            x_pre_f = x_pre[idxs]
            x_flip_f = x_flip[idxs]
            y_f = y[idxs]
            t_low = torch.zeros(idxs.size(0), device=x0.device)
            t_high = torch.ones(idxs.size(0), device=x0.device)
            delta = x_flip_f - x_pre_f

            for _ in range(self.srchiter):
                t_mid = (t_low + t_high) / 2.0
                t_view = t_mid.view(-1, *([1] * (x_pre_f.ndim - 1)))
                x_mid = clamp(x_pre_f + t_view * delta, *self.clip)
                with torch.no_grad():
                    pred_mid = self.model(x_mid).argmax(1)
                is_flip = pred_mid.ne(y_f)
                t_high = torch.where(is_flip, t_mid, t_high)
                t_low = torch.where(is_flip, t_low, t_mid)

            t_view = t_high.view(-1, *([1] * (x_pre_f.ndim - 1)))
            x_refined = clamp(x_pre_f + t_view * delta, *self.clip)
            x_adv[idxs] = x_refined
            d_i[idxs] = _lp_distance(x_refined, x0[idxs], p=self.norm)

        return x_adv.detach(), d_i.detach(), flipped.detach(), curve


# ---------------------- WDA ----------------------
def _margin_grad_for_j(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Compute ∇_x (z_j − z_y) for per-sample j (shape [B])."""
    x = x.detach().requires_grad_(True)
    logits = model(x)
    B = x.size(0)
    v = torch.zeros_like(logits)
    idx = torch.arange(B, device=x.device)
    v[idx, j] = 1.0
    v[idx, y] -= 1.0
    g = torch.autograd.grad(outputs=logits, inputs=x, grad_outputs=v, retain_graph=False, create_graph=False)[0]
    return g.detach()


def _select_fixed_j(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    norm: str = "Linf",
    topk: int | None = None,
    steps: int = 1,
    alpha: float | None = None,
    metric: str = "ce",
    no_probe: bool = False,
    clip: Tuple[float, float] = (0.0, 1.0),
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Select fixed j* per sample via short probe toward each candidate.

    For each candidate j in top-k impostors (or all), simulate `steps` fixed-j margin
    updates with step `alpha` (projecting to the ε-ball around x), score by CE/DLR,
    the magnitude of the margin `(z_j - z_y)`, or the raw impostor logit.
    and keep the j with the highest score.
    """
    device = x.device
    B = x.size(0)

    if batch_size is not None and batch_size > 0 and batch_size < B:
        outs = []
        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            j_chunk = _select_fixed_j(
                model,
                x[start:end],
                y[start:end],
                eps=eps,
                norm=norm,
                topk=topk,
                steps=steps,
                alpha=alpha,
                metric=metric,
                no_probe=no_probe,
                clip=clip,
                batch_size=None,
            )
            outs.append(j_chunk)
        return torch.cat(outs, dim=0)
    with torch.no_grad():
        logits0 = model(x)
        C = logits0.size(1)
    if topk is None:
        topk = C - 1
    cand_js = _topk_nontrue_classes(logits0, y, k=topk)  # [B, k]

    best_score = torch.full((B,), -1e9, device=device)
    best_j = cand_js[:, 0].clone()

    if no_probe or steps <= 0 or (alpha is not None and alpha == 0):
        n = norm.lower()
        best_dual = torch.full((B,), -1e9, device=device)
        for t in range(cand_js.size(1)):
            j_t = cand_js[:, t]
            g_t = _margin_grad_for_j(model, x, y, j_t)
            flat = g_t.view(B, -1).abs()
            if n == 'linf':
                dual = flat.sum(dim=1)
            elif n == 'l2':
                dual = flat.norm(p=2, dim=1)
            else:
                dual = flat.max(dim=1).values
            take = dual > best_dual
            best_dual = torch.where(take, dual, best_dual)
            best_j = torch.where(take, j_t, best_j)
        return best_j.detach()

    if alpha is None:
        alpha = (2.0 * eps / max(1, steps)) if norm.lower() == 'linf' else (1.0 * eps / max(1, steps))

    metric = (metric or 'ce').lower()

    def score_logits_ce_dlr(lg: torch.Tensor) -> torch.Tensor:
        if metric == 'dlr':
            x_sorted, ind_sorted = lg.sort(dim=1)
            ind = (ind_sorted[:, -1] == y).float()
            uidx = torch.arange(lg.shape[0], device=lg.device)
            num = lg[uidx, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
            den = (x_sorted[:, -1] - x_sorted[:, -3]).clamp_min(1e-12)
            return -(num / den)
        elif metric == 'ce':
            return F.cross_entropy(lg, y, reduction='none')
        else:
            raise ValueError("score_logits_ce_dlr called with unsupported metric")

    for t in range(cand_js.size(1)):
        j_t = cand_js[:, t]
        x_trial = x.clone().detach()
        for _ in range(max(1, steps)):
            g = _margin_grad_for_j(model, x_trial, y, j_t)
            u = _best_direction_from_grad(g, p=norm)
            x_trial = clamp(lp_project(x_trial + alpha * u, x, eps, p=norm), *clip)
        with torch.no_grad():
            logits = model(x_trial)
            if metric == 'logit':
                idx = torch.arange(logits.size(0), device=logits.device)
                s = (logits[idx, j_t] - logits[idx, y]).abs()
            elif metric == 'logit_raw':
                idx = torch.arange(logits.size(0), device=logits.device)
                s = logits[idx, j_t]
            else:
                s = score_logits_ce_dlr(logits)
        take = s > best_score
        best_score = torch.where(take, s, best_score)
        best_j = torch.where(take, j_t, best_j)
    return best_j.detach()

def run_wda(
    model,
    x_orig,
    y_orig,
    *,
    eps: float,
    norm: str = "Linf",
    steps: int = 50,
    bs: int = 64,
    topk: int | None = None,
    probe_steps: int = 1,
    probe_alpha: float | None = None,
    no_probe: bool = False,
    step_size: float | None = None,
    device: str | torch.device = "cuda",
    verbose: bool = True,
    return_history: bool = True,
    loss_metric: str = "ce",
    select_metric: str | None = 'logit_raw',
):
    """Run WDA attack with j* selected per sample by probing candidates.

    Returns x_adv, robust_curve, j_star.
    """
    dev = torch.device(device)
    x = x_orig.to(dev)
    y = y_orig.to(dev)

    # Initial accuracy (batched to avoid OOM)
    with torch.no_grad():
        preds = []
        for i in range(0, x.size(0), bs):
            preds.append(model(x[i:i+bs]).argmax(1))
        y_pred = torch.cat(preds, dim=0)
    clean_acc = float((y_pred == y).float().mean().item())
    if verbose:
        logging.info(f"--------------------------  running {norm}-attack with epsilon {eps:.5f} --------------------------")
        logging.info(f"initial accuracy: {clean_acc*100:.2f}%")

    robust_flags = (y_pred == y).detach().clone()
    num_robust = int(robust_flags.sum().item())
    x_adv = x.detach().clone()
    global_acc_steps = torch.zeros(steps + 1, num_robust, dtype=torch.float32, device=x.device) if return_history and num_robust > 0 else None

    robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
    if num_robust > 1:
        robust_lin_idcs.squeeze_()

    n_batches = int(np.ceil(max(1, num_robust) / bs))
    all_j_star = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
    for batch_idx in range(n_batches):
        start = batch_idx * bs
        end = min((batch_idx + 1) * bs, num_robust)
        ids = robust_lin_idcs[start:end]
        if ids.numel() == 0:
            continue
        if len(ids.shape) > 1:
            ids = ids.squeeze(-1)
        xb = x[ids].clone()
        yb = y[ids].clone()

        effective_select_metric = (select_metric or 'logit_raw')

        atk = WDA(
            model,
            eps=eps,
            norm=norm,
            steps=steps,
            alpha=step_size,
            select_topk=topk,
            select_steps=probe_steps,
            select_alpha=probe_alpha,
            select_metric=effective_select_metric,
            select_batch_size=bs,
            no_probe=no_probe,
            loss_metric=loss_metric,
        )
        if return_history:
            xb_adv, curve, j_star = atk.perturb(xb, yb, return_history=True, verbose=verbose)
            curve_t = torch.tensor(curve, device=x.device, dtype=torch.float32)
            global_acc_steps[:, start:end] = curve_t.view(-1, 1).expand(steps + 1, end - start)
        else:
            xb_adv, j_star = atk.perturb(xb, yb, return_history=False, verbose=verbose)

        all_j_star[ids] = j_star
        with torch.no_grad():
            out = model(xb_adv).argmax(1)
            false_batch = (out != yb)
            x_adv[ids[false_batch]] = xb_adv[false_batch].detach().to(x_adv.device)
            robust_flags[ids[false_batch]] = False

        if verbose:
            logging.info(f"{atk} - {batch_idx+1}/{n_batches} - {false_batch.sum().item()} out of {xb.shape[0]} successfully perturbed")

    robust_accuracy = float(robust_flags.float().mean().item())
    if verbose:
        logging.info(f"robust accuracy after attack: {robust_accuracy*100:.2f}%")

    if return_history and global_acc_steps is not None:
        curve_out = global_acc_steps.mean(dim=1).detach().cpu().tolist()
    else:
        curve_out = None
    return x_adv, curve_out, all_j_star.detach().cpu()


class WDA:
    """WDA"""

    def __init__(
        self,
        model: torch.nn.Module,
        eps: float,
        *,
        norm: str = 'Linf',
        steps: int = 50,
        alpha: Optional[float] = None,
        select_topk: Optional[int] = None,
        select_steps: int = 1,
        select_alpha: Optional[float] = None,
        select_metric: str = 'ce',
        select_batch_size: Optional[int] = None,
        clip: Tuple[float, float] = (0.0, 1.0),
        no_probe: bool = False,
        loss_metric: str = 'ce',
        step_schedule: Optional[Sequence[float]] = None,
    ) -> None:
        self.model = model
        self.eps = float(eps)
        self.norm = norm
        self.steps = int(steps)
        self.alpha = alpha
        self.select_topk = select_topk
        self.select_steps = int(select_steps)
        self.select_alpha = select_alpha
        metric = (select_metric or 'logit_raw').lower()
        if metric not in {'ce', 'dlr', 'logit', 'logit_raw'}:
            raise ValueError("select_metric must be 'ce', 'dlr', 'logit', or 'logit_raw'")
        self.select_metric = metric
        self.select_batch_size = select_batch_size
        self.clip = clip
        self.no_probe = no_probe
        lm = loss_metric.lower()
        if lm not in {'ce', 'dlr'}:
            raise ValueError("loss_metric must be 'ce' or 'dlr'")
        self.loss_metric = lm
        self.step_schedule = list(step_schedule) if step_schedule is not None else None

    def perturb(self,
                x: torch.Tensor,
                y: torch.Tensor,
                *,
                x_init: Optional[torch.Tensor] = None,
                return_history: bool = False,
                verbose: bool = False):
        start_time = time.time()
        self.model.eval()
        x0 = x.detach()

        if x_init is not None:
            x_init = x_init.detach().to(x0.device)
            x_init = clamp(lp_project(x_init, x0, self.eps, p=self.norm), *self.clip)
            select_base = x_init
        else:
            select_base = x0

        j_star = _select_fixed_j(
            self.model, select_base, y,
            eps=self.eps, norm=self.norm,
            topk=self.select_topk, steps=self.select_steps,
            alpha=self.select_alpha, metric=self.select_metric,
            no_probe=self.no_probe,
            clip=self.clip,
            batch_size=self.select_batch_size,
        )

        if self.step_schedule is not None and self.alpha is not None:
            raise ValueError("Cannot set both 'alpha' and 'step_schedule'.")

        schedule = None
        base_alpha: Optional[float] = None
        if self.step_schedule is not None:
            if len(self.step_schedule) != self.steps:
                raise ValueError("step_schedule length must equal the number of attack steps")
            schedule = [float(s) for s in self.step_schedule]

        if schedule is None:
            if self.alpha is None:
                base_alpha = (2.5 * self.eps / self.steps) if self.norm.lower() == 'linf' else (1.5 * self.eps / self.steps)
            else:
                base_alpha = float(self.alpha)

        x_adv = select_base.clone()
        curve = []
        if return_history:
            with torch.no_grad():
                pred0 = self.model(x_adv).argmax(1)
                curve.append(float((pred0 == y).float().mean().item()))

        logits = self.model(x_adv)
        loss_vals = float(attack_loss(logits, y, self.loss_metric).mean().item())
        pred = logits.argmax(1)
        robust = float((pred == y).float().mean().item())
        logging.info(f"[wda] init loss:{loss_vals:.6f} robust accuracy: {robust:.6f} - time: {time.time() - start_time:.2f}s")

        loss_best = None
        metric_label = 'loss' if self.loss_metric == 'ce' else 'dlr'
        for it in range(self.steps):
            g = _margin_grad_for_j(self.model, x_adv, y, j_star)
            u = _best_direction_from_grad(g, p=self.norm)
            if schedule is not None:
                step_alpha = schedule[it]
            else:
                if base_alpha is None:
                    raise RuntimeError("step size not initialized")
                step_alpha = base_alpha
            x_adv = clamp(lp_project(x_adv + step_alpha * u, x0, self.eps, p=self.norm), *self.clip)

            if return_history or verbose:
                with torch.no_grad():
                    logits = self.model(x_adv)
                    loss_vals = attack_loss(logits, y, self.loss_metric)
                    if loss_best is None:
                        loss_best = loss_vals.detach().clone()
                    else:
                        loss_best = torch.maximum(loss_best, loss_vals.detach())
                    avg_metric = float(loss_vals.mean().item())
                    best_metric_mean = float(loss_best.mean().item())
                    pred = logits.argmax(1)
                    robust = float((pred == y).float().mean().item())
                    if self.norm.lower() == 'linf':
                        d = (x_adv - x0).abs().view(x0.size(0), -1).max(dim=1).values
                    elif self.norm.lower() == 'l2':
                        d = (x_adv - x0).view(x0.size(0), -1).norm(p=2, dim=1)
                    else:
                        d = (x_adv - x0).view(x0.size(0), -1).abs().sum(dim=1)
                    d_mean = float(d.mean().item())
                    d_max = float(d.max().item())
                    if verbose:
                        
                        logging.info(
                            f"[wda] iteration: {it+1}/{self.steps} - best {metric_label}: {best_metric_mean:.6f} - avg {metric_label}: {avg_metric:.6f} - "
                            f"robust accuracy: {robust:.6f} - ||delta|| mean: {d_mean:.5f} max: {d_max:.5f} - step size: {step_alpha:.5f} - time: {time.time() - start_time:.2f}s"
                        )
                    if return_history:
                        curve.append(robust)

        x_adv = x_adv.detach()
        if return_history:
            return x_adv, curve, j_star
        return x_adv, j_star
