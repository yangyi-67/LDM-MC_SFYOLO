import torch
import torch.nn.functional as F
from typing import List, Tuple


def _is_detect_conv(module_path: str) -> bool:
    # Heuristic: Detect head conv layers are typically models.yolo.Detect.m[i]
    # We apply dropout on outputs of those convs to induce stochasticity.
    return module_path.endswith('.model.24.m.0') or module_path.endswith('.model.24.m.1') or module_path.endswith('.model.24.m.2')


def register_mc_dropout_hooks(model: torch.nn.Module, p: float = 0.2, only_detect: bool = True) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks that apply dropout to selected module outputs, regardless of training/eval mode.
    Returns list of hook handles to remove() later.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook():
        def hook(_module, _inp, out):
            if isinstance(out, (list, tuple)):
                return tuple(F.dropout(o, p=p, training=True) if torch.is_tensor(o) else o for o in out)
            return F.dropout(out, p=p, training=True)
        return hook

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if only_detect:
                if 'model.' in name and '.m.' in name:
                    # Likely detect head convs reside in Detect.m
                    handles.append(m.register_forward_hook(make_hook()))
            else:
                handles.append(m.register_forward_hook(make_hook()))
    return handles


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N,4], b:[M,4]
    x1 = torch.max(a[:, None, 0], b[None, :, 0])
    y1 = torch.max(a[:, None, 1], b[None, :, 1])
    x2 = torch.min(a[:, None, 2], b[None, :, 2])
    y2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


def aggregate_mc_predictions(preds_per_pass: List[List[torch.Tensor]],
                             iou_match: float = 0.6,
                             min_votes: int = 2,
                             var_thr: float = 1e9) -> List[torch.Tensor]:
    """
    Aggregate predictions across T passes for a batch.
    Input: preds_per_pass is list length T; each element is list length B of tensors [Ni,6] (x1,y1,x2,y2,conf,cls)
    Returns: list length B of tensors [M,6] aggregated (x1,y1,x2,y2,conf,cls) after vote+variance filtering.
    """
    T = len(preds_per_pass)
    B = len(preds_per_pass[0]) if T > 0 else 0
    out: List[torch.Tensor] = []
    device = preds_per_pass[0][0].device if (T > 0 and len(preds_per_pass[0]) > 0 and preds_per_pass[0][0].numel() > 0) else torch.device('cpu')
    for b in range(B):
        # Collect all predictions from T passes
        dets: List[torch.Tensor] = []
        pass_ids: List[torch.Tensor] = []
        for t in range(T):
            if preds_per_pass[t][b].numel() == 0:
                continue
            dets.append(preds_per_pass[t][b])
            pass_ids.append(torch.full((preds_per_pass[t][b].shape[0], 1), t, device=preds_per_pass[t][b].device))
        if not dets:
            out.append(torch.zeros((0, 6), device=device))
            continue
        dets = torch.cat(dets, dim=0)  # [K,6]
        pass_ids = torch.cat(pass_ids, dim=0)  # [K,1]
        boxes = dets[:, :4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        # Sort by conf desc for greedy grouping
        order = torch.argsort(confs, descending=True)
        boxes = boxes[order]
        confs = confs[order]
        clss = clss[order]
        pass_ids = pass_ids[order, 0]
        visited = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)

        agg_boxes: List[torch.Tensor] = []
        agg_confs: List[float] = []
        agg_clss: List[float] = []
        for i in range(boxes.shape[0]):
            if visited[i]:
                continue
            cls_i = clss[i]
            # match same class
            same_cls = (clss == cls_i) & (~visited)
            ious = iou_xyxy(boxes[i:i+1], boxes[same_cls]).squeeze(0)  # [M]
            idxs = torch.where(same_cls)[0][ious >= iou_match]
            if idxs.numel() == 0:
                idxs = torch.tensor([i], device=boxes.device)
            # keep highest conf per pass id to avoid duplicate votes from same pass
            sel_pass = {}
            uniq_idxs: List[int] = []
            for j in idxs.tolist():
                pid = int(pass_ids[j].item())
                if pid not in sel_pass:
                    sel_pass[pid] = j
                else:
                    # keep higher conf
                    if confs[j] > confs[sel_pass[pid]]:
                        sel_pass[pid] = j
            uniq_idxs = list(sel_pass.values())
            votes = len(uniq_idxs)
            if votes >= min_votes:
                group_boxes = boxes[uniq_idxs]
                group_confs = confs[uniq_idxs]
                # weighted average by confidence
                w = (group_confs / (group_confs.sum() + 1e-6)).view(-1, 1)
                mean_box = (group_boxes * w).sum(dim=0)
                mean_conf = float(group_confs.mean().item())
                var_conf = float(group_confs.var(unbiased=False).item())
                if var_conf <= var_thr:
                    agg_boxes.append(mean_box)
                    agg_confs.append(mean_conf)
                    agg_clss.append(float(cls_i.item()))
            visited[idxs] = True

        if len(agg_boxes) == 0:
            out.append(torch.zeros((0, 6), device=device))
            continue
        agg_boxes = torch.stack(agg_boxes, dim=0)
        agg_confs = torch.tensor(agg_confs, device=agg_boxes.device)
        agg_clss = torch.tensor(agg_clss, device=agg_boxes.device)
        out.append(torch.cat([agg_boxes, agg_confs[:, None], agg_clss[:, None]], dim=1))
    return out

