"""Snap a stylised image back onto the render it came from.

Image models rarely keep the object pixel-exact: it drifts or gets rescaled a
little. We estimate the foreground of the stylised image and fit a
scale+translate so its bounding box matches the render mask, then report IoU.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def foreground_mask(img, thresh=28):
    """Foreground = everything not connected to the image border through
    background-coloured pixels. img: PIL RGB. Returns bool (H,W)."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    close = np.abs(a - bg).max(-1) <= thresh
    lab, _ = ndi.label(close)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    fg = ~np.isin(lab, edge)
    fg = ndi.binary_opening(fg, iterations=2)
    lab, n = ndi.label(fg)
    if n > 1:  # drop specks (stray shadows / artefacts)
        sizes = ndi.sum(fg, lab, np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= 0.01 * sizes.max()) + 1
        fg = np.isin(lab, keep)
    return fg


def _bbox(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def align_to_mask(img, target, bg=(255, 255, 255), max_scale=1.2):
    """img: PIL RGB (square); target: bool render mask at the same size.
    Returns (aligned PIL image, fg mask bool, info dict)."""
    fg = foreground_mask(img)
    info = {"iou_before": _iou(fg, target), "aligned": False}
    info["iou"] = info["iou_before"]
    if fg.sum() < 16 or target.sum() < 16:
        return img, fg, info
    sx0, sy0, sx1, sy1 = _bbox(fg)
    tx0, ty0, tx1, ty1 = _bbox(target)
    a = (sx1 - sx0) / (tx1 - tx0)
    e = (sy1 - sy0) / (ty1 - ty0)
    if not (1 / max_scale <= a <= max_scale and 1 / max_scale <= e <= max_scale):
        return img, fg, info
    coef = (a, 0, sx0 - tx0 * a, 0, e, sy0 - ty0 * e)
    warped = img.transform(img.size, Image.AFFINE, coef, resample=Image.BICUBIC, fillcolor=tuple(bg))
    fg2 = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).transform(
        img.size, Image.AFFINE, coef, resample=Image.BILINEAR)) > 127
    iou2 = _iou(fg2, target)
    if iou2 > info["iou_before"] + 0.002:
        info.update(aligned=True, iou=iou2, scale=(round(1 / a, 4), round(1 / e, 4)),
                    shift=(round(float(tx0 - sx0 / a), 1), round(float(ty0 - sy0 / e), 1)))
        return warped, fg2, info
    return img, fg, info


def match_to_views(images, renders):
    """Assign painted images (any order / any names) to views by silhouette shape.

    images, renders: lists of PIL images. Returns (assignment, scores) where
    assignment[i] = index of the image used for view i (or None) and scores[i] its IoU.
    """
    from scipy.optimize import linear_sum_assignment
    S = 256
    tmasks = [foreground_mask(r.convert("RGB").resize((S, S), Image.BILINEAR)) for r in renders]
    cost = np.zeros((len(renders), len(images)))
    for j, im in enumerate(images):
        im = im.convert("RGB")
        side = max(im.size)  # pad to square (views are square)
        sq = Image.new("RGB", (side, side), (255, 255, 255))
        sq.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
        sq = sq.resize((S, S), Image.BILINEAR)
        for i, t in enumerate(tmasks):
            _, _, info = align_to_mask(sq, t)
            cost[i, j] = -info["iou"]
    rows, cols = linear_sum_assignment(cost)
    assign, scores = [None] * len(renders), [0.0] * len(renders)
    for r, c in zip(rows, cols):
        assign[r], scores[r] = int(c), float(-cost[r, c])
    return assign, scores
