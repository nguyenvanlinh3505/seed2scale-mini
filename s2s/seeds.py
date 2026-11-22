"""Pick the four seed demonstrations from lerobot/pusht, one per workspace corner (paper Sec. III-C).
Chọn 4 demo seed từ lerobot/pusht, mỗi góc vùng làm việc một demo (paper mục III-C).

The dataset does not store the T-block pose, so we locate the block in the first frame by its
color (LightSlateGray) and take the episode whose block starts closest to each corner.
Dataset không lưu vị trí khối T, nên ta tìm khối T trong khung hình đầu bằng màu (LightSlateGray)
và lấy episode có khối T bắt đầu gần mỗi góc nhất.
"""
from __future__ import annotations

import numpy as np

BLOCK_RGB = np.array([119, 136, 153])
# Corners of the block spawn region (100..400 px of 512) mapped to 96 px images.
# Các góc của vùng sinh khối T (100..400 px trên 512) quy về ảnh 96 px.
CORNERS_96 = np.array([[100, 100], [400, 100], [100, 400], [400, 400]]) * 96 / 512


def block_centroid(img: np.ndarray, tol: int = 30) -> np.ndarray | None:
    """(x, y) centroid of block-colored pixels in an HxWx3 uint8 image, or None.
    Tâm (x, y) của các điểm ảnh màu khối T trong ảnh HxWx3 uint8, hoặc None."""
    mask = np.abs(img.astype(int) - BLOCK_RGB).max(-1) < tol
    if mask.sum() < 10:
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def pick_corner_episodes(first_frames: dict[int, np.ndarray]) -> list[int]:
    """Map {episode_index: first frame} to 4 distinct episodes, one per corner.
    Từ {episode_index: khung đầu} chọn 4 episode khác nhau, mỗi góc một episode."""
    cents = {ep: c for ep, img in first_frames.items() if (c := block_centroid(img)) is not None}
    chosen: list[int] = []
    for corner in CORNERS_96:
        ep = min((e for e in cents if e not in chosen), key=lambda e: np.linalg.norm(cents[e] - corner))
        chosen.append(ep)
    return chosen


def load_seed_episodes(repo_id: str = "lerobot/pusht", n_scan: int | None = None) -> list[dict]:
    """Return 4 seed episodes as dicts of numpy arrays: images (T,96,96,3) uint8, states (T,2), actions (T,2).
    Trả về 4 episode seed dạng dict mảng numpy."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id)
    ep_idx = np.asarray(ds.hf_dataset["episode_index"]).reshape(-1)
    starts = {int(e): int(np.argmax(ep_idx == e)) for e in np.unique(ep_idx)}
    if n_scan:
        starts = dict(list(starts.items())[:n_scan])

    def frame(i: int) -> np.ndarray:
        img = ds[i]["observation.image"]  # (3,H,W) float in [0,1]
        return (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    chosen = pick_corner_episodes({e: frame(s) for e, s in starts.items()})
    episodes = []
    for e in chosen:
        idx = np.nonzero(ep_idx == e)[0]
        episodes.append({
            "episode_index": e,
            "images": np.stack([frame(int(i)) for i in idx]),
            "states": np.stack([ds.hf_dataset[int(i)]["observation.state"] for i in idx]).astype(np.float32),
            "actions": np.stack([ds.hf_dataset[int(i)]["action"] for i in idx]).astype(np.float32),
        })
    return episodes
