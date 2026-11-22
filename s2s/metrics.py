"""Trajectory smoothness metrics from the paper's Table IV.
Các chỉ số độ mượt quỹ đạo theo Bảng IV của paper.

  TV:     total variation, sum |a_{t+1} - a_t| (normalised by length) / tổng biến thiên (chia theo độ dài)
  jerk:   mean |third difference| (rate of change of acceleration) / trung bình |sai phân bậc ba|
  hf:     share of non-DC spectral energy above 1/4 of the Nyquist band / tỉ lệ năng lượng tần số cao
The paper does not give exact normalisations, so values are comparable within this project only.
Paper không nêu cách chuẩn hóa chính xác, nên giá trị chỉ so sánh được trong dự án này.
"""
from __future__ import annotations

import numpy as np


def smoothness(actions: np.ndarray) -> dict:
    a = np.asarray(actions, dtype=np.float64) / 512.0  # unit workspace / chuẩn về vùng làm việc đơn vị
    if len(a) < 4:
        return {"tv": float("nan"), "jerk": float("nan"), "hf_ratio": float("nan")}
    tv = np.abs(np.diff(a, axis=0)).sum() / len(a)
    jerk = np.abs(np.diff(a, n=3, axis=0)).mean()
    spec = np.abs(np.fft.rfft(a - a.mean(0), axis=0)) ** 2
    spec = spec[1:].sum(1)  # drop DC, sum x/y / bỏ thành phần DC, cộng x/y
    cut = max(1, len(spec) // 4)
    hf = spec[cut:].sum() / spec.sum() if spec.sum() > 0 else 0.0
    return {"tv": float(tv), "jerk": float(jerk), "hf_ratio": float(hf)}
