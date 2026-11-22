"""SuperTiny collector, built from the paper's description (Sec. III-A, Fig. 1, Eq. 1-3).
Bộ thu thập SuperTiny, dựng theo mô tả trong paper (mục III-A, Hình 1, Eq. 1-3).

  vision:   ResNet-18 feature map -> 1x1 conv projection -> spatial tokens
  language: pretrained T5-small encoder (frozen; the single task string is encoded once)
  state:    compact MLP
  decoder:  Transformer decoder, learnable queries cross-attend to M_t = [z_vis; z_lang; z_state]
  head:     linear action head -> chunk of K actions
  rollout:  exponential temporal ensembling over overlapping chunks (Eq. 3)

Not reported in the paper, chosen here / Paper không nêu, ở đây tự chọn:
  decoder width/depth (sized so the total is ~48M like the paper), L1 loss, ImageNet input
  normalization, random-shift augmentation, actions/states scaled from [0, 512] to [-1, 1].
  độ rộng/độ sâu decoder (để tổng ~48M như paper), loss L1, chuẩn hóa ảnh kiểu ImageNet,
  augmentation dịch ảnh ngẫu nhiên, action/state quy từ [0, 512] về [-1, 1].
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

SPACE = 512.0  # PushT workspace size in px / kích thước vùng làm việc PushT
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_unit(x):
    """[0, 512] -> [-1, 1]"""
    return x / SPACE * 2 - 1


def from_unit(x):
    """[-1, 1] -> [0, 512]"""
    return (x + 1) / 2 * SPACE


class SuperTiny(nn.Module):
    def __init__(self, chunk_size: int = 30, d_model: int = 256, n_layers: int = 4, state_dim: int = 2,
                 action_dim: int = 2, task: str = "", t5_name: str = "t5-small", pretrained: bool = True):
        super().__init__()
        self.chunk_size, self.action_dim = chunk_size, action_dim
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vision = nn.Sequential(*list(backbone.children())[:-2])  # keep the spatial map / giữ bản đồ không gian
        self.vis_proj = nn.Conv2d(512, d_model, kernel_size=1)         # PAPER: 1x1 conv projection

        # PAPER: pretrained T5-small encodes the instruction. Frozen; one task -> encode once and cache.
        # PAPER: T5-small pretrained mã hóa câu lệnh. Đóng băng; chỉ một nhiệm vụ -> mã hóa một lần rồi lưu.
        from transformers import T5EncoderModel, T5Tokenizer
        self.t5 = T5EncoderModel.from_pretrained(t5_name) if pretrained else None
        if self.t5 is not None:
            self.t5.requires_grad_(False)
            tok = T5Tokenizer.from_pretrained(t5_name, legacy=False)(task or "task", return_tensors="pt")
            with torch.no_grad():
                lang = self.t5(**tok).last_hidden_state  # (1, L, 512)
        else:
            lang = torch.zeros(1, 4, 512)
        self.register_buffer("lang_tokens", lang, persistent=False)
        self.lang_proj = nn.Linear(lang.shape[-1], d_model)

        self.state_mlp = nn.Sequential(nn.Linear(state_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))  # memory position embeddings / mã vị trí
        self.queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model, nhead=8, dim_feedforward=4 * d_model, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.head = nn.Linear(d_model, action_dim)

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """images (B,3,H,W) float in [0,1]; states (B,2) in [-1,1] -> actions (B,K,2) in [-1,1].
        images (B,3,H,W) giá trị [0,1]; states (B,2) trong [-1,1] -> actions (B,K,2) trong [-1,1]."""
        x = (images - IMAGENET_MEAN.to(images)) / IMAGENET_STD.to(images)
        z_vis = self.vis_proj(self.vision(x)).flatten(2).transpose(1, 2)        # (B, HW, d)
        z_lang = self.lang_proj(self.lang_tokens).expand(images.shape[0], -1, -1)  # (B, L, d)
        z_state = self.state_mlp(states).unsqueeze(1)                           # (B, 1, d)
        memory = torch.cat([z_vis, z_lang, z_state], dim=1)                     # Eq. 1
        memory = memory + self.pos[:, : memory.shape[1]]
        out = self.decoder(self.queries.expand(images.shape[0], -1, -1), memory)  # Eq. 2
        return torch.tanh(self.head(out))


def n_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    if getattr(model, "t5", None) is None:
        return total, total
    return total, sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_shift(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random translation by up to `pad` px (common for PushT from pixels).
    Dịch ảnh ngẫu nhiên tối đa `pad` px (thường dùng cho PushT từ ảnh)."""
    b, _, h, w = images.shape
    padded = F.pad(images, (pad,) * 4, mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b,))
    oy = torch.randint(0, 2 * pad + 1, (b,))
    return torch.stack([padded[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w] for i in range(b)])


class ChunkDataset(torch.utils.data.Dataset):
    """(image, state, action chunk) samples from a list of trajectories. Chunks past the episode end
    repeat the last action (the agent holds still), with a mask so padding is not over-weighted.
    Mẫu (ảnh, state, chuỗi action) từ danh sách quỹ đạo. Phần chunk vượt quá cuối episode lặp lại
    action cuối (tác tử đứng yên), có mask để phần đệm không bị tính quá nặng."""

    def __init__(self, trajectories: list[dict], chunk_size: int):
        self.trajs, self.k = trajectories, chunk_size
        self.index = [(i, t) for i, tr in enumerate(trajectories) for t in range(len(tr["actions"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        i, t = self.index[n]
        tr = self.trajs[i]
        T = len(tr["actions"])
        idx = np.minimum(np.arange(t, t + self.k), T - 1)
        mask = (np.arange(t, t + self.k) < T).astype(np.float32)
        img = torch.from_numpy(tr["images"][t]).permute(2, 0, 1).float() / 255
        state = to_unit(torch.from_numpy(np.asarray(tr["states"][t], dtype=np.float32)))
        act = to_unit(torch.from_numpy(np.asarray(tr["actions"][idx], dtype=np.float32)))
        return img, state, act, torch.from_numpy(mask)


def train_collector(model: SuperTiny, trajectories: list[dict], steps: int, lr: float, batch: int,
                    device: str, log=print) -> list[float]:
    """Behavior cloning with masked L1 on action chunks for a fixed number of gradient steps
    (so iteration cost does not explode as the dataset grows). Returns the loss every 10% of training.
    Behavior cloning với L1 có mask trên chuỗi action, chạy một số bước gradient cố định
    (để chi phí mỗi vòng không phình to khi dữ liệu tăng). Trả về loss tại mỗi mốc 10%."""
    ds = ChunkDataset(trajectories, model.chunk_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=len(ds) > batch,
                                     num_workers=0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    model.to(device).train()
    losses, running, step = [], [], 0
    while step < steps:
        for img, st, act, mask in dl:
            img, st, act, mask = img.to(device), st.to(device), act.to(device), mask.to(device)
            pred = model(random_shift(img), st)
            loss = ((pred - act).abs().mean(-1) * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running.append(loss.item())
            step += 1
            if step % max(1, steps // 10) == 0 or step == steps:
                losses.append(float(np.mean(running)))
                running = []
                log(f"    collector step {step}/{steps} loss={losses[-1]:.4f}")
            if step >= steps:
                break
    return losses


class TemporalEnsembler:
    """Eq. 3: a_t = sum_k w_k A_{t-k}[k], w_k = exp(-lambda k) / sum_j exp(-lambda j), per environment.
    Eq. 3 cho từng môi trường; chỉ cộng các chunk thật sự phủ tới bước t."""

    def __init__(self, n_envs: int, chunk_size: int, lam: float):
        self.k, self.lam = chunk_size, lam
        self.history: list[list[np.ndarray]] = [[] for _ in range(n_envs)]  # newest first / mới nhất trước

    def __call__(self, env_i: int, chunk: np.ndarray) -> np.ndarray:
        h = self.history[env_i]
        h.insert(0, chunk)
        del h[self.k:]
        acts = np.stack([h[k][k] for k in range(len(h))])
        w = np.exp(-self.lam * np.arange(len(h)))
        return (w[:, None] * acts).sum(0) / w.sum()
