# Seed2Scale-mini: what does a self-evolving data engine actually buy you?
# Seed2Scale-mini: một data engine tự tiến hóa thực sự mang lại điều gì?

A scaled-down, paper-faithful replication of **Seed2Scale** (Tai et al., arXiv:2603.08260) on the
**PushT** simulator, built to run on one GPU (molab, RTX Pro 6000 96 GB).
Bản tái hiện thu nhỏ, bám sát paper **Seed2Scale** (Tai et al., arXiv:2603.08260) trên mô phỏng **PushT**,
thiết kế để chạy trên một GPU (molab, RTX Pro 6000 96 GB).

![worlds](docs/worlds.png)

## Questions / Câu hỏi

1. From **4 seed demos**, does the target model (**SmolVLA**) keep improving as self-evolution iterates? *(paper Fig. 4)*
   Từ **4 demo seed**, model mục tiêu (**SmolVLA**) có tiếp tục tốt lên qua các vòng tự tiến hóa không? *(Hình 4)*
2. What does the **VLM verifier** add on top of a plain success check? *(paper Sec. IV-G, "SuperTiny⁻")*
   **VLM verifier** đóng góp gì thêm so với chỉ kiểm tra thành công? *(mục IV-G, "SuperTiny⁻")*
3. Without any filter, does self-training **collapse**? *(the paper's motivation; not ablated there)*
   Không lọc gì cả thì quá trình tự học có **sụp đổ** không? *(động lực của paper; paper không làm ablation này)*
4. Do **Parallel Worlds** (same physics, new appearance) transfer to **unseen** worlds? *(paper Table I)*
   **Parallel Worlds** (cùng vật lý, khác hình ảnh) có giúp chuyển sang thế giới **chưa thấy** không? *(Bảng I)*
5. What does the VLV actually reward (smoothness? length?) *(paper Fig. 7, Table IV)*
   VLV thật sự thưởng cho điều gì (độ mượt? độ dài?) *(Hình 7, Bảng IV)*

## Faithfulness / Mức độ bám sát paper

| Component | Paper | Here / Ở đây |
|---|---|---|
| Seeds | 4 teleop demos at the 4 workspace corners | 4 `lerobot/pusht` demos whose T-block starts nearest each corner (found from the first frame) |
| Collector SuperTiny | ResNet-18 + 1×1 conv, T5-small, state MLP, query decoder, K=30, temporal ensembling, 48M | same design, 51M (decoder size not reported) |
| Verifier | success gate, then frozen **Qwen3-VL-32B** scores 0–10 with task + rollout video + seed reference video; γ fixed a priori | same; rubric and γ=7 fixed in `s2s/verifier.py` / `s2s/config.py` before any run |
| Self-evolution | D(i+1) = D(i) ∪ D_silver(i), 8 iterations | same; N_rollout = 256 (not reported) |
| Target | SmolVLA, "trained from scratch" per iteration, 100k steps, AdamW 3e-4, batch 16 | SmolVLA, pretrained SmolVLM2 backbone + new action expert per dataset, AdamW 3e-4, batch 16, **20k steps** (budget) |
| Parallel Worlds | 10 worlds (some 3DGS) + 3 unseen, replay actions, feasibility check, VLV filter | 5 seen + 3 unseen **color** worlds, exact action replay (physics identical), success re-check, VLV filter |
| Tasks / robots | 4 Agibot A2 + 2 GR-1 tasks, real robot | PushT only, simulation only |

Why "from scratch" = pretrained backbone: in LeRobot 0.7.0, `load_vlm_weights=false` randomly
initialises the whole VLM *and* keeps it frozen; LeRobot warns this is "unlikely to yield good results".
Vì sao "from scratch" = backbone pretrained: trong LeRobot 0.7.0, `load_vlm_weights=false` khởi tạo ngẫu
nhiên toàn bộ VLM *và* đóng băng nó; chính LeRobot cảnh báo cách này "unlikely to yield good results".

## Run 1 finding: 4 seeds do not bootstrap on PushT / Phát hiện lần chạy 1: 4 seed không khởi động được trên PushT

With the paper's 4 corner seeds, 8 iterations × 256 rollouts gave **0/2,048 successes** for Seed2Scale*
and **1/2,048** for SuperTiny⁻ (best coverage 0.81 and 0.96; mean 0.06–0.17). The collector memorised the
demos (training loss 0.005) and the self-evolution loop never started, so the VLV was never even called.
Self-evolution needs a collector with small but **non-zero** early success (the paper reports < 15%); on
PushT, 4 demos do not provide it. Run 1 also exposed a bug: model initialisation was not seeded, so arms
differed by luck (mean coverage 0.08 vs 0.16 at iteration 0 on identical data). Both arms are kept in
`runs/` (and on the Hub) as a negative result.
Với 4 seed ở 4 góc như paper, 8 vòng × 256 rollout cho **0/2.048 thành công** ở Seed2Scale* và **1/2.048**
ở SuperTiny⁻ (độ phủ cao nhất 0,81 và 0,96; trung bình 0,06–0,17). Collector học thuộc các demo (loss 0,005)
và vòng tự tiến hóa không bao giờ khởi động, nên VLV chưa từng được gọi. Tự tiến hóa cần collector có
success ban đầu nhỏ nhưng **khác 0** (paper báo < 15%); trên PushT, 4 demo không đủ. Lần chạy 1 cũng lộ ra
một lỗi: khởi tạo model không cố định seed, nên các nhánh khác nhau do may rủi (độ phủ trung bình 0,08 so
với 0,16 ở vòng 0 trên cùng dữ liệu). Cả hai nhánh được giữ trong `runs/` (và trên Hub) làm kết quả âm tính.

Run 2 changes / Thay đổi ở lần chạy 2:
- fixed random seeds for model init and training / cố định seed ngẫu nhiên khi khởi tạo và train;
- a `calibrate` stage measures iteration-0 collector success with 4, 8, 16, 32 seeds and the pipeline uses
  the smallest count reaching 3% (corners first, then farthest-point spread) / giai đoạn `calibrate` đo
  success vòng 0 với 4, 8, 16, 32 seed và pipeline dùng số nhỏ nhất đạt 3% (góc trước, rồi trải đều);
- fewer collector steps (4000 / 2000) to reduce memorisation / ít bước train collector hơn để giảm học thuộc;
- each seed count writes to its own `runs_n<N>/`, `results_n<N>/` so runs never mix / mỗi số seed ghi vào
  thư mục riêng để các lần chạy không trộn lẫn.

## Arms and target models / Các nhánh và model mục tiêu

| Arm | Filter | Paper name |
|---|---|---|
| `s2s` | success gate + VLV | Seed2Scale* |
| `gate_only` | success gate | SuperTiny⁻ |
| `no_filter` | none | (extra) |

Targets, trained in this priority order so a cut-off run still answers the main questions:
Các model mục tiêu, train theo thứ tự ưu tiên để nếu bị dừng giữa chừng vẫn trả lời được câu hỏi chính:
`seed_only`, `s2s_i8`, `gate_only_i8`, `s2s_full_i8` (+ Parallel Worlds), `no_filter_i8`, `s2s_i4`, `s2s_i2`.
Each is evaluated on 100 fixed seeds in the source world and in 3 unseen worlds (Wilson 95% CIs).
Mỗi model được đánh giá trên 100 seed cố định ở thế giới gốc và 3 thế giới chưa thấy (khoảng tin cậy Wilson 95%).

## Run on molab / Chạy trên molab

1. Push this repo to GitHub; open `molab_pipeline.py` in molab with a GPU attached.
   Đẩy repo lên GitHub; mở `molab_pipeline.py` trên molab, bật GPU.
2. Fill in the repo URL, a private HF repo name and an HF **write** token; click **Install**.
   Điền URL repo, tên repo HF riêng tư và token HF quyền **ghi**; bấm **Install**.
3. Start with stages `setup,smoke`: it checks the simulator, the VLV server and times SmolVLA training.
   Chạy trước `setup,smoke`: kiểm tra mô phỏng, server VLV và đo tốc độ train SmolVLA.
4. Then `evolve,worlds,targets,analyze`. Every stage is resumable and mirrored to the HF repo,
   so after molab's 12 h limit you just start again.
   Sau đó `evolve,worlds,targets,analyze`. Mọi giai đoạn chạy tiếp được và được sao lưu lên repo HF,
   nên sau giới hạn 12 giờ của molab chỉ cần bấm chạy lại.

Locally / Trên máy: `pip install -e . && pytest -q` (16 CPU tests, no GPU / không cần GPU).

## Layout / Cấu trúc

```
s2s/config.py          every setting, tagged PAPER or DEVIATION / mọi cấu hình, gắn nhãn PAPER hoặc DEVIATION
s2s/worlds.py          PushT worlds: same physics, new colors / các thế giới PushT
s2s/seeds.py           4 corner seed demos / 4 demo seed ở 4 góc
s2s/supertiny.py       SuperTiny collector + training + temporal ensembling (Eq. 1-3)
s2s/rollout.py         batched rollouts, exact replay / rollout theo batch, chạy lại chính xác
s2s/verifier.py        success gate + Qwen3-VL verifier client (Eq. 4-6)
s2s/evolve.py          self-evolution loop with resume (Eq. 5-7) / vòng tự tiến hóa
s2s/parallel_worlds.py projection into seen worlds (Eq. 8-10) / chiếu sang các thế giới
s2s/export.py          trajectories -> LeRobotDataset
s2s/train_target.py    SmolVLA via lerobot-train
s2s/eval_target.py     evaluation with LeRobot's rollout loop / đánh giá bằng vòng rollout của LeRobot
s2s/analyze.py         figures and tables / hình và bảng
s2s/pipeline.py        resumable orchestration + HF sync / điều phối chạy tiếp được + đồng bộ HF
molab_pipeline.py      marimo notebook for molab / notebook marimo cho molab
```

## Known limitations / Hạn chế đã biết

- PushT is 2-D and visually simple; results show the *mechanism*, not the paper's robot numbers.
  PushT là bài 2D, hình ảnh đơn giản; kết quả cho thấy *cơ chế*, không phải các con số robot của paper.
- Parallel Worlds here change colors only (no new objects, layouts or cameras), and seeds cannot be
  projected because the dataset lacks their exact start states.
  Parallel Worlds ở đây chỉ đổi màu (không đổi vật thể, bố cục hay camera), và seed không chiếu được vì
  dataset không có trạng thái đầu chính xác của chúng.
- SmolVLA trains 20k steps instead of 100k, and the scaling curve uses 4 points instead of 8.
  SmolVLA train 20k bước thay vì 100k, và đường mở rộng dùng 4 điểm thay vì 8.
- One seed per configuration (no repeated runs), so differences inside the CIs are not claims.
  Mỗi cấu hình chạy một lần (không lặp lại), nên chênh lệch nằm trong khoảng tin cậy không được coi là kết luận.

## Reference / Tài liệu tham khảo

C. Tai et al., *Seed2Scale: A Self-Evolving Data Engine with Parallel Worlds Expansion for Scalable Robot
Learning*, arXiv:2603.08260, 2026. Project page: https://terminators2025.github.io/Seed2Scale.github.io
