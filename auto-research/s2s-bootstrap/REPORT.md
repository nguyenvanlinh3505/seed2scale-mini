# Auto-research report: make Seed2Scale self-evolution work on PushT
# Báo cáo auto-research: làm cho vòng tự tiến hóa Seed2Scale hoạt động trên PushT

## Outcome / Kết quả

**Success criterion: NOT met.** No configuration raised SuperTiny collector success across self-evolution iterations
by more than the noise (criterion: late_gain > +4 pp and above the gate-only control). The loop stopped by the
contract's rule: 3 consecutive experiments without a significant gain.
**Tiêu chí: KHÔNG đạt.** Không cấu hình nào làm success của collector tăng qua các vòng vượt mức nhiễu. Dừng theo quy
tắc: 3 thí nghiệm liên tiếp không có mức tăng đáng kể.

## What was tried / Những gì đã thử

Collector success (%) per iteration, 256 rollouts from identical start states; noise of late_gain ≈ ±1.9 pp (1 SE).
Success (%) của collector theo vòng, 256 rollout từ cùng trạng thái đầu; nhiễu của late_gain ≈ ±1,9 điểm (1 SE).

| id | Change / Thay đổi | Success % by iteration | \|D\| | late_gain (pp) | Verdict |
|---|---|---|---|---|---|
| E0 | run 2 baseline: warm-start collector, γ=7 | 7.0 7.0 4.3 2.7 3.1 3.9 1.2 1.6 | 32→42 | **−5.5** | decline / giảm |
| E0 control | same, gate-only (no VLV) | 5.1 8.6 6.2 6.6 5.9 7.8 3.9 4.3 | 32→156 | −3.5 | decline / giảm |
| E1 | H1: retrain from scratch each iteration, γ=7 | 5.5 5.9 9.8 0.0 5.9 5.5 | 34→38 | +0.2 | no change / không đổi |
| E3 | H3: scratch + γ=6 | 5.5 10.2 6.2 0.0 0.0 6.2 | 42→81 | −2.3 | no change / không đổi |
| E4 | H5: scratch, gate-only control | 5.5 0.0 5.9 5.9 6.2 6.2 | 46→108 | +0.8 | no change / không đổi |

(E2, E5–E8 — warm γ6, 512 rollouts, exploration noise — were not run. / Chưa chạy E2, E5–E8.)

## Hypotheses / Giả thuyết

| id | Hypothesis / Giả thuyết | Verdict / Kết luận | Evidence / Bằng chứng |
|---|---|---|---|
| H1 | The run-2 decline comes from cumulative over-training of a warm-started collector on a tiny dataset. Sự suy giảm ở lần chạy 2 do train dồn quá mức. | **Supported (weak)** / **Được ủng hộ (yếu)** | Late success 1.4% (E0) → 5.7% (E1); the decline disappears with scratch retraining. One run each. / Train lại từ đầu xóa xu hướng giảm; mỗi bên một lần chạy. |
| H2/H3 | Data starvation (VLV accepts ~1 of 14 successes) stops the rise; γ=6 fixes it. Thiếu dữ liệu làm success không tăng. | **Refuted** / **Bị bác bỏ** | E3 accepts ~12/iteration and doubles \|D\|, yet late_gain −2.3 pp. |
| H5 | The VLV filter adds value over the success gate alone. VLV có giá trị hơn chỉ kiểm tra thành công. | **No measurable effect** / **Không đo được tác dụng** | E3 −2.3 vs E4 +0.8 pp: inside the noise. Also E0 vs its control: both decline. |
| — | More self-generated successful data improves the collector (the core premise). Thêm dữ liệu thành công tự sinh làm collector tốt lên (giả định cốt lõi). | **Not observed** / **Không quan sát thấy** | E4: \|D\| 46 → 108 (×2.3), success flat ~6%. |
| H4, H6 | 512 rollouts; exploration noise. | **Untested** / **Chưa kiểm tra** | H4 adds data, which E3/E4 show does not help, so it is unlikely to change the result. [Inference] |

## Analysis / Phân tích

1. **Training variance dominates.** In E4, iterations 1 and 2 were trained on the *same* 46 trajectories and scored
   0.0% vs 5.9%. Collapses to 0/256 appear in every scratch arm (E1 it. 3, E3 it. 3–4, E4 it. 1). Any real effect of
   the data engine would have to exceed this ~6-point training noise; none does.
   **Phương sai khi train lấn át.** Vòng 1 và 2 của E4 train trên cùng 46 quỹ đạo mà cho 0,0% và 5,9%. Mọi nhánh train
   lại từ đầu đều có vòng sụp về 0/256.
2. **Self-generated successes add little new information.** [Inference] The accepted trajectories are the collector's
   own lucky successes from similar start states, so training on them is close to self-distillation: it does not
   widen the state coverage that PushT's strict success threshold (coverage ≥ 0.95) requires.
   **Thành công tự sinh mang ít thông tin mới.** [Suy luận] Quỹ đạo được nhận là các lần thành công may mắn của chính
   collector, nên train trên chúng gần như tự chưng cất, không mở rộng vùng trạng thái mà ngưỡng 0,95 đòi hỏi.
3. **The VLV is coarse.** Qwen3-VL-32B gives almost only 3, 6 or 10, so γ acts as a 3-level switch; it trades data
   quantity against an unverified notion of quality, and neither side moved success.
   **VLV cho điểm thô** (gần như chỉ 3, 6, 10), nên γ chỉ là công tắc 3 mức.
4. **Downstream:** SmolVLA trained on the 32 seeds alone scored 0/100 in the source world (mean coverage 0.19) and 0%
   in the 3 unseen worlds, so the target model could not show a scaling effect either.
   **Model mục tiêu:** SmolVLA chỉ train trên 32 seed đạt 0/100 ở thế giới gốc và 0% ở 3 thế giới chưa thấy.

## Limitations & threats to validity / Hạn chế

- One run per configuration; with the variance above, small effects (< ~5 pp) cannot be detected.
  Mỗi cấu hình chạy một lần; hiệu ứng nhỏ hơn ~5 điểm không phát hiện được.
- 6 iterations here vs 8 in the paper; PushT (2-D, one camera) vs real robot tasks; 32 seeds instead of 4 (4 seeds
  never bootstrapped, run 1). / 6 vòng thay vì 8; PushT thay vì robot thật; 32 seed thay vì 4.
- γ=6 was chosen after seeing VLV scores (violates the paper's a-priori rule; reported, not hidden).
  γ=6 được chọn sau khi thấy điểm VLV (vi phạm quy tắc của paper; đã báo cáo).
- The paper gives no collector training details (optimizer, steps, warm vs scratch), so our collector schedule is a
  guess. A negative result here does not refute the paper's real-robot claims.
  Paper không nêu cách train collector, nên lịch train ở đây là phỏng đoán. Kết quả âm ở đây không bác bỏ kết quả
  robot thật của paper.

## Next steps (not pursued) / Bước tiếp theo (chưa làm)

1. Repeat each arm with 3 seeds to measure training variance properly. / Lặp mỗi nhánh 3 seed.
2. Reduce collector variance (ensembles, early stopping on held-out seeds) before testing the data engine again.
   Giảm phương sai của collector trước khi thử lại.
3. Add exploration (H6) so accepted data covers new states. / Thêm khám phá để dữ liệu phủ trạng thái mới.

## Reproduce / Tái hiện

Pipeline stage `research=E1+E3+E4` in `molab_pipeline.py` (≈1.6 h on an RTX Pro 6000); experiment definitions in
`s2s/experiments.py`; raw results mirrored to the private HF dataset repo. All runs are in `experiments.jsonl`.
Chạy stage `research=E1+E3+E4` trong `molab_pipeline.py` (~1,6 giờ); định nghĩa ở `s2s/experiments.py`.
