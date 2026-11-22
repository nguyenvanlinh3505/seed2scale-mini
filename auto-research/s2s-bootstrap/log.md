# Research log / Nhật ký nghiên cứu

## Recon / Khảo sát (from runs 1-2 / từ lần chạy 1-2)

- 4 seeds: 0-1 successes in 2,048 rollouts; loop never starts. 32 seeds: iteration-0 success 6.6%.
  4 seed: 0-1 thành công / 2.048 rollout. 32 seed: success vòng 0 là 6,6%.
- E0 (run 2, Seed2Scale*): success 7.0 -> 1.6%, |D| 32 -> 42. The VLV gives almost only 3, 6 or 10; gamma=7
  keeps only the 10s (~10% of successes). gate_only: 5.1 -> 4.3%, |D| 32 -> 156. no_filter: collapse (-> 1.2%).
  VLV gần như chỉ cho 3, 6 hoặc 10; gamma=7 chỉ giữ điểm 10 (~10% số ván thành công).
- The collector warm-starts and adds a fixed 2,000 steps per iteration: with s2s data barely growing that is
  ~25 epochs per iteration on the same ~40 trajectories (cumulative ~18k steps) -> likely memorisation.
  Collector train tiếp, thêm 2.000 bước mỗi vòng; với dữ liệu s2s gần như không tăng là ~25 epoch/vòng trên cùng
  ~40 quỹ đạo (tổng ~18k bước), nhiều khả năng học thuộc lòng.
- The paper gives no collector training details (no optimizer, steps, epochs, warm vs scratch) and no gamma.
  Paper không nêu cách train collector và không nêu gamma.

## Hypotheses (ranked by expected gain / cost) / Giả thuyết (xếp theo lợi ích / chi phí)

| id | idea / ý tưởng | mechanism / cơ chế | cost |
|---|---|---|---|
| H1 | retrain collector from scratch each iteration, fixed epoch budget | removes cumulative over-training; each pi(i) depends only on D(i) (literal reading of Eq. 5) | low |
| H2 | gamma = 6 (accept "6" as well as "10") | VLV scores are coarse (3/6/10); gamma=7 starves the loop of data | low; violates "a priori" -> flagged |
| H3 | H1 + H2 | both causes at once | low |
| H4 | 512 rollouts / iteration | 2x successes to filter -> more accepted data per iteration | 2x rollout time |
| H5 | gate-only control with H1 (and with H3 settings) | needed for the success criterion (VLV vs no VLV under the same training) | low |
| H6 | exploration noise sigma = 8 px on actions | more diverse starts reaching the goal; paper adds none | low |

## Experiments / Thí nghiệm

(see experiments.jsonl; analysis appended per experiment / xem experiments.jsonl; phân tích được thêm sau mỗi thí nghiệm)

### E1 (H1: scratch retraining, gamma 7) — discarded / loại
success% 5.5 5.9 9.8 0.0 5.9 5.5 -> late_gain +0.2 pp (noise ~1.9 pp). |D| 34 -> 38.
- Retraining from scratch removes E0's decline (late mean 1.4% in E0 vs 5.7% here): the decline in run 2 was
  most likely cumulative over-training, not the data. / Train lại từ đầu xóa được xu hướng giảm của E0.
- But no rise: the VLV accepts ~1 of ~14 successes per iteration, so D(i) barely grows; nothing new to learn.
  Nhưng không tăng: VLV chỉ nhận ~1/14 ván thành công mỗi vòng, D gần như không lớn thêm.
- Iteration 3 = 0/256 with D=35: scratch training on ~35 trajectories is high-variance (one bad run). Future
  runs need repeat seeds before any claim. / Train từ đầu trên ~35 quỹ đạo có phương sai lớn.
- Next: E3 (gamma 6) tests the data-starvation explanation directly; E4 is its control.

### E3 (H3: scratch retraining + gamma 6) — discarded / loại
success% 5.5 10.2 6.2 0.0 0.0 6.2 -> late_gain -2.3 pp. Accepted 10 14 12 0 0 13; |D| 42 -> 81.
- gamma 6 fixes data starvation: ~12 accepted per iteration instead of ~1 (E1), |D| doubles.
  gamma 6 giải quyết việc thiếu dữ liệu: ~12 quỹ đạo được nhận mỗi vòng thay vì ~1, |D| tăng gấp đôi.
- Success still does not rise; two collapsed iterations (0/256 at 3 and 4) -> H2/H3 refuted: data quantity was not
  the bottleneck. / Success vẫn không tăng; H2/H3 bị bác bỏ: lượng dữ liệu không phải nút thắt.
- gamma 6 was chosen after seeing the VLV score distribution: this violates the paper's a-priori rule (flagged).
  gamma 6 được chọn sau khi thấy phân bố điểm VLV: vi phạm quy tắc "cố định trước" của paper (đã ghi chú).

### E4 (H5: gate-only control, scratch retraining) — discarded / loại
success% 5.5 0.0 5.9 5.9 6.2 6.2 -> late_gain +0.8 pp. Accepted 14 0 15 15 16 16; |D| 46 -> 108.
- Even more data (every success kept) leaves success flat at ~6%. / Thêm nhiều dữ liệu hơn nữa, success vẫn ~6%.
- VLV filter vs no VLV (E3 vs E4): -2.3 vs +0.8 pp, inside the noise -> no measurable VLV effect.
  Có VLV so với không (E3 vs E4): chênh lệch nằm trong nhiễu -> không đo được tác dụng của VLV.
- Iterations 1 and 2 were trained on the SAME 46 trajectories and gave 0.0% vs 5.9%: training variance alone
  spans the whole range we are trying to measure. / Vòng 1 và 2 train trên CÙNG 46 quỹ đạo mà cho 0,0% và 5,9%:
  riêng phương sai khi train đã phủ hết khoảng ta muốn đo.

## Stop / Dừng
Stopping rule met: 3 consecutive experiments (E1, E3, E4) without a significant gain (> 4 pp). See REPORT.md.
Đạt điều kiện dừng: 3 thí nghiệm liên tiếp không có mức tăng đáng kể. Xem REPORT.md.
