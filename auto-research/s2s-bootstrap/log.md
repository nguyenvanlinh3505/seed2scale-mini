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
