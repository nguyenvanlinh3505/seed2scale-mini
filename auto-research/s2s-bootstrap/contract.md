# Research contract: make Seed2Scale self-evolution work on PushT
# Hợp đồng nghiên cứu: làm cho vòng tự tiến hóa Seed2Scale hoạt động trên PushT

**Objective / Mục tiêu.** In the Seed2Scale* arm (success gate + Qwen3-VL verifier), the SuperTiny collector's
success rate should rise across self-evolution iterations, as in the paper's Fig. 7.
Ở nhánh Seed2Scale* (cổng thành công + verifier Qwen3-VL), success của collector SuperTiny phải tăng qua các
vòng tự tiến hóa, như Hình 7 của paper.

**Primary metric / Chỉ số chính: `late_gain_pp` (max).** Mean collector success of the last two iterations minus
iteration 0, in percentage points. Success = env `is_success` (coverage >= 0.95) over 256 rollouts per iteration
from fixed start states (`rollout_seeds(i, 256)`, identical across experiments).
Trung bình success hai vòng cuối trừ vòng 0, tính bằng điểm phần trăm, trên 256 rollout với trạng thái đầu cố định.

**Noise / Nhiễu.** At ~6% success, one iteration's binomial SE is ~1.5 pp; the late-mean minus iteration-0
difference has SE ~1.9 pp. A gain counts only if > 4 pp (~2 SE). Each config is run once (one seed): results
within the noise are reported as "no significant difference".
Ở mức ~6%, sai số chuẩn mỗi vòng ~1,5 điểm; hiệu số ~1,9 điểm. Chỉ tính là tăng nếu > 4 điểm (~2 SE).

**Success criterion / Tiêu chí đạt.** Seed2Scale* arm with late_gain_pp > +4 AND late success above the
gate-only control run with the same settings.
Nhánh Seed2Scale* có late_gain_pp > +4 VÀ success cuối cao hơn nhánh gate-only cùng cấu hình.

**Guardrails / Ràng buộc.**
- The success gate, the rollout start states, the success threshold (0.95) and the eval protocol never change.
  Cổng thành công, trạng thái đầu rollout, ngưỡng 0,95 và cách đánh giá không bao giờ đổi.
- The VLV rubric text does not change. gamma may be varied, but every gamma tried is reported, and the paper's
  rule (gamma fixed a priori) is flagged as violated for any gamma chosen after seeing results.
  Rubric VLV không đổi. Có thể thử gamma khác nhưng phải báo cáo mọi giá trị đã thử, và ghi rõ là vi phạm quy tắc
  "gamma cố định trước" của paper nếu chọn gamma sau khi đã thấy kết quả.
- The SmolVLA target evaluation (100 fixed seeds 100000+) is NOT used to choose configs; it is run once on the
  final chosen config. / Đánh giá SmolVLA không dùng để chọn cấu hình; chỉ chạy một lần trên cấu hình cuối.

**Budget / Ngân sách.** molab GPU (free), this session only; up to ~8 experiments of 6 iterations
(~25-40 min each, VLV server kept up). No paid resources.
GPU molab (miễn phí), chỉ trong phiên này; tối đa ~8 thí nghiệm, mỗi thí nghiệm 6 vòng.

**Stopping / Dừng khi.** Success criterion met; or budget spent; or 3 consecutive experiments without a
significant gain over the best so far.

**Scope / Phạm vi.** Changes go through new, opt-in flags in `s2s/evolve.py` and a `research` pipeline stage;
defaults (and therefore runs 1-2) stay reproducible. Outputs under `<work>/research/E<n>/`.
