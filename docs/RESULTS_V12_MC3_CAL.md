# V12: MC3 audit sau freeze trên CAL

Đã đo đủ bốn shard GPU, 200 source video CAL và 2.264 trial encode/decode + inference MC3 duy nhất. Policy V12-A và V12-B giữ nguyên freeze. Trạng thái component MC3: A `FAIL_ON_REUSED_CAL`, B `FAIL_ON_REUSED_CAL`. Đây là diagnostic trên CAL đã dùng, **chưa xác nhận độc lập trên DEV/holdout mới**.

[Preregistration khóa trước scoring](PREREGISTRATION_V12_MC3_CAL_AUDIT.md), [JSON đầy đủ](../results/v12_mc3_audit/mc3_result.json), [chẩn đoán và primary reference](../results/v12_mc3_audit/assessment.json), [archive/hash](../results/v12_mc3_audit/README.md).

## MC3 so với identity

| Policy / analyzer | BD-rate % [CI95] vs identity | BD-accuracy pp [CI95] | Worst same-QP gap pp |
| --- | --- | --- | --- |
| v6 | -1.67 [-5.73; +2.39] | -0.054 [-2.60; +2.42] | -8.50 |
| area112 | +2.31 [-2.35; +7.53] | -2.279 [-5.09; +0.53] | -12.50 |
| v12a | -2.37 [-6.13; +1.54] | -0.004 [-2.22; +2.34] | -7.00 |
| v12b | -3.21 [-7.75; +0.39] | +1.238 [-0.89; +3.41] | -4.00 |

Các con số của V6 và fixed area112 trong bảng là lượt đo mới trên đúng cohort CAL này; không thay thế các cohort hoặc codec lịch sử. CI95 là percentile bootstrap ghép cặp theo source video, 2.000 draw seed 20261010, cùng resample matrix cho chín comparison. Phải báo số draw hợp lệ trong JSON khi đường cong mất miền giao; không đổi logic BD-rate/bootstrap.

## So sánh trực tiếp trên MC3

| Direct comparison | BD-rate % [CI95] | BD-accuracy pp [CI95] | Worst same-QP gap pp |
| --- | --- | --- | --- |
| v12a_vs_v6 | -0.43 [-2.42; +1.47] | +0.011 [-1.17; +1.39] | +0.00 |
| v12b_vs_v6 | -1.61 [-4.97; +0.73] | +1.318 [-0.05; +2.75] | -0.50 |
| v12a_vs_area112 | -4.41 [-7.90; -1.27] | +2.215 [+0.27; +4.37] | +1.00 |
| v12b_vs_area112 | -5.73 [-9.94; -2.33] | +3.528 [+1.34; +5.91] | -1.50 |
| v12b_vs_v12a | -1.19 [-4.16; +1.00] | +1.310 [-0.09; +2.68] | -2.50 |

BD-rate trực tiếp được tính trên hai đường cong của từng cặp, không trừ hai BD-rate đã neo identity. Comparison B/A được đăng ký trước đo và chỉ mô tả. Báo cả hai policy, không dùng kết quả MC3 để chọn hoặc hiệu chỉnh lại policy.

## Top-1 từng QP và chuyển đúng/sai

| QP | Identity Top-1 % | V6 Top-1 % | Area112 Top-1 % | A Top-1 % / gap pp | B Top-1 % / gap pp |
| --- | --- | --- | --- | --- | --- |
| 30 | 65.50 | 63.00 | 64.00 | 65.00 / -0.50 | 62.50 / -3.00 |
| 35 | 58.50 | 50.00 | 49.50 | 51.50 / -7.00 | 54.50 / -4.00 |
| 40 | 47.00 | 48.00 | 41.00 | 48.50 / +1.50 | 48.50 / +1.50 |
| 45 | 29.50 | 26.50 | 17.00 | 26.50 / -3.00 | 26.50 / -3.00 |
| 50 | 9.00 | 8.50 | 4.50 | 8.50 / -0.50 | 8.50 / -0.50 |

| QP | A switches | A MC3 wins/losses vs V6 | B switches | B MC3 wins/losses vs V6 |
| --- | --- | --- | --- | --- |
| 30 | 62 | 6 / 2 | 64 | 3 / 4 |
| 35 | 24 | 5 / 2 | 50 | 10 / 1 |
| 40 | 2 | 1 / 0 | 13 | 1 / 0 |
| 45 | 0 | 0 / 0 | 0 | 0 / 0 |
| 50 | 0 | 0 / 0 | 0 | 0 / 0 |

QP45/50 của cả hai policy giữ stream V6 theo freeze; MC3 ở hai QP đó bằng V6 từng nguồn. Cơ chế chỉ can thiệp QP30/35/40 nên không sửa được bất kỳ deficit nào ở QP45/50. Counts từ raw source records được lưu trong assessment.json; Top-1 ở 200 nguồn có bước 0,5 pp.

## Gate và phạm vi kết luận

Component MC3 được khóa: BD-rate<0, upper CI≤+1%, BD-accuracy>0, worst same-QP gap≥−1,00 pp, ít nhất 1.900 valid draw cho rate và accuracy. Trạng thái trong JSON là PASS/FAIL trên CAL hoặc INCONCLUSIVE; không gọi là đạt gate trên holdout. Ngưỡng nghiên cứu gốc cả hai primary<−15% giữ nguyên và KHÔNG ĐẠT trên CAL.

V12-A trượt upper CI (+1,543% > +1%), BD-accuracy (−0,003832 pp, hơi âm) và worst gap (−7,00 pp tại QP35). V12-B qua rate point, upper CI (+0,385%), BD-accuracy (+1,238 pp) và số draw, nhưng trượt worst gap (−4,00 pp tại QP35). Cả hai vẫn có gap −3,00 pp tại QP45 vì giữ V6, nên chỉ sửa quyết định QP30/35/40 không thể đạt toàn bộ guard same-QP trên cohort này. Kết quả FAIL được báo nguyên vẹn.

Point estimate MC3 của B âm hơn A và V6, nhưng CI trực tiếp B/V6 [−4,97%; +0,73%] và B/A [−4,16%; +1,00%] đều chứa 0: chưa có bằng chứng rõ về cải thiện. Cả hai có direct BD-rate tốt hơn fixed area112 với CI hoàn toàn âm trong audit mô tả này; không suy ra tổng quát hóa hoặc điều chỉnh policy từ MC3.

V12-B semantic giữ primary CAL đã báo: R2 -13.03%, R3 -12.92%. Mean bpp từng QP của MC3 khớp tuyệt đối hai primary trong JSON đã freeze. Primary CI dùng seed 20261009; không ghép thành joint bootstrap ba analyzer với audit MC3 seed 20261010.

MC3 không tham gia fit/chọn V12, nhưng các kết quả MC3 trước đây đã thúc đẩy giả thuyết; không tuyên bố kiến trúc hoàn toàn chưa từng quan sát. CAL chính là tập đã dùng để chọn V12 bằng primary/proxy. Không chạy TEST hoặc holdout trong audit này. **Fresh DEV, holdout mới và runtime đầy đủ: CHƯA ĐO.**

## Provenance và tái lập

Preregistration commit `fda9e069643e084d715c6f71a0cdefc9000f0204`; scorer/merge commit `43f62f3aff682b557bc00bbb4707479abbe268b7` tại `munnn01/preeee`. Result SHA-256 `b626431c0d84a71d876a120b3fd83850fc66bc69ecfd1dd6fede78e1f1627cc4`; selection SHA-256 `bd677379e4420a50a71e8b0a630f1ce3519f42707da21667ac198eb3c93d9103`; fingerprint nguồn `ce5acf9334d4f683fdc7757d53b0d5be0ce80cc5a6fd4a4340e30c9357e79721`. Frozen parent commit/hash đầy đủ, trọng số/state MC3, hashes protocol/index/metric/bootstrap, seed và đơn vị resample nằm trong JSON. Bản trong test_pre là bản sao byte-identical của audit chung từ preeee, không phải lượt chạy độc lập.

Chạy lại merge sau kiểm toán từ repo preeee ở commit scorer:

```powershell
python -B -m research.v12_mc3_audit.study --prereg-commit fda9e069643e084d715c6f71a0cdefc9000f0204 merge `
  --shard-dir results/v12_mc3_audit/raw/shard0 --shard-dir results/v12_mc3_audit/raw/shard1 `
  --shard-dir results/v12_mc3_audit/raw/shard2 --shard-dir results/v12_mc3_audit/raw/shard3 `
  --out <fresh-output.json>
```

Tổng timer encode/decode audit 705.727 s; tổng inference MC3 105.701 s qua bốn worker. Đây là chi phí chấm các stream đã khóa, không phải full-selector overhead hay wall-time parallel. Hardware/versions theo manifest; log full pytest và receipt được giữ cùng kết quả.

## Ghi chú kỹ thuật về test

Lần full pytest đầu gặp 42 lỗi quyền thư mục tạm Windows. Dùng basetemp mới giải quyết môi trường và lộ một lỗi fixture: dictionary stream tạo từ set có thứ tự phụ thuộc hash seed, nên stream tốt có thể được chấm trước stream cố ý hỏng. Đã đổi sang dict.fromkeys để fixture luôn đặt identity128 đầu tiên như assertion yêu cầu. Đây là sửa test, không đổi scorer hoặc luật kiểm tra từng stream trước inference; không chạy lại Kaggle hoặc bootstrap, không đổi một giá trị thực nghiệm nào.

[Patch, hash fixture và log của cả ba lần chạy](../results/v12_mc3_audit/test_fixture_correction.json). Full suite cuối dùng basetemp mới trong workspace; scorer/merge vẫn đúng commit đã preregister.
