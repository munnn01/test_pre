# V12-B semantic: kết quả CAL

**PASS gate sàng lọc CAL; trạng thái `FREEZE_BEFORE_NEW_DEV_PROTOCOL`. Gate nghiên cứu gốc −15%: KHÔNG ĐẠT trên CAL. Mục tiêu đầy đủ ba analyzer, mc3_18, DEV mới và holdout: CHƯA ĐO.** Đây là phép phát triển trên CAL đã dùng; CI sau chọn policy chỉ mô tả, chưa hiệu chỉnh bất định do lựa chọn.

Đã xác minh 4/4 job COMPLETE phiên bản v1, đủ 200 nguồn/1.000 source-QP observations và 6000 trial encode/decode. Tất cả 2.235 stream trùng giữa hai hướng có cùng số byte và SHA-256 pixel decode. Hai analyzer chính dùng correctness trong index CAL lịch sử đã khóa; worker chỉ đo proxy/codec, không chạy AR inference mới. [JSON kết quả](../results/v12_lowqp/calibration_result.json), [phân tích mô tả](../results/v12_lowqp/assessment.json), [freeze](../results/v12_lowqp/frozen_policy.json).

## Policy được chọn và điều kiện

Policy `{"tau_relative": 0.25, "rate_slack": 0.25, "qp_mode": "lowmid"}`; 8/24 cấu hình khả thi, 127/1.000 lựa chọn đổi so với V6 trên 91/200 nguồn. QP 45/50 giữ V6. Lựa chọn theo ranking đã preregister: proxy reduction lớn nhất trong tập feasible của riêng hướng này, rồi các tie-break đã khóa. Không so sánh độ lớn proxy reduction giữa hai hướng vì hai proxy có ý nghĩa khác nhau.

CAL feasible: mỗi primary BD-rate <−10%, BD-accuracy >0, worst same-QP gap ≥−1 pp so với identity, rate không kém V6 quá 2 pp và proxy reduction >1e−6. Chính sách được chọn còn có đủ 2.000/2.000 draw hợp lệ cho BD-rate/BD-accuracy ở mọi comparator. Gate gốc và gate tương lai ba analyzer giữ nguyên.

## Đường cong gộp so với identity

| Analyzer | BD-rate % [CI95] vs identity | BD-accuracy pp [CI95] | Worst same-QP gap pp vs identity | Valid rate draws |
| --- | --- | --- | --- | --- |
| r2plus1d_18 | -13.03 [-16.24; -9.38] | +9.34 [+7.08; +11.78] | +1.50 | 2000 |
| r3d_18 | -12.92 [-15.79; -10.32] | +8.59 [+6.63; +10.71] | +0.50 | 2000 |

Khoảng CI của R2 vẫn vượt −10% về phía ít âm hơn. Vì đây là CAL đã dùng để chọn policy, CI không xác nhận mức tiết kiệm trên dữ liệu mới. Worst same-QP gap trong bảng này neo identity; bảng dưới neo từng comparator riêng.

## Comparator ghép cặp trên cùng CAL

| Anchor | Analyzer | Direct BD-rate % [CI95] | BD-accuracy pp [CI95] | Worst same-QP gap pp |
| --- | --- | --- | --- | --- |
| v6 | r2plus1d_18 | +1.36 [+0.25; +2.51] | -1.20 [-2.30; -0.20] | -1.00 |
| v6 | r3d_18 | +1.94 [+1.08; +2.85] | -1.55 [-2.39; -0.89] | -1.00 |
| area112 | r2plus1d_18 | -12.08 [-15.65; -9.03] | +10.05 [+7.42; +12.69] | +8.50 |
| area112 | r3d_18 | -6.50 [-9.53; -3.15] | +3.65 [+1.40; +6.07] | +2.00 |

Cả hai primary có BD-rate trực tiếp so với V6 dương với CI hoàn toàn dương: policy trả một phần hiệu quả rate–accuracy của V6 để bảo vệ proxy. Suy giảm BD-accuracy và thay đổi Top-1 được báo đầy đủ. Cả hai vẫn tốt hơn fixed area112 trên CAL; điều đó cho thấy selector có giá trị vượt baseline downscaling cố định trong phạm vi CAL, chưa phải bằng chứng tổng quát hóa.

## Thay đổi theo QP

| QP | Switches / 200 | Bpp change vs V6 % | R2 wins / losses vs V6 | R3 wins / losses vs V6 |
| --- | --- | --- | --- | --- |
| 30 | 64 | +5.412 | 2 / 3 | 1 / 3 |
| 35 | 50 | +4.001 | 1 / 3 | 0 / 2 |
| 40 | 13 | +0.808 | 0 / 0 | 0 / 0 |
| 45 | 0 | +0.000 | 0 / 0 | 0 / 0 |
| 50 | 0 | +0.000 | 0 / 0 | 0 / 0 |

Wins/losses là chuyển đúng/sai trong primary index lịch sử, so với V6 trên 200 nguồn, không phải quan sát MC3. QP 45/50 không đổi; deficit downstream tại các QP đó, nếu xuất hiện trên dữ liệu mới, sẽ không được cơ chế này sửa.

## Chi phí đã đo và phần còn thiếu

Tổng internal encode/decode timers qua bốn máy: 1941.520 s; source + decoded proxy timers: 70.083 s, trung bình 0.3504 s/video cho toàn bộ năm QP đã chạy. Đây là tổng thời gian các vòng đo pilot, không phải elapsed wall-time của bốn job song song hay overhead đầy đủ của selector V6. Source I/O, model startup/download, inference của selector V6, peak memory và benchmark máy chuẩn chưa được đo trong pilot này.

V12-A chạy proxy CPU; V12-B dùng ResNet18 GPU. Môi trường hai hướng khác CPU/CUDA nên các internal timers không phải phép so sánh tốc độ được kiểm soát. Số trial pilot cũng không thể dùng làm tỷ lệ overhead encoder triển khai: cả hai sẽ cần chi phí lựa chọn V6 trong pipeline đầy đủ.

MC3 đã được quan sát trong các phiên bản trước và thúc đẩy giả thuyết V12; việc fit/chọn của V12 loại trừ MC3. Không tuyên bố đây là kiến trúc hoàn toàn chưa từng nhìn thấy.

Kiểm toán kết quả và 19 test V12 hiện tại đều PASS. [Record kiểm tra và test](../results/v12_lowqp_validation.json).

## Hướng tiếp theo

Commit hai policy và toàn bộ CAL choices trước bước tiếp theo. Ưu tiên V12-A cho đánh giá phát triển tiếp vì cơ chế đơn giản và point estimate trên primary tốt hơn; giữ V12-B làm comparator. Đây là lựa chọn nghiên cứu dựa trên CAL, chưa có kiểm định ghép cặp A/B preregistered hay bằng chứng MC3. Một protocol mới phải khóa fresh DEV source-disjoint, hai policy và các gate trước khi đo ba analyzer; mọi lựa chọn phải freeze trước inference. Gate đầy đủ kiểm tra MC3 <0, upper CI ≤+1%, mọi worst same-QP gap ≥−1 pp. Holdout mới phải khóa riêng và chỉ đánh giá một lần ở cuối.

## Toàn bộ lưới đã đăng ký

| tau | slack | QP mode | R2 BD-rate % | R3 BD-rate % | R2 worst gap pp | R3 worst gap pp | Proxy reduction (own scale) | Switches | CAL feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.1 | low | -10.3121 | -10.7626 | -1.50 | -1.50 | 0.00346938 | 191 | False |
| 0.0 | 0.1 | lowmid | -7.3417 | -8.0183 | -1.50 | -1.50 | 0.00757864 | 324 | False |
| 0.0 | 0.25 | low | -7.6447 | -7.8886 | -3.00 | -2.00 | 0.01078447 | 360 | False |
| 0.0 | 0.25 | lowmid | -4.2654 | -3.3145 | -3.00 | -2.00 | 0.01695799 | 500 | False |
| 0.0 | 0.5 | low | -8.0852 | -7.6235 | 0.00 | 0.00 | 0.01394961 | 368 | False |
| 0.0 | 0.5 | lowmid | -4.4422 | -2.6357 | 0.00 | 0.00 | 0.02017218 | 508 | False |
| 0.1 | 0.1 | low | -11.9318 | -12.6813 | -0.50 | 0.50 | 0.00275768 | 108 | False |
| 0.1 | 0.1 | lowmid | -9.6880 | -11.3964 | -0.50 | 0.50 | 0.00591514 | 182 | False |
| 0.1 | 0.25 | low | -8.2253 | -8.7624 | -2.50 | -1.00 | 0.01032338 | 315 | False |
| 0.1 | 0.25 | lowmid | -5.5072 | -5.4159 | -2.50 | -1.00 | 0.01605965 | 424 | False |
| 0.1 | 0.5 | low | -8.8568 | -7.7039 | 0.50 | 0.00 | 0.01373919 | 350 | False |
| 0.1 | 0.5 | lowmid | -5.8517 | -4.0415 | 0.50 | 0.00 | 0.01952356 | 459 | False |
| 0.25 | 0.1 | low | -13.7829 | -14.4547 | 1.00 | 0.50 | 0.00051159 | 14 | True |
| 0.25 | 0.1 | lowmid | -13.7490 | -14.4244 | 1.00 | 0.50 | 0.00073594 | 18 | True |
| 0.25 | 0.25 | low | -13.2445 | -13.1146 | 1.50 | 0.50 | 0.00469571 | 114 | True |
| 0.25 | 0.25 | lowmid | -13.0289 | -12.9181 | 1.50 | 0.50 | 0.00551359 | 127 | True |
| 0.25 | 0.5 | low | -12.7995 | -11.6441 | 3.00 | 0.50 | 0.00904231 | 198 | False |
| 0.25 | 0.5 | lowmid | -12.5803 | -11.4442 | 3.00 | 0.50 | 0.00986019 | 211 | False |
| 0.5 | 0.1 | low | -14.0506 | -14.4871 | 2.00 | 0.50 | 0.00000000 | 0 | False |
| 0.5 | 0.1 | lowmid | -14.0506 | -14.4871 | 2.00 | 0.50 | 0.00000000 | 0 | False |
| 0.5 | 0.25 | low | -14.0474 | -14.4822 | 2.00 | 0.50 | 0.00008379 | 2 | True |
| 0.5 | 0.25 | lowmid | -14.0474 | -14.4822 | 2.00 | 0.50 | 0.00008379 | 2 | True |
| 0.5 | 0.5 | low | -14.0142 | -14.4306 | 2.00 | 0.50 | 0.00046863 | 9 | True |
| 0.5 | 0.5 | lowmid | -14.0142 | -14.4306 | 2.00 | 0.50 | 0.00046863 | 9 | True |

## Truy xuất và provenance

| Artifact / scope | Value |
| --- | --- |
| Analysis commit | `545a83b11f893293aa8d6c733bd4779f77ae8aff` |
| Worker commit | `f75e6231ff7a29396d9eb66ee5591fdb1107f956` |
| Preregistration commit | `225e06509ebdd72744ae82c6b75a40ff6f2c16d1` |
| Source fingerprint | `ce5acf9334d4f683fdc7757d53b0d5be0ce80cc5a6fd4a4340e30c9357e79721` |
| CAL input SHA-256 | `db9fd6eef315657b33f9e0d557260c7af2a89857633193e6d41c83b056a7983d` |
| Primary index SHA-256 | `35b42f2874e909fddfc38363c6d0b47ef8a5e85abfcc4adcbea40eaaabb4150d` |
| Result SHA-256 | `366d1a8a85cf0a75a5b03f962e9f5162a7775c8b6ce240c9cd9bf3694b1eb861` |
| Bootstrap unit | source video; all QPs, arms and primary analyzers paired |
| Bootstrap seed / draws | 20261009 / 2000 |
| Worker Python / Torch / Torchvision | 3.12.13 / 2.10.0+cu128 / 0.25.0+cu128 |

Archive, records, manifest và log gốc được giữ trong `results/v12_lowqp/raw/shardN/`; artifact manifest chứa SHA-256 của toàn bộ file. Metric BD-rate và logic bootstrap không đổi. Mỗi số trong báo cáo được lưu trong calibration_result.json hoặc assessment.json.

## Cập nhật sau freeze: MC3 đã đo

[Audit MC3 trên cùng CAL](RESULTS_V12_MC3_CAL.md): -3.21 [-7.75; +0.39]%; component `FAIL_ON_REUSED_CAL`. Các mục CHƯA ĐO phía trên mô tả trạng thái tại freeze CAL; nay MC3 CAL đã có số, DEV/holdout mới vẫn CHƯA ĐO. Policy/choices và JSON CAL gốc không thay đổi.
