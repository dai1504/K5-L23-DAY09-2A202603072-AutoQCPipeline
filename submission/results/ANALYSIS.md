# Phân tích train raw và clean

## 1. Mức chênh mAP50 và giới hạn của hai seed

Với hai seed, `raw` đạt mAP50 trung bình **0,7258 ± 0,0002** và `clean` đạt **0,7294 ± 0,0024**. Chênh trung bình `clean − raw` là **+0,0036** (bảng tổng hợp làm tròn thành `+0,004`). Hai chênh theo seed đều dương: seed 0 là `+0,0021`, seed 1 là `+0,0052`.

Tín hiệu này nhỏ: nó lớn hơn độ lệch chuẩn riêng của raw, nhưng chỉ khoảng 1,5 lần độ lệch chuẩn gộp xấp xỉ 0,0024. Quan trọng hơn, chỉ có hai seed; độ lệch chuẩn của chính chênh seed là khoảng 0,0022 và khoảng tin cậy với n=2 rất rộng. Vì vậy có thể nói clean **có dấu hiệu** cải thiện mAP50, nhưng chưa thể kết luận cải thiện có ý nghĩa thống kê hay có khả năng lặp lại. Cần chạy thêm seed với cùng model, epoch, batch và tập test.

## 2. `raw` thực sự train trên bao nhiêu ảnh?

Hai file kết quả `raw_seed0.json` và `raw_seed1.json` ghi `train_images: 224`. Đây là số ảnh script đã **materialize** vào thư mục `work/train/raw/train/images`, không phải một bằng chứng rằng Ultralytics đã nhận đủ 224 ảnh. `scripts/materialize_dataset.py` đếm file trước khi Ultralytics kiểm tra ảnh/nhãn; Ultralytics có thể bỏ ảnh hỏng, dòng nhãn không parse được, ID lớp sai hoặc box tràn.

Log cảnh báo ở cell train của Colab không được lưu trong `submission/results/`, nên không thể khôi phục trung thực số ảnh Ultralytics thực sự dùng từ các artifact hiện có. Báo cáo này không thay 224 bằng một số suy đoán. Khi chạy lại, cần lưu toàn bộ stdout của cell 6 hoặc cache/summary dataset của Ultralytics, rồi ghi số “scanning … images, … corrupt” vào đây. Điều này đặc biệt quan trọng vì `02_error_list_v1.csv` đã phát hiện 2 ảnh hỏng, 3 dòng sai định dạng, 6 dòng ID lớp sai và 8 box tràn ảnh.

## 3. Rò rỉ test trong raw

Có. `01_intake.csv` và `02_error_list_v1.csv` ghi **11** lỗi `LEAK_TEST`, mỗi lỗi chỉ rõ một ảnh raw trùng ảnh trong `data/test`; test có 100 ảnh, nên ít nhất **11/100 = 11%** ảnh đánh giá bị rò rỉ vào raw train.

Không thể suy ra raw bị thổi phồng chính xác bao nhiêu điểm mAP50 chỉ từ tỷ lệ 11%: mAP là trung bình theo lớp/điểm confidence và ảnh trùng có độ khó khác nhau. Chênh +0,0036 của clean so với raw cũng không phải “mức rò rỉ”, vì clean đồng thời bỏ ảnh trùng, sửa nhãn và có ít ảnh train hơn. Cách kiểm đúng là train thêm một biến thể `raw_no_leak` chỉ loại 11 ảnh `LEAK_TEST` (giữ nguyên mọi ảnh/nhãn khác), cùng hai seed và siêu tham số; so sánh raw với `raw_no_leak` trên cùng test. Có thể kiểm độc lập bằng md5/phash giữa train và test, đồng thời mở các cặp do pipeline báo.

## 4. Lớp hưởng lợi nhiều nhất

Theo AP50 trung bình của hai seed, **bus** tăng nhiều nhất: **0,6949 → 0,7153, +0,0204**. Car cũng tăng `+0,0155`, person tăng `+0,0057`, còn truck giảm `−0,0270`.

| Lớp | AP50 raw | AP50 clean | Chênh clean − raw |
|---|---:|---:|---:|
| car | 0,7375 | 0,7530 | +0,0155 |
| person | 0,8220 | 0,8277 | +0,0057 |
| truck | 0,6484 | 0,6215 | −0,0270 |
| bus | 0,6949 | 0,7153 | +0,0204 |

Raw có lỗi tên lớp `Bus` (viết hoa) trong `data.yaml`, cùng các lỗi ID lớp, dòng hỏng, box tràn/zero, box trùng và các nghi vấn `POSSIBLE_WRONG_CLASS`. Việc chuẩn hoá tên lớp và loại/sửa các nhãn lỗi là lời giải thích hợp lý cho tăng điểm bus, nhưng không chứng minh quan hệ nhân quả riêng cho bus: chỉ hai seed, nhiều thay đổi xảy ra cùng lúc, và truck giảm. Cần ablation theo từng loại sửa hoặc thêm seed để kết luận theo lớp.

## 5. Clean còn lỗi gì và có phải chặn trên?

Không. Clean hiện là bản **Hold**, chưa phải “dữ liệu sạch” theo nghĩa chặn trên. Sau sửa và người duyệt mô phỏng, `07_error_list_v3.csv` vẫn có 66 cảnh báo: 20 `POSSIBLE_FALSE_POS`, 18 `POSSIBLE_LOOSE_BOX`, 12 `POSSIBLE_MISSING`, 7 `BOX_TINY`, 3 `EMPTY_LABEL`, 3 `BOX_HUGE` và 3 `POSSIBLE_WRONG_CLASS`. Hàng đợi ban đầu có **121 ảnh**, vượt ngân sách duyệt **80 ảnh**; gate còn **34/193 = 17,62%** ảnh cảnh báo chưa duyệt, cao hơn ngưỡng Hold 10%.

Vì vậy kết quả clean chỉ là một biến thể đã QC một phần. Chặn trên đáng tin cần có nhãn người cho toàn bộ hàng đợi còn lại (hoặc biến thể `oracle` dùng `review_reference` theo thiết kế lab), sau đó train cùng protocol và nhiều seed. Không được diễn giải mAP của clean hiện tại là hiệu quả tối đa của làm sạch dữ liệu.
