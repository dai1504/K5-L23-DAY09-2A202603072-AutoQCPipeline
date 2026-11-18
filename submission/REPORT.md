# Báo cáo Day 09 — Nguyễn Đại, 2A202603072

## 1. Kết luận trước

Pipeline kết luận **Hold**. Cửa vào quét 224 ảnh, nhận 193 ảnh và loại 31 ảnh hỏng, quá nhỏ, trùng hoặc rò rỉ sang kho test/validation. Sau khi máy tự sửa và người duyệt mô phỏng xem tối đa 80 ảnh, còn 34/193 ảnh có cảnh báo chưa duyệt (17,62%), cao hơn ngưỡng Hold 10%. Dataset nháp `clean_v1` đã được xuất để kiểm tra/thí nghiệm, nhưng chưa được phát hành.

## 2. Lỗi và phân việc

Số liệu từ `02_error_list_v1.csv`; “dòng” được dùng cho lỗi nhãn, “ảnh” cho các lỗi cấp ảnh.

| Mã lỗi | Số ảnh/dòng | Ai sửa | Vì sao |
|---|---:|---|---|
| `BAD_CLASS_ID` | 6 dòng | Máy | Xoá dòng có ID ngoài danh sách lớp. |
| `BAD_CLASS_NAME` | 2 lỗi / 1 file | Máy | Ghi lại `names` trong `data.yaml` theo cấu hình. |
| `BAD_FORMAT` | 3 dòng | Máy | Xoá dòng không parse được một cách an toàn. |
| `BOX_OUT_OF_IMAGE` | 8 dòng | Máy | Cắt box về biên ảnh hoặc xoá nếu không còn diện tích. |
| `BOX_ZERO` | 4 dòng | Máy | Xoá box không có diện tích. |
| `DUP_BOX` | 7 dòng | Máy | Xoá box lặp cùng lớp. |
| `MISSING_LABEL_FILE` | 6 ảnh | Người | Máy không thể bịa nhãn cho ảnh chưa có file nhãn. |
| `DUP_IMAGE_EXACT`, `DUP_IMAGE_NEAR`, `LEAK_TEST` | 8, 7, 11 ảnh | Máy | Loại ảnh trùng/rò rỉ để tránh làm bẩn dữ liệu và đánh giá. |
| `CORRUPT_IMAGE`, `BAD_SIZE` | 2, 3 ảnh | Máy | Không đủ điều kiện để train/đánh nhãn đáng tin. |
| `POSSIBLE_MISSING`, `POSSIBLE_WRONG_CLASS`, `POSSIBLE_FALSE_POS`, `POSSIBLE_LOOSE_BOX` | 52, 17, 26, 23 ảnh | Người | Đây là suy luận ngữ nghĩa từ model lớn hơn, chỉ nên cảnh báo. |

`05_review_queue.csv` xếp theo tổng trọng số mã lỗi và tăng ưu tiên theo confidence của checker. Sáu ảnh chưa có nhãn được đẩy lên đầu nhờ trọng số `NEEDS_LABEL=10`; tiếp theo là các nghi vấn sai lớp/thiếu nhãn. Với ngân sách 80 ảnh, hàng đợi 121 ảnh để lại 41 ảnh chưa được xem; sau mô phỏng còn 34 ảnh có cảnh báo.

## 3. Calibration

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | 16 | `DUP_IMAGE_NEAR`: 1,00 / 0,50 → 1,00 / 0,83 | Loại ảnh là hành động khó đảo ngược, nên giữ precision 1,00; 16 tăng recall cao nhất trước khi 20 làm recall tụt xuống 0,50. |
| `second_opinion.conf` | 0,25 | 0,60 | Missing: 0,50 / 1,00 → 0,57 / 1,00; Wrong class: 0,56 / 0,83 → 0,62 / 0,83 | Giữ khả năng phát hiện thiếu nhãn, đồng thời giảm báo động giả cho sai lớp. |

Calibration có thể đánh lừa vì chỉ gồm 39 ảnh và các lỗi được chèn có chủ đích. Bằng chứng trên raw: tăng `phash_dist` từ 2 lên 16 làm số ảnh bị loại ở cửa vào tăng từ 19 lên 31, còn hàng đợi vẫn có 121 ảnh. Vì vậy không thể dùng F1 của calib để khẳng định cùng mức hiệu quả trên dữ liệu thật; cần mở xem các cặp ảnh bị gộp trước khi Release.

## 4. Cửa chặn và điểm mù

Kiểm tra không có người duyệt bằng `--budget 0` trả mã thoát **20 (Rework)**: sau vòng sửa tự động vẫn còn 6 lỗi bắt buộc `MISSING_LABEL_FILE`, không xuất dataset. Điều này chứng minh lỗi mà máy không thể bịa nhãn không thể bị che bởi cảnh báo hay ngân sách duyệt.

Hai điểm mù rõ ràng là: (1) `POSSIBLE_LOOSE_BOX` ở ngưỡng đã chọn chỉ đạt recall 0,20 trên calibration, vì model lớn không luôn phân biệt box lỏng nhẹ; (2) `POSSIBLE_FALSE_POS` có precision 0,31, tức checker và nhãn người có thể bất đồng dù nhãn không sai. Ngoài ra, ảnh trùng bị biến đổi mạnh hơn phạm vi phash và lỗi mà cả model gán nhãn lẫn checker cùng mắc vẫn có thể lọt qua pipeline.

## 5. Train: raw vs clean

Chưa chạy. Phần train Colab là tùy chọn và cần chạy công bằng raw/clean với cùng model, epoch và ít nhất hai seed trước khi kết luận mAP50.

## 6. Quyết định và ghi chú AI

Quyết định hiện tại là **Hold**, không phải Release. Để đạt Release, cần tăng/nghiệm thu năng lực duyệt hơn 80 ảnh hoặc giảm hàng đợi bằng các ngưỡng được kiểm chứng thêm trên dữ liệu mới; đồng thời phải đảm bảo mọi ảnh thiếu nhãn đã được người gán. Không nên chỉ tăng ngưỡng để ép tỷ lệ cảnh báo xuống dưới 10%.

Đã dùng AI hỗ trợ hiện thực bốn module `label_format`, `gate`, `autofix` và `second_opinion` từ đặc tả trong `GUIDE.md`. Mã được đối chiếu với docstring, chạy test riêng từng module và chạy lại toàn bộ 79 test. Một lỗi môi trường đã gặp: `make calibrate` gọi Python hệ thống không có `PyYAML`; đã chạy đúng lệnh bằng `.venv/bin/python` thay vì thay đổi môi trường hệ thống.
