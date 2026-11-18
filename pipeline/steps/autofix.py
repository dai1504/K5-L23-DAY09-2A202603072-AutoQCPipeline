"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]
    fixes = []
    remove_codes = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    removed = set()
    for issue in errors:
        if issue.code in remove_codes and issue.image not in removed:
            image = ds.image_path(issue.image)
            if image:
                image.unlink()
            label = ds.label_path(issue.image)
            if label.exists():
                label.unlink()
            removed.add(issue.image)
            fixes.append(dict(image=issue.image, line=issue.line or "", action="remove_image",
                              before=issue.code, after="đã xoá ảnh và nhãn"))

    # Lines are numbered in the original file, so collect all removals before rewriting it.
    drops = defaultdict(set)
    clips = defaultdict(set)
    for issue in errors:
        if issue.image in removed or issue.line is None:
            continue
        if issue.code in {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}:
            drops[issue.image].add(issue.line)
        elif issue.code == "BOX_OUT_OF_IMAGE":
            clips[issue.image].add(issue.line)
    for stem in sorted(set(drops) | set(clips)):
        path = ds.label_path(stem)
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8", errors="replace").splitlines()
        rewritten = []
        for no, before in enumerate(original, 1):
            if no in drops[stem]:
                fixes.append(dict(image=stem, line=no, action="drop_line", before=before, after=""))
                continue
            if no in clips[stem]:
                try:
                    cls, x, y, w, h = parse_yolo_line(before)
                    x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
                    x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
                    if x2 <= x1 or y2 <= y1:
                        fixes.append(dict(image=stem, line=no, action="drop_line", before=before, after=""))
                        continue
                    after = f"{cls} {(x1 + x2) / 2:.10g} {(y1 + y2) / 2:.10g} {x2 - x1:.10g} {y2 - y1:.10g}"
                    fixes.append(dict(image=stem, line=no, action="clip_box", before=before, after=after))
                    rewritten.append(after)
                    continue
                except ValueError:
                    # A simultaneous malformed-line issue will normally have put it in drops.
                    pass
            rewritten.append(before)
        path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")

    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        before = data.get("names", {})
        data["names"] = {i: name for i, name in enumerate(cfg["classes"])}
        ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        fixes.append(dict(image="data.yaml", line="", action="rename_classes", before=str(before), after=str(data["names"])))
    for issue in errors:
        if issue.code == "MISSING_LABEL_FILE" and issue.image not in removed:
            fixes.append(dict(image=issue.image, line=issue.line or "", action="needs_label",
                              before="(không có file)", after="chuyển người gán"))
    return fixes
