# zone_counting.py

from collections import defaultdict

import cv2
from ultralytics import YOLO

# ---------------- ตั้งค่า ----------------
MODEL_PATH = "runs/detect/train-2/weights/yolo26n.pt"
SOURCE = "cat_video02.mp4"
OUT_PATH = "cat_counted2.mp4"
#ZONE = (480, 720, 1440, 1050)     # (x1, y1, x2, y2) ขนาดภาพจริงของคลิป
ZONE = (480, 430, 1440, 1030)
TRACKER = "my_tracker.yaml"       # หรือ "bytetrack.yaml" ถ้าไม่ได้ปรับเอง
CONF = 0.1                        # ต่ำไว้ให้ tracker กรองเอง (ดู my_tracker.yaml)
MIN_CONF_SHOW = 0.5               # กรอบที่มั่นใจต่ำกว่านี้ไม่นำมาใช้นับ/วาด
DEDUP_IOMIN = 0.8                 # กรอบที่ทับกรอบอื่น >= สัดส่วนนี้ (ของกรอบเล็ก) จะเก็บแค่อันที่มั่นใจสูงกว่า
                                  # ตั้ง 1.0 เพื่อปิด (แนะนำปิดถ้าคลิปมีแมวหลายตัวซ้อนกัน)
DEVICE = 0                        # ใช้ "cpu" ถ้าเครื่องไม่มี GPU
MIN_FRAMES = 20                   # ต้องอยู่ในกรอบกี่เฟรมจึงนับว่าเป็นแมวจริง
COUNT_MODE = "peak"             # "unique" = แมวที่เดินเข้ามา, "peak" = จำนวนสูงสุดที่เห็นพร้อมกัน
POINT_MODE = "bottom"             # "bottom" = กึ่งกลางขอบล่างของกรอบ, "center" = กึ่งกลางกรอบ
MERGE_GAP_FRAMES = 90             # ID เก่าหายไปไม่เกินกี่เฟรมจึงจะรวมกับ ID ใหม่ (90 ~ 3 วินาที)
MERGE_DIST_RATIO = 0.15           # ระยะห่างสูงสุดที่รวมได้ เป็นสัดส่วนของความกว้างภาพ
# ------------------------------------------


def point_in_zone(cx, cy, zone):
    x1, y1, x2, y2 = zone
    return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)


def dedup_boxes(items, thr):
    """
    items: list ของ (box[x1,y1,x2,y2], raw_id, conf)
    ถ้ากรอบสองกรอบทับกันมาก (พื้นที่ทับ / พื้นที่กรอบที่เล็กกว่า >= thr)
    ให้เก็บเฉพาะกรอบที่ conf สูงกว่า คืน list ในรูปแบบเดิม
    """
    if thr >= 1.0:
        return items
    order = sorted(range(len(items)), key=lambda k: -items[k][2])
    kept = []
    for k in order:
        bx = items[k][0]
        ax = max(0.0, bx[2] - bx[0]) * max(0.0, bx[3] - bx[1])
        dup = False
        for j in kept:
            by = items[j][0]
            iw = min(bx[2], by[2]) - max(bx[0], by[0])
            ih = min(bx[3], by[3]) - max(bx[1], by[1])
            if iw <= 0 or ih <= 0:
                continue
            ay = max(0.0, by[2] - by[0]) * max(0.0, by[3] - by[1])
            if (iw * ih) / max(1e-9, min(ax, ay)) >= thr:
                dup = True
                break
        if not dup:
            kept.append(k)
    return [items[k] for k in sorted(kept)]


def resolve_ids(dets, frame_idx, state, max_dist, max_gap):
    """
    แปลง ID ดิบของ tracker เป็น "ID แมว" ที่รวมตัวที่ขาดช่วงแล้ว
    dets  : list ของ (raw_id, cx, cy) ของเฟรมนี้ (cx, cy = กึ่งกลางกรอบ)
    state : dict {"alias": {raw: cat}, "last": {cat: (frame, cx, cy)}}
    คืนค่า: dict {raw_id: cat_id}
    """
    alias, last = state["alias"], state["last"]

    # แมวที่ปรากฏในเฟรมนี้อยู่แล้ว (ผ่าน ID ที่เคยแมปไว้)
    visible = {alias[r] for r, _, _ in dets if r in alias}

    for raw, cx, cy in dets:
        if raw in alias:
            continue
        # หาแมวที่ "หายไป" ใกล้ที่สุดภายในช่วงเวลาและระยะที่กำหนด
        best, best_d = None, None
        for cat, (f, lx, ly) in last.items():
            if cat in visible or frame_idx - f > max_gap:
                continue
            d = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if d <= max_dist and (best_d is None or d < best_d):
                best, best_d = cat, d
        alias[raw] = best if best is not None else raw
        visible.add(alias[raw])

    for raw, cx, cy in dets:
        last[alias[raw]] = (frame_idx, cx, cy)

    return {raw: alias[raw] for raw, _, _ in dets}


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    state = {"alias": {}, "last": {}}
    cat_number = {}                      # cat_id -> เลขที่แสดง (1, 2, 3, ...)
    frames_in_zone = defaultdict(int)    # cat_id -> จำนวนเฟรมที่อยู่ในกรอบ
    confirmed = set()                    # cat_id ที่ผ่านเกณฑ์ MIN_FRAMES
    raw_ids_seen = set()                 # ID ดิบทั้งหมดที่ tracker ให้มา (ไว้เทียบ)
    peak = 0
    max_dist = MERGE_DIST_RATIO * w

    x1z, y1z, x2z, y2z = ZONE

    results = model.track(
        source=SOURCE,
        tracker=TRACKER,
        persist=True,
        stream=True,
        verbose=False,
        device=DEVICE,
        conf=CONF,
    )

    for frame_idx, r in enumerate(results):
        frame = r.orig_img.copy()
        cv2.rectangle(frame, (x1z, y1z), (x2z, y2z), (0, 255, 255), 2)

        current_confirmed = 0

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            raw_ids = r.boxes.id.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()

            # คัดเฉพาะกรอบที่มั่นใจพอ
            cand = [(b, i, c) for b, i, c in zip(boxes, raw_ids, confs) if c >= MIN_CONF_SHOW]
            cand = dedup_boxes(cand, DEDUP_IOMIN)
            items = [(b, i) for b, i, c in cand]

            dets = [(i, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, i in items]
            mapping = resolve_ids(dets, frame_idx, state, max_dist, MERGE_GAP_FRAMES)

            for (x1, y1, x2, y2), raw in items:
                raw_ids_seen.add(raw)
                cat = mapping[raw]
                if cat not in cat_number:
                    cat_number[cat] = len(cat_number) + 1

                cx = int((x1 + x2) / 2)
                cy = int(y2) if POINT_MODE == "bottom" else int((y1 + y2) / 2)
                inside = point_in_zone(cx, cy, ZONE)

                if inside:
                    frames_in_zone[cat] += 1
                    if frames_in_zone[cat] >= MIN_FRAMES:
                        confirmed.add(cat)

                is_confirmed = inside and cat in confirmed
                if is_confirmed:
                    current_confirmed += 1

                # เขียว = นับแล้ว, ส้ม = อยู่ในกรอบแต่ยังไม่ครบ MIN_FRAMES, แดง = นอกกรอบ
                color = (0, 255, 0) if is_confirmed else ((0, 165, 255) if inside else (0, 0, 255))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(frame, f"cat {cat_number[cat]}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        peak = max(peak, current_confirmed)
        count = peak if COUNT_MODE == "peak" else len(confirmed)

        cv2.putText(frame, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"now in zone: {current_confirmed}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"\nCOUNT_MODE              : {COUNT_MODE}")
    print(f"Cats counted            : {peak if COUNT_MODE == 'peak' else len(confirmed)}")
    print(f"Peak cats at same time  : {peak}")
    print(f"Confirmed cats (merged) : {len(confirmed)}")
    print(f"Raw tracker IDs seen    : {len(raw_ids_seen)}  {sorted(raw_ids_seen)}")
    print(f"ID aliases (raw -> cat) : {state['alias']}")
    print(f"Output saved to         : {OUT_PATH}")


if __name__ == "__main__":
    main()
