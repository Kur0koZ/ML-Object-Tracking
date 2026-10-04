"""
zone_counting_fixed.py
นับรถที่เข้าพื้นที่ (Zone Counting) ด้วย YOLO + ByteTrack

หมายเหตุ: โค้ดนี้ประกอบใหม่จากชิ้นส่วนที่ OCR ได้ในสไลด์ (หน้า 31)
- ค่า ZONE ในสไลด์อ่านไม่ออก ต้องแก้ให้ตรงกับวิดีโอของคุณเอง
- ส่วนที่ไม่ปรากฏในสไลด์ ผมเติมให้ตามลอจิกในหน้า 30 (centroid + set() กันนับซ้ำ)

วิธีรัน:  python zone_counting_fixed.py
"""

import cv2
from ultralytics import YOLO

# ---------------- ตั้งค่า ----------------
MODEL_PATH = "runs/detect/train-2/weights/yolo26n.pt"                # weights ที่เทรนไว้ (runs/detect/train/weights/best.pt)
SOURCE = "cats_video.mp4"                  # วิดีโอที่จะนับรถ
OUT_PATH = "output_counted.mp4"       # ไฟล์ผลลัพธ์
ZONE = (480, 720, 1440, 1050)            # (x1, y1, x2, y2) <-- ค่าตัวอย่าง ต้องแก้ให้ตรงกับภาพ
TRACKER = "my_tracker.yaml"
MIN_FRAMES = 15
# ------------------------------------------


def point_in_zone(cx, cy, zone):
    """เช็คว่าจุด (cx, cy) อยู่ในกรอบสี่เหลี่ยมหรือไม่"""
    x1, y1, x2, y2 = zone
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return xmin <= cx <= xmax and ymin <= cy <= ymax


def main():
    model = YOLO(MODEL_PATH)

    # อ่านคุณสมบัติของวิดีโอ เพื่อสร้างไฟล์ผลลัพธ์ขนาดเท่ากัน
    cap = cv2.VideoCapture(SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))

    counted_ids = set()   # เก็บ ID ที่นับแล้ว กันนับซ้ำ
    count = 0

    x1z, y1z, x2z, y2z = ZONE

    results = model.track(
        source=SOURCE,
        tracker="bytetrack.yaml",   # หรือ "botsort.yaml"
        persist=True,               # ให้ ID ต่อเนื่องระหว่างเฟรม
        stream=True,                # ประมวลผลทีละเฟรม ประหยัดหน่วยความจำ
        verbose=False,
        device=0,
        conf=0.4,
    )

    for r in results:
        frame = r.orig_img.copy()
        in_zone_now = 0

        # วาดกรอบพื้นที่นับ
        cv2.rectangle(frame, (x1z, y1z), (x2z, y2z), (0, 255, 255), 2)

        # เฟรมที่ยังไม่มี track (boxes.id เป็น None) ให้ข้ามการนับ
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()

            for (x1, y1, x2, y2), track_id in zip(boxes, ids):
                # จุดศูนย์กลาง (centroid) ของ bounding box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                inside = point_in_zone(cx, cy, ZONE)

                # อยู่ในกรอบ และ ID นี้ยังไม่เคยถูกนับ -> นับ +1
                if inside and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    count += 1

                color = (0, 255, 0) if inside else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # แสดงยอดรวมบนภาพ
        cv2.putText(
            frame,
            f"Count: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    writer.release()
    print(f"\nTotal cats counted : {count}")
    print(f"Output saved to        : {OUT_PATH}")


if __name__ == "__main__":
    main()
