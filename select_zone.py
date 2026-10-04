"""
select_zone.py
ช่วยหาค่า ZONE จากคลิป โดยเลื่อนดูเฟรมที่เห็นแมวชัด ๆ ก่อน แล้วค่อยลากกรอบ

วิธีรัน:  python select_zone.py

ขั้นที่ 1: เลื่อนแถบ "frame" ไปยังเฟรมที่เห็นพื้นที่ที่ต้องการนับ
           แล้วกด Enter (หรือ Space) เพื่อใช้เฟรมนี้  (กด Esc เพื่อออก)
ขั้นที่ 2: ลากเมาส์ครอบพื้นที่ที่ต้องการนับ แล้วกด Enter
           (กด c เพื่อยกเลิก)
"""

import cv2

VIDEO = "cat_video.mp4"   # ต้องเป็นคลิปเดียวกับ SOURCE ในสคริปต์หลัก
MAX_W = 1280              # ย่อภาพให้พอดีหน้าจอ (พิกัดจะถูกแปลงกลับเป็นขนาดจริงให้เอง)
WIN = "Pick frame (slider, Enter = use this frame, Esc = quit)"


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"เปิดคลิปไม่ได้: {VIDEO}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = read_frame(cap, 0)
    if first is None:
        raise SystemExit("อ่านเฟรมจากคลิปไม่ได้")

    h, w = first.shape[:2]
    scale = min(1.0, MAX_W / w)

    def resize(img):
        return cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img

    # ---------- ขั้นที่ 1: เลือกเฟรม ----------
    cv2.namedWindow(WIN)
    cv2.createTrackbar("frame", WIN, 0, max(total - 1, 1), lambda v: None)

    chosen = first
    last_pos = -1
    while True:
        pos = cv2.getTrackbarPos("frame", WIN)
        if pos != last_pos:
            frame = read_frame(cap, pos)
            if frame is not None:
                chosen = frame
            last_pos = pos
        cv2.imshow(WIN, resize(chosen))

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):      # Enter หรือ Space
            break
        if key == 27:            # Esc
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("ออกจากโปรแกรม")

    cap.release()
    cv2.destroyAllWindows()

    # ---------- ขั้นที่ 2: ลากกรอบ ----------
    x, y, bw, bh = cv2.selectROI(
        "Select zone (Enter = confirm, c = cancel)", resize(chosen), False
    )
    cv2.destroyAllWindows()

    if bw == 0 or bh == 0:
        raise SystemExit("ยกเลิกหรือไม่ได้เลือกกรอบ")

    # แปลงพิกัดกลับเป็นขนาดจริงของวิดีโอ
    x1, y1 = int(x / scale), int(y / scale)
    x2, y2 = int((x + bw) / scale), int((y + bh) / scale)

    print(f"Video size: {w} x {h}")
    print(f"ZONE = ({x1}, {y1}, {x2}, {y2})")


if __name__ == "__main__":
    main()
