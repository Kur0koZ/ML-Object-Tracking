from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"C:\ML-ObjectTracking\Dataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2,
        device=0,
    )