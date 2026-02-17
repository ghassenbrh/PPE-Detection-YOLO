import cv2
from ultralytics import YOLO

# Charger le modèle final
model = YOLO(r"C:\Users\User\Desktop\projet de securite industrielle\runs\detect\ppe_total_detector\weights\best.pt")

# Mapping des couleurs (BGR)
# Vert = Sécurisé / Rouge = Alerte / Orange = Attention
colors = {
    'with_mask': (0, 255, 0), 'Gloves': (0, 255, 0), 'safety_vest': (0, 255, 0),
    'without_mask': (0, 0, 255), 'NO-Gloves': (0, 0, 255), 'no_safety_vest': (0, 0, 255),
    'incorrectly_worn_mask': (0, 165, 255)
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Inférence optimisée à 512
        results = model.predict(source=frame, imgsz=512, conf=0.6, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                color = colors.get(label, (255, 255, 255))

                # Dessin de la box et du label
                b = box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)
                cv2.putText(frame, f"{label}", (int(b[0]), int(b[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("IA de Surveillance EPI - Masques, Gants, Gilets", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()