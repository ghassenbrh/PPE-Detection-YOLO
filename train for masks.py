from ultralytics import YOLO
import torch


def main():
    # Utilisation du GPU (NVIDIA)
    device = 0 if torch.cuda.is_available() else 'cpu'

    # 1. Charger YOLO11 (plus performant pour 3 classes)
    model = YOLO('yolov8n.pt')

    # 2. Entraînement
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=512,  # Taille optimisée pour ce dataset (vu sur votre capture)
        batch=16,  # Réduire à 8 si la mémoire GPU est pleine
        device=device,
        name='mask_safety_final'
    )


if __name__ == '__main__':
    main()