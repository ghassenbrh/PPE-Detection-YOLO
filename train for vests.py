import torch
from ultralytics import YOLO


def train_complete_ppe():
    # Détection automatique du GPU
    device = 0 if torch.cuda.is_available() else 'cpu'

    # Charger le modèle précédent (Masks + Gloves)
    last_checkpoint = r"C:\Users\User\Desktop\projet de securite industrielle\runs\detect\gloves and masks_safety_final\weights\best.pt"
    model = YOLO(last_checkpoint)

    # Lancer l'entraînement avec les gilets
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=512,  # On garde 512 pour la fluidité webcam
        batch=48,
        device=device,
        name='vests gloves and masks_safety_final'
    )


if __name__ == '__main__':
    train_complete_ppe()