import torch
from ultralytics import YOLO

def main():
    # Vérification de l'accélération matérielle (CUDA)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Dispositif d'entraînement : {device}")

    # 1. Charger votre modèle de masque précédent comme point de départ
    model = YOLO( r"C:\Users\User\Desktop\projet de securite industrielle\runs\detect\mask_safety_final\weights\best.pt")

    # 2. Lancer l'entraînement global sur le dataset fusionné
    model.train(
        data='data.yaml',      # Doit contenir les 5 classes
        epochs=100,            # 100 itérations pour stabiliser les gants
        imgsz=512,             # Résolution optimisée (votre choix)
        batch=48,              # Batch plus élevé car 512 consomme moins de VRAM
        device=device,         # Force l'utilisation du GPU
       ## patience=20,           # Arrêt si le modèle stagne
        save=True,
        name='gloves and masks_safety_final'
    )

    print("Entraînement terminé. Modèle : runs/detect/ppe_mask_gloves_512/weights/best.pt")

if __name__ == '__main__':
    main()