# 🛡️ PPE Detection System (Industrial Safety) - YOLOv8

Ce projet est un système de surveillance en temps réel basé sur l'IA pour détecter le port des Équipements de Protection Individuelle (EPI).

## 🚀 Fonctionnalités
Détection simultanée de **7 classes** critiques :
- **Visage :** `with_mask`, `without_mask`, `incorrectly_worn_mask`
- **Mains :** `Gloves`, `NO-Gloves`
- **Corps :** `safety_vest`, `no_safety_vest`

## 📊 Évolution du projet
Le modèle a été entraîné de manière incrémentale par transfert d'apprentissage :
1. **Étape 1 :** Détection de masques avec correction des faux positifs (3419 images).
2. **Étape 2 :** Extension à la détection de gants.
3. **Étape 3 :** Finalisation avec le port du gilet de sécurité.

## 🛠️ Spécifications techniques
- **Modèle :** YOLOv8 (Inférence ultra-rapide pour webcam).
- **Résolution :** 512x512 pixels.
- **Optimisation :** Entraînement sur GPU avec augmentations de données (Flip, Brightness).

## 🖥️ Installation & Usage
1. Cloner le repo : `git clone https://github.com/ghassenbrh/PPE-Detection-YOLO.git`
2. Installer les dépendances : `pip install ultralytics opencv-python`
3. Lancer la détection : `python main.py`
