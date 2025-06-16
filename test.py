import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import sys
from einops import rearrange
from vit_pytorch import ViT

# ==== Définition du modèle ====
class VideoViT(nn.Module):
    def __init__(self, num_classes, frames=32, image_size=224, dim=1024, depth=3, heads=8):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=16,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=2048,
            channels=3,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.frames = frames

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.vit(x)
        x = rearrange(x, '(b t) d -> b t d', b=b, t=t)
        x = x.mean(dim=1)
        return x

# ==== Paramètres ====
model_path = r"C:/Users/hp/Desktop/pfe/video_vit_epoch_30.pth"
video_path = r"C:/Users/hp/Desktop/pfe/vid1.mp4"
num_classes = 4
labels = ['walking', 'Sitting', 'Standing Still', 'Meet and Split']
num_frames = 32
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Prétraitement ====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== Charger le modèle ====
model = VideoViT(num_classes=num_classes, frames=num_frames, image_size=img_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ==== Lecture vidéo ====
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f" Impossible d'ouvrir la vidéo : {video_path}")
    sys.exit()

frame_buffer = []

print(" Lecture en cours... Appuie sur Q pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_frame = transform(rgb)
    frame_buffer.append(tensor_frame)

    if len(frame_buffer) >= num_frames:
        clip = torch.stack(frame_buffer[:num_frames]).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(clip)
            _, predicted = torch.max(output, 1)
            label = labels[predicted.item()]
            print(f" Action détectée : {label}")

        # Affichage sur la vidéo
        cv2.putText(frame, f"Action: {label}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Rectangle approximatif autour du corps
        h, w, _ = frame.shape
        x1, y1 = int(w*0.3), int(h*0.2)
        x2, y2 = int(w*0.7), int(h*0.9)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        frame_buffer = []  # Réinitialiser le buffer

    # Afficher chaque frame
    cv2.imshow("Video Action Recognition", frame)

    # Sortie avec Q
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
