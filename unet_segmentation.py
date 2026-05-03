import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim

# ==========================================
# 1. GESTION DES DONNÉES (Dataset Custom)
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # On récupère la liste des fichiers (identiques dans les deux dossiers)
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # Chargement en RGB pour l'image et Grayscale pour le masque
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# ==========================================
# 2. ARCHITECTURE U-NET
# ==========================================
# 
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        b = self.bottleneck(self.pool(s2))
        d2 = self.up2(b)
        d2 = torch.cat((s2, d2), dim=1) # Skip Connection
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((s1, d1), dim=1) # Skip Connection
        d1 = self.dec1(d1)
        return torch.sigmoid(self.final(d1))

# ==========================================
# 3. CONFIGURATION ET LANCEMENT
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "segmentation/image/img"
MASK_DIR = "segmentation/mask/mask"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = SegmentationDataset(IMG_DIR, MASK_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss() # Binary Cross Entropy pour les masques N&B

print(f"Prêt à entraîner sur {len(dataset)} paires image/masque.")

# Boucle d'entraînement simplifiée
for epoch in range(1):
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch 1 terminée. Loss: {loss.item():.4f}")