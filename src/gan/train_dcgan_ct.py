import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ========= SETTINGS =========
DATA_PATH = "CT_dataset_split/train"
SAVE_DIR = "generated_CT_images"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
LATENT_DIM = 100
EPOCHS = 30
DEVICE = torch.device("cpu")

# ========= DATA =========
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # CT → grayscale
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========= GENERATOR =========
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.ConvTranspose2d(LATENT_DIM, 512, 7, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# ========= DISCRIMINATOR =========
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*14*14, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ========= MODELS =========
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5,0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5,0.999))

print("\nStarting CT DCGAN Training...")

# ========= TRAIN LOOP =========
for epoch in range(EPOCHS):

    for i,(real_imgs,_) in enumerate(loader):

        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size,1).to(DEVICE)
        fake_labels = torch.zeros(batch_size,1).to(DEVICE)

        # ---- Train D ----
        optimizer_D.zero_grad()

        outputs_real = D(real_imgs)
        loss_real = criterion(outputs_real, real_labels)

        z = torch.randn(batch_size,LATENT_DIM,1,1).to(DEVICE)
        fake_imgs = G(z)

        outputs_fake = D(fake_imgs.detach())
        loss_fake = criterion(outputs_fake,fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # ---- Train G ----
        optimizer_G.zero_grad()

        outputs = D(fake_imgs)
        loss_G = criterion(outputs,real_labels)

        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss_D:{loss_D.item():.4f} Loss_G:{loss_G.item():.4f}")

    if (epoch+1)%5==0:
        save_image(fake_imgs[:16],
                   f"{SAVE_DIR}/epoch_{epoch+1}.png",
                   normalize=True)

print("\nCT DCGAN Training Finished.")