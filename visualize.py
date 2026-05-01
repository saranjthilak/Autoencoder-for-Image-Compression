import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from model import Autoencoder

model = Autoencoder()
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
image, _ = dataset[0]

with torch.no_grad():
    output = model(image.unsqueeze(0))

# Plot
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image.squeeze(), cmap='gray')

plt.subplot(1,2,2)
plt.title("Reconstructed")
plt.imshow(output.view(28,28), cmap='gray')

plt.show()