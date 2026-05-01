import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

model = Autoencoder().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    loss_total = 0

    for images, _ in loader:
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images.view(-1, 28*28))
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    print(f"Epoch {epoch+1}, Loss: {loss_total:.4f}")

torch.save(model.state_dict(), "autoencoder.pth")