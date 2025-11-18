import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

# ----------------------------
# 1. Initialize W&B
# ----------------------------
wandb.init(
    entity="florinacho", #TODO change with team name
    project="test", #TODO change with project name
    config={
        "epochs": 3,
        "batch_size": 64,
        "lr": 1e-3,
        "architecture": "SimpleCNN"
    }
)
config = wandb.config

# ----------------------------
# 2. Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda_status = {
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
}
wandb.log(cuda_status)
print(f"Using device: {cuda_status['cuda_device_name']}")

# ----------------------------
# 3. Data loading
# ----------------------------

x = torch.randn(1000, config.input_dim)
y = torch.randint(0, config.output_dim, (1000,))

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# ----------------------------
# 4. Define a simple CNN
# ----------------------------
model = nn.Sequential(
    nn.Linear(config.input_dim, config.hidden_dim),
    nn.ReLU(),
    nn.Linear(config.hidden_dim, config.output_dim)
)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# 5. Training loop
# ----------------------------
for epoch in range(config.epochs):
    model.train()
    total_loss, correct = 0.0, 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(dim=1) == target).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})

    print(f"Epoch {epoch+1}/{config.epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

wandb.finish()

