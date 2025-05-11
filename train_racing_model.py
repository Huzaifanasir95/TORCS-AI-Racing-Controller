import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the telemetry data
try:
    data = pd.read_csv('data/Dataset.csv')
except FileNotFoundError:
    print("Error: 'Dataset.csv' not found in the data directory.")
    exit(1)

# Define features and targets
features = [
    'Angle', ' CurrentLapTime', ' Damage', ' DistanceFromStart', ' DistanceCovered',
    ' FuelLevel', ' Gear', ' LastLapTime', ' RPM', ' SpeedX', ' SpeedY', ' SpeedZ',
    ' Track_1', 'Track_2', 'Track_3', 'Track_4', 'Track_5', 'Track_6', 'Track_7',
    'Track_8', 'Track_9', 'Track_10', 'Track_11', 'Track_12', 'Track_13', 'Track_14',
    'Track_15', 'Track_16', 'Track_17', 'RacePosition'
]

targets = ['Steering', ' Acceleration', 'Braking']

# Verify required columns
required_columns = features + targets
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Error: Missing columns in Dataset.csv: {missing_columns}")
    exit(1)

# Data preprocessing
print("Preprocessing data...")

# Remove any rows with missing values
data = data.dropna(subset=features + targets)

# Filter for high-quality data
data = data[data[' SpeedX'] > 0]  # Moving forward
data = data[data[' Damage'] < 1000]  # Not heavily damaged

# Check if filtered data is empty
if data.empty:
    print("Error: No data remains after filtering.")
    exit(1)

print(f"Filtered data contains {len(data)} samples.")

# Extract features and targets
X = data[features].values
y = data[targets].values

# Handle missing values
X = np.nan_to_num(X, nan=0.0)
y = np.nan_to_num(y, nan=0.0)

# Normalize the features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Save the scaler
joblib.dump(scaler_X, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define an improved neural network model
class RacingModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RacingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        x = torch.tanh(x)  # Normalize outputs to [-1, 1]
        return x

# Initialize the model
input_size = X_train.shape[1]
output_size = y_train.shape[1]
model = RacingModel(input_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
num_epochs = 100
best_val_loss = float('inf')
patience = 15
patience_counter = 0
train_losses = []
val_losses = []
current_lr = optimizer.param_groups[0]['lr']

print("Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    old_lr = current_lr
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    if old_lr != current_lr:
        print(f"\nLearning rate decreased from {old_lr:.6f} to {current_lr:.6f}")
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'best_racing_model.pth')
        print("Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.savefig('training_history.png')
plt.close()

print("Training completed. Best model saved as 'best_racing_model.pth'")
print("Training history plot saved as 'training_history.png'")