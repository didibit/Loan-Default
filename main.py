
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import model  # Import the model from model.py
import train  # Import training logic from train.py
import test  # Import testing logic from test.py

# Hyperparameters and configurations
batch_size = 32
learning_rate = 0.001
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate a synthetic dataset (replace with actual dataset loading logic)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                             torch.tensor(y_train, dtype=torch.long))
test_data = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                            torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model_instance = model.MyModel()  # Replace 'MyModel' with the actual model class name
model_instance.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)

# Train the model
print("Training the model...")
train.train_model(model_instance, train_loader, criterion, optimizer, device, epochs)

# Test the model
print("Evaluating the model...")
test.evaluate_model(model_instance, test_loader, device)
