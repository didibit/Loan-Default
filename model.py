import torch
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc)
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# read data
data = pd.read_csv('training_loan_data.csv', header=1)
data.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

X = data_cleaned_no_duplicates.drop('bad_flag', axis=1)
y = data_cleaned_no_duplicates['bad_flag']

# Split the data into Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.25,          # 25% for Validation
    random_state=42,         # Ensures reproducibility
    stratify=y               # Maintains the distribution of 'bad_flag'
)

print(f"Training Set Size (before SMOTE): {X_train.shape[0]} samples")
print(f"Validation Set Size: {X_val.shape[0]} samples")

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the Training data
scaler.fit(X_train[x_numerical_columns])

# Transform the Training and Validation data
X_train_scaled = X_train.copy()
X_train_scaled[x_numerical_columns] = scaler.transform(X_train[x_numerical_columns])

X_val_scaled = X_val.copy()
X_val_scaled[x_numerical_columns] = scaler.transform(X_val[x_numerical_columns])

# Convert to NumPy arrays if they are not already
X_train_scaled = X_train_scaled.values.astype('float32')
X_val_scaled = X_val_scaled.values.astype('float32')
y_train = y_train.values
y_val = y_val.values

# Apply SMOTE on the training set to handle class imbalance
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

print(f"Training Set Size (after SMOTE): {X_train_resampled.shape[0]} samples")

# Convert the resampled data to tensors
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

X_train_tensor = X_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)

# Define batch size
batch_size = 5345

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 42),
            nn.ReLU(),
            nn.Linear(42, 35),
            nn.ReLU(),
            nn.Dropout(0.4896),
            nn.Linear(35, 1)  # Output layer for binary classification
        )
        
    def forward(self, x):
        return self.network(x)

input_dim = X_train_resampled.shape[1]
model = BinaryClassificationModel(input_dim).to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0073409)

num_epochs = 100
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * batch_X.size(0)
    
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation phase
    model.eval()
    epoch_val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            epoch_val_loss += loss.item() * batch_X.size(0)
            
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())
    
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    binary_preds = (all_preds >= 0.5).astype(int)
    
    accuracy = accuracy_score(all_labels, binary_preds)
    val_accuracies.append(accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {epoch_train_loss:.4f} | '
          f'Val Loss: {epoch_val_loss:.4f} | '
          f'Val Accuracy: {accuracy:.4f}')

# Final Evaluation on Validation Set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X)
        preds = torch.sigmoid(outputs)
        all_preds.append(preds.cpu())
        all_labels.append(batch_y.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

binary_preds = (all_preds >= 0.5).astype(int)

accuracy = accuracy_score(all_labels, binary_preds)
roc_auc = roc_auc_score(all_labels, all_preds)
auprc = average_precision_score(all_labels, all_preds)
f1 = f1_score(all_labels, binary_preds)

print("\nValidation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot Training and Validation Loss
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.legend()
plt.show()

# Save the model (optional)
torch.save(model.state_dict(), 'binary_classification_model_smote.pth')
print("\nModel saved as 'binary_classification_model_smote.pth'.")



from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from bayes_opt import BayesianOptimization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1).to(device)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

# Define a function to create and train the model given hyperparameters
def train_and_evaluate(hn1, hn2, dr, lr, batch_size):
    hn1 = int(hn1)
    hn2 = int(hn2)
    dr = float(dr)
    lr = float(lr)
    batch_size = int(batch_size)

    class TunedBinaryClassificationModel(nn.Module):
        def __init__(self, input_dim, hn1, hn2, dr):
            super(TunedBinaryClassificationModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, hn1)
            self.fc2 = nn.Linear(hn1, hn2)
            self.fc3 = nn.Linear(hn2, 1)
            self.dropout = nn.Dropout(dr)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    model = TunedBinaryClassificationModel(input_dim, hn1, hn2, dr).to(device)

    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create DataLoaders with the given batch_size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 100  
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds)
    return roc_auc


def objective(hn1, hn2, dr, lr, batch_size):
    auc = train_and_evaluate(hn1, hn2, dr, lr, batch_size)
    return auc  # We want to maximize AUC, and bayes_opt maximizes by default

# Define search space
pbounds = {
    'hn1': (32, 128),
    'hn2': (16, 64),
    'dr': (0.0, 0.5),
    'lr': (1e-5, 1e-2),
    'batch_size': (32, 6800)
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# Initialize and run optimization
# init_points=10, then n_iter=40 for a total of 50 evaluations
optimizer.maximize(init_points=10, n_iter=40)

print("Best Result:", optimizer.max)
best_params = optimizer.max['params']
print("Best Hyperparameters found by Bayesian Optimization:")
print(best_params)


## Check Val's performance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

# best_params from optimizer.max['params']
best_params ={'batch_size': 5345.879758680623, 'dr': 0.48961483633324776, 
              'hn1': 42.06460657796822, 'hn2': 35.822043448436816, 'lr': 0.0073409113435169565}
best_params = optimizer.max['params']
print("Best Hyperparameters:", best_params)

hn1 = int(best_params['hn1'])
hn2 = int(best_params['hn2'])
dr = float(best_params['dr'])
lr = float(best_params['lr'])
batch_size = int(best_params['batch_size'])

# Re-create DataLoaders with best batch_size
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1).to(device)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class FinalBinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, hn1, hn2, dr):
        super(FinalBinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hn1)
        self.fc2 = nn.Linear(hn1, hn2)
        self.fc3 = nn.Linear(hn2, 1)
        self.dropout = nn.Dropout(dr)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = FinalBinaryClassificationModel(input_dim, hn1, hn2, dr).to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the final model with more epochs for better performance
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_X.size(0)
    epoch_train_loss /= len(train_loader.dataset)

    # Optional: Monitor validation performance
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds)
    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val ROC AUC: {roc_auc:.4f}')

# Final evaluation on validation set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch_y.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Convert probabilities to binary predictions
binary_preds = (all_preds >= 0.5).astype(int)

# Calculate final metrics
accuracy = accuracy_score(all_labels, binary_preds)
roc_auc = roc_auc_score(all_labels, all_preds)
auprc = average_precision_score(all_labels, all_preds)
f1 = f1_score(all_labels, binary_preds)

print("\nFinal Validation Metrics with Optimized Hyperparameters:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"F1 Score: {f1:.4f}")
