import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#load the data
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv')

# Separate features and target
X = data.drop('win', axis=1)
y = data['win']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

print(X_train.shape)  # (num_samples, num_features)
print(y_train.shape)  # (num_samples, 1)


# Create DataLoader for training data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Instantiate the model
model = LogisticRegressionModel(input_dim=X_train.shape[1])

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define training loop
num_epochs = 1000
train_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

#evaluate the model
model.eval()

with torch.no_grad():
    train_outputs = model(X_train_tensor)
    train_preds = (train_outputs >= 0.5).float()

    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs >= 0.5).float()

train_accuracy = (train_preds == y_train_tensor).sum().item() / y_train_tensor.size(0)
test_accuracy = (test_preds == y_test_tensor).sum().item() / y_test_tensor.size(0)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

## Write your code here

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools

y_test = y_test_tensor.numpy().astype(int)
y_pred_test_labels = test_preds.numpy().astype(int)

cm  = confusion_matrix(y_test, y_pred_test_labels)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(2)
plt.xticks(tick_marks, ['Loss', 'Win'], rotation=45)
plt.yticks(tick_marks, ['Loss', 'Win'])

thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred_test_labels, target_names=['Loss', 'Win']))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_labels)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

## Write your code here
# Save the model

torch.save(model.state_dict(), 'logistic_model.pth')


# Load the model

loaded_model = LogisticRegressionModel(input_dim=X_train.shape[1])
loaded_model.load_state_dict(torch.load('logistic_model.pth'))

# Ensure the loaded model is in evaluation mode
loaded_model.eval()


# Evaluate the loaded model

with torch.no_grad():
    test_outputs_loaded = loaded_model(X_test_tensor)
    test_preds_loaded = (test_outputs_loaded >= 0.5).float()

test_accuracy_loaded = (test_preds_loaded == y_test_tensor).sum().item() / y_test_tensor.size(0)
print(f"Test Accuracy (Loaded Model): {test_accuracy_loaded * 100:.2f}%")

# Learning rates to test
learning_rates = [0.01, 0.05, 0.1]
num_epochs = 100
test_accuracies = {}

# Loop over learning rates
for lr in learning_rates:
    # Reinitialize model and optimizer
    model = LogisticRegressionModel(input_dim=X_train.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_preds = (test_outputs >= 0.5).float()

    test_accuracy = (test_preds == y_test_tensor).sum().item() / y_test_tensor.size(0)
    test_accuracies[lr] = test_accuracy
    print(f"Learning Rate: {lr}, Test Accuracy: {test_accuracy*100:.2f}%")

# Identify best learning rate
best_lr = max(test_accuracies, key=test_accuracies.get)
print(f"\nBest Learning Rate: {best_lr} with Test Accuracy: {test_accuracies[best_lr]*100:.2f}%")

## Write your code here

import pandas as pd
import matplotlib.pyplot as plt

# Extract the weights of the linear layer
weights = model.linear.weight.data.numpy().flatten()

# Create a DataFrame for feature importance
feature_names = X.columns
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': weights
})

# sort by feature importance
feature_importance['abs_importance'] = feature_importance['Importance'].abs()
feature_importance = feature_importance.sort_values(by='abs_importance', ascending=False)

print(feature_importance)

#Plot feature importance 
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()
