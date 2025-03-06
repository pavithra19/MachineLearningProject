import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Defining feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Defining classification layers
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x

def visualize_data(X: np.ndarray, T: np.ndarray) -> None:
    # Displaying sample images and class distribution
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(f"Label: {T[i]}")
    plt.show()

    unique, counts = np.unique(T, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def correct_data(X: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Normalizing pixel values and computing sample weights
    X = X.float() / 255.0
    unique, counts = torch.unique(T, return_counts=True)
    class_weights = 1.0 / counts.float()
    sample_weights = class_weights[T]
    return X, T, sample_weights

def train_and_evaluate(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                       criterion: nn.Module, optimizer: optim.Optimizer, epochs: int) -> list[float]:
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        
        for inputs, labels in train_loader:
            if inputs.dim() == 5:
                inputs = inputs.squeeze(-1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluating model performance on test data
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                if inputs.dim() == 5:
                    inputs = inputs.squeeze(-1)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if labels.dim() == 2:
                    labels = torch.argmax(labels, dim=1)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        logging.info(f'Epoch {epoch + 1}, Test Accuracy: {accuracy:.2f}%')
    
    return test_accuracies

def compute_confusion_matrix(model: nn.Module, test_loader: DataLoader) -> np.ndarray:
    # Computing confusion matrix for model evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs.dim() == 5:
                inputs = inputs.squeeze(-1)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    cm = [[0 for _ in range(10)] for _ in range(10)]
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1
    
    return np.array(cm)

def main(npz_file: str, clean_test_file: str, test_percentage: int) -> None:
    # Loading and preparing data
    data = np.load(npz_file)
    X_train_raw, T_train_raw = torch.from_numpy(data['X']), torch.from_numpy(data['T']).long()
    
    logging.info(f"Training data shape: {X_train_raw.shape}")
    logging.info(f"Training labels shape: {T_train_raw.shape}")
    
    visualize_data(X_train_raw.numpy(), T_train_raw.numpy())
    X_train_corrected, T_train_corrected, sample_weights = correct_data(X_train_raw, T_train_raw)

    clean_data = np.load(clean_test_file)
    X_clean_test_raw = torch.from_numpy(clean_data['arr_0'])
    T_clean_test_raw = torch.from_numpy(clean_data['arr_1']).float()
    
    logging.info(f"Clean test data shape: {X_clean_test_raw.shape}")
    logging.info(f"Clean test labels shape: {T_clean_test_raw.shape}")

    X_train_raw = X_train_raw.float().unsqueeze(1).squeeze(-1)
    X_train_corrected = X_train_corrected.unsqueeze(1).squeeze(-1)
    X_clean_test_raw = X_clean_test_raw.float().unsqueeze(1).squeeze(-1)

    # Training and evaluating models on different datasets
    
    # Scenario 1: Training on uncorrected data, evaluating on clean test data
    logging.info("Training on uncorrected data...")
    
    train_loader_uncorrected = DataLoader(TensorDataset(X_train_raw, T_train_raw), batch_size=64)
    clean_test_loader = DataLoader(TensorDataset(X_clean_test_raw, T_clean_test_raw), batch_size=64)

    model_uncorrected = CNN()
    criterion_uncorrected = nn.CrossEntropyLoss()
    optimizer_uncorrected = optim.Adam(model_uncorrected.parameters(), lr=0.001)

    train_and_evaluate(model_uncorrected, train_loader_uncorrected, clean_test_loader,
                       criterion_uncorrected, optimizer_uncorrected, epochs=10)

    # Scenario 2: Evaluating on uncorrected test data
    logging.info("Evaluating on uncorrected test data...")
    uncorrected_test_loader = DataLoader(TensorDataset(X_train_raw, T_train_raw), batch_size=64)

    cm_uncorrected_test_data = compute_confusion_matrix(model_uncorrected, uncorrected_test_loader)
    logging.info("Confusion Matrix for Uncorrected Test Data:")
    logging.info(f"\n{cm_uncorrected_test_data}")

    # Scenario 3: Training on corrected data, evaluating on clean test data
    logging.info("Training on corrected data...")
    
    train_loader_corrected = DataLoader(TensorDataset(X_train_corrected, T_train_corrected), batch_size=64)

    model_corrected = CNN()
    criterion_corrected = nn.CrossEntropyLoss()
    optimizer_corrected = optim.Adam(model_corrected.parameters(), lr=0.001)

    train_and_evaluate(model_corrected, train_loader_corrected, clean_test_loader,
                       criterion_corrected, optimizer_corrected, epochs=10)

    # Scenario 4: Evaluating on corrected test data
    logging.info("Evaluating on corrected test data...")
    
    corrected_test_loader = DataLoader(TensorDataset(X_train_corrected, T_train_corrected), batch_size=64)

    cm_corrected_test_data = compute_confusion_matrix(model_corrected, corrected_test_loader)
    logging.info("Confusion Matrix for Corrected Test Data:")
    logging.info(f"\n{cm_corrected_test_data}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logging.error("Usage: python cnn_proj2.py <npz_file> <clean_test_file> <test_percentage>")
        sys.exit(1)

    npz_file_path = sys.argv[1]
    clean_test_file_path = sys.argv[2]
    test_percentage_value = int(sys.argv[3])
    main(npz_file_path, clean_test_file_path, test_percentage_value)
