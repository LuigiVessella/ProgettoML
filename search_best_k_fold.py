import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F


class SimpleFFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFFN, self).__init__()
        
        # Strati completamente connessi (fully connected)
        self.fc1 = nn.Linear(input_size, hidden_size)  # Strato nascosto
        self.fc2 = nn.Linear(hidden_size, output_size)  # Strato di output

    def forward(self, x):
        # Passaggio attraverso il primo strato e applicazione della funzione di attivazione ReLU
        x = F.relu(self.fc1(x))
        
        # Passaggio attraverso il secondo strato (output)
        x = self.fc2(x)
        
        return x

# Parametri globali
INPUT_SIZE = 28 * 28  # Dimensione ridotta delle immagini MNIST
OUTPUT_SIZE = 10      # 10 classi (MNIST)
EPOCHS = 10           # Numero di epoche per ciascun fold


# Hyperparametri di RProp
eta_plus_candidates = [1.2, 1.3, 1.5]
eta_minus_candidates = [0.5, 0.6, 0.7]
hidden_nodes_candidates = [32, 64, 128]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

# Rimuovere il subset di 10000 campioni e usare l'intero dataset
subset_indices = np.random.choice(len(train_dataset), 10000, replace=False)
subset_dataset = Subset(train_dataset, subset_indices)

# K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
best_hyperparams = None
best_accuracy = 0

# Esempio di utilizzo delle funzioni
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

for eta_plus in eta_plus_candidates:
    for eta_minus in eta_minus_candidates:
        for hidden_nodes in hidden_nodes_candidates:
            accuracies = []
            print(f"Testing hyperparams: η+={eta_plus}, η-={eta_minus}, hidden_nodes={hidden_nodes}")
            
            for train_idx, val_idx in kf.split(subset_dataset):
                # Create DataLoaders for train and validation sets
                batch_size_train = len(train_idx)
                batch_size_val = len(val_idx)
                train_loader = DataLoader(Subset(subset_dataset, train_idx), batch_size_train, shuffle=True)
                val_loader = DataLoader(Subset(subset_dataset, val_idx), batch_size_val, shuffle=False)
                
                # Inizializza il modello, la perdita e l'ottimizzatore
                model = SimpleFFN(INPUT_SIZE, hidden_nodes, OUTPUT_SIZE).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Rprop(model.parameters(),etas=(eta_minus, eta_plus))
                
                # Training
                for epoch in range(EPOCHS):
                    model.train()
                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)
                        images = images.view(-1, INPUT_SIZE)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                # Validation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        images = images.view(-1, INPUT_SIZE)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                accuracy = correct / total
                accuracies.append(accuracy)
            
            # Calcola l'accuratezza media per il fold corrente
            mean_accuracy = np.mean(accuracies)
            print(f"Mean accuracy for η+={eta_plus}, η-={eta_minus}, hidden_nodes={hidden_nodes}: {mean_accuracy:.4f}")
            
            # Aggiorna i migliori iperparametri
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_hyperparams = (eta_plus, eta_minus, hidden_nodes)

print(f"Best Hyperparameters: η+={best_hyperparams[0]}, η-={best_hyperparams[1]}, hidden_nodes={best_hyperparams[2]} with accuracy={best_accuracy:.4f}")
# Scrivere un messaggio in un file
with open('output.txt', 'a') as file:
    file.write(f"Best Hyperparameters: η+={best_hyperparams[0]}, η-={best_hyperparams[1]}, hidden_nodes={best_hyperparams[2]} with accuracy={best_accuracy:.4f}\n")
