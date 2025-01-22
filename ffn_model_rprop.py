
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.optim.rprop
import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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

def plot_training_results(train_losses, val_losses, val_accuracies, batch_size, eta_plus, eta_minus):
    """
    Plot the training and validation losses and validation accuracies over epochs.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Plot training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss', linestyle='--')
    plt.title(f'Loss vs. Epochs; Batch size:{batch_size}; eta_plus:{eta_plus}; eta_minus:{eta_minus}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
#da Pierluigi
def create_dataloaders(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

# Calcolo di media e deviazione standard
def check_normalization(data_loader):
    total_sum = 0
    total_sum_squared = 0
    total_pixels = 0

    for images, _ in data_loader:
        total_sum += images.sum()
        total_sum_squared += (images ** 2).sum()
        total_pixels += images.numel()

    mean = total_sum / total_pixels
    std = torch.sqrt(total_sum_squared / total_pixels - mean ** 2)

    return mean.item(), std.item()


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_losses = []
    val_losses = []
    val_accuracies = []
    n_total_steps = len(train_loader)
    print(f'Total steps: {n_total_steps}')

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        n_train_samples = 0

        # Training phase
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device) 
            images = images.view(-1, 28 * 28)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)
            n_train_samples += labels.size(0)

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}]')

        scheduler.step()
        avg_train_loss = total_train_loss / n_train_samples
        train_losses.append(avg_train_loss)

        # Validation phase
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    return train_losses, val_losses, val_accuracies

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_val_loss = total_val_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_val_loss, accuracy

def check_distribution(train_dataset, val_dataset, test_dataset):
    y_train = train_dataset.dataset.targets[train_dataset.indices].numpy()
    y_val = val_dataset.dataset.targets[val_dataset.indices].numpy()
    y_test = test_dataset.targets.numpy()
        
    datasets = {
        'Train': y_train,
        'Val': y_val,
        'Test': y_test
    }
        
    data = []
    for set_name, y_data in datasets.items():
        unique, counts = np.unique(y_data, return_counts=True)
        for digit, count in zip(unique, counts):
            data.append({
                'Dataset': set_name,
                'Digit': digit,
                'Count': count
            })
        
    df = pd.DataFrame(data)
        
    plt.figure(figsize=(15, 8))
    df_pivot = df.pivot(index='Digit', columns='Dataset', values='Count')
    df_pivot.plot(kind='bar', width=0.8)
        
    plt.title('Digit distribution in MNIST datasets')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.legend(title='Dataset', loc='upper right')
    plt.grid(True, alpha=0.3)    
    plt.show()
    
#da Pierluigi
def data_check(train_dataset, val_dataset, test_dataset):
    print(f'train_dataset data shape: {train_dataset.dataset.data[train_dataset.indices].shape}')
    print(f'train_dataset targets shape: {train_dataset.dataset.targets[train_dataset.indices].shape}')

    print(f'val_dataset data shape: {val_dataset.dataset.data[val_dataset.indices].shape}')
    print(f'val_dataset targets shape: {val_dataset.dataset.targets[val_dataset.indices].shape}')

    print(f'test_dataset data shape: {test_dataset.data.shape}')
    print(f'test_datset targets shape: {test_dataset.targets.shape}')

    print(f'Classes: {train_dataset.dataset.classes}')

   # for i in range(6):
   #     plt.subplot(2, 3, i+1)
   #     plt.xticks([])
   #     plt.yticks([])
   #     plt.imshow(train_dataset.dataset.data[train_dataset.indices][i], cmap='gray')
   #     plt.xlabel(f'Label: {train_dataset.dataset.targets[train_dataset.indices][i]}')
        
        
if __name__ == '__main__':
    
    transform = transforms.Compose([
        #transforms.resize((10, 10)),
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    batch_size = len(train_dataset) 
    batch_size_val = len(val_dataset)
    batch_size_test = len(test_dataset)

    train_loader = create_dataloaders(train_dataset, batch_size, shuffle=True)
    val_loader = create_dataloaders(val_dataset, batch_size_val, shuffle=False)
    test_loader = create_dataloaders(test_dataset, batch_size_test, shuffle=False)
    check_distribution(train_dataset, val_dataset, test_dataset)
    
    print("Data check:")
    data_check(train_dataset, val_dataset, test_dataset)
    
    #mean, std = check_normalization(train_loader)
    #print(f"Mean: {mean:.4f}, Std: {std:.4f}")

    # Esempio di utilizzo delle funzioni
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    learning_rate = 0.001

    best_eta_plus = 1.07 # Parametro eta+ (specificato manualmente)
    best_eta_minus = 0.5  # Parametro eta- (specificato manualmente)
    
    # Parametri
    input_size = 784   # Esempio di input per immagini MNIST (28x28)
    hidden_size = 250 # Numero di neuroni nello strato nascosto
    output_size = 10   # Numero di classi (es. MNIST ha 10 classi)

    # Creazione del modello
    model = SimpleFFN(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Rprop(params=model.parameters(), lr = learning_rate, etas=(best_eta_minus, best_eta_plus))
    optimizer = torch.optim.Rprop(model.parameters(), etas=(best_eta_minus, best_eta_plus))

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    num_epochs = 35

    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs
    )
    
    plot_training_results(train_losses, val_losses, val_accuracies, batch_size, best_eta_plus, best_eta_minus)


    print("Training losses:", train_losses)
    print("Validation losses:", val_losses)
    print("Validation accuracies:", val_accuracies)
    
    
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Ottieni un batch di immagini
    images, labels = next(iter(test_loader))

    # Supponiamo che `model` sia il tuo modello addestrato
    model.eval()  # Imposta il modello in modalità valutazione

    # Preprocessa l'immagine
    image = images.view(-1, 28 * 28).to(device)

    # Ottieni i logit dal modello
    logits = model(image)

    # Applica softmax per ottenere le probabilità
    probabilities = F.softmax(logits, dim=1)
    
    # Visualizza l'immagine e l'istogramma delle probabilità previste fianco a fianco
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Visualizza l'immagine
    ax[0].imshow(images[0].view(28, 28), cmap='gray')
    ax[0].set_title('Immagine MNIST')
    ax[0].axis('off')

    # Visualizza l'istogramma delle probabilità previste
    ax[1].bar(np.arange(10), probabilities.cpu().detach().numpy()[0])
    ax[1].set_xticks(np.arange(10))
    ax[1].set_xlabel('Classe')
    ax[1].set_ylabel('Probabilità')
    ax[1].set_title('Probabilità Predette')

    plt.tight_layout()
    plt.show()