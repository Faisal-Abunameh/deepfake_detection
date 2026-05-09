import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# --- Configuration ---
DATASET_PATH = Path("Dataset")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (256, 256)  # Resize all images to 256x256

# Device configuration (use GPU if available)
device = torch.device('cuda')
print(f"Using device: {device}")

# --- 1. Data Preparation ---
# Define image transformations (resizing, converting to tensor, and normalizing)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def get_dataloaders():
    try:
        train_dataset = datasets.ImageFolder(root=DATASET_PATH / "Train", transform=transform)
        val_dataset = datasets.ImageFolder(root=DATASET_PATH / "Validation", transform=transform)
        test_dataset = datasets.ImageFolder(root=DATASET_PATH / "Test", transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Classes: {train_dataset.classes}")
        return train_loader, val_loader, test_loader, train_dataset.classes
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your 'Dataset' folder contains 'Train', 'Validation', and 'Test' subfolders.")
        exit()

# --- 2. CNN Model Definition ---
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        # Use a pre-trained ResNet50
        try:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            self.resnet = models.resnet50(pretrained=True)
            
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.resnet(x)

def predict_image(image_path, model_path="resnet50.pth"):
    from PIL import Image
    import os
    
    if not os.path.exists(model_path):
        print(f"\n[!] Model file '{model_path}' not found.")
        print("[!] Please train the CNN model first by running: python cnn_model.py")
        return None
        
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"\n[!] Error opening image for CNN: {e}")
        return None
        
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model = DeepfakeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Classes: 0 -> Fake, 1 -> Real (usually, based on alphabetical folder names 'Fake', 'Real')
        # To be safe we should check the mapping, but standard is alphabetical.
        # So 0=Fake, 1=Real
        class_idx = predicted.item()
        conf_score = confidence.item() * 100
        
        result_label = "Real" if class_idx == 1 else "Fake"
        
        print(f"\nCNN Classification: {result_label} (Confidence: {conf_score:.2f}%)")
        return result_label, conf_score

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    
    # Initialize model, loss function, and optimizer
    model = DeepfakeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    print("\nStarting Training...")
    log_file_path = Path("training_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write("--- CNN Training Log ---\n")
        
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(train_loader)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # --- Progress Bar ---
            percent = 100 * (i + 1) / num_batches
            bar_len = 30
            filled = int(bar_len * (i + 1) / num_batches)
            bar = '█' * filled + '-' * (bar_len - filled)
            
            # Current stats
            curr_loss = running_loss / (i + 1)
            curr_acc = 100 * correct / total
            
            sys.stdout.write(f"\rEpoch [{epoch+1}/{EPOCHS}] |{bar}| {percent:.1f}% - Loss: {curr_loss:.4f}, Acc: {curr_acc:.2f}% ")
            sys.stdout.flush()
            
        sys.stdout.write("\n") # Newline after progress bar
        
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        log_msg = (f"Epoch [{epoch+1}/{EPOCHS}] Summary - "
                   f"Train Loss: {running_loss/num_batches:.4f}, Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        print(log_msg)
        
        # Save log to file
        with open(log_file_path, "a") as log_file:
            log_file.write(log_msg + "\n")

    # --- 4. Testing Phase ---
    print("\nEvaluating on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Confusion Matrix elements
    # Assuming 0 = Fake, 1 = Real based on alphabetical order
    TP = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
    TN = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
    FP = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
    FN = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))

    test_total = TP + TN + FP + FN
    test_acc = 100 * (TP + TN) / test_total if test_total > 0 else 0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print("TESTING RESULTS")
    print("="*40)
    print(f"Final Test Accuracy: {test_acc:.2f}%\n")
    
    print("--- Confusion Matrix ---")
    print(f"                    Predicted Fake (0) | Predicted Real (1)")
    print(f"Actual Fake (0)   | {TN:<18} | {FP}")
    print(f"Actual Real (1)   | {FN:<18} | {TP}\n")
    
    print("--- Detailed Metrics ---")
    print(f"Precision: {precision:.4f} (When it predicts Real, how often is it correct?)")
    print(f"Recall:    {recall:.4f} (Out of all actual Reals, how many did it find?)")
    print(f"F1-Score:  {f1_score:.4f} (Harmonic mean of Precision and Recall)")
    print("="*40)
    
    # Save test results to log
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n--- Final Test Results ---\n")
        log_file.write(f"Test Accuracy: {test_acc:.2f}%\n")
        log_file.write(f"Confusion Matrix -> TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n")
        log_file.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}\n")

    # Generate and Save Confusion Matrix Image
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        cm = np.array([[TN, FP], [FN, TP]])
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, f'{z}', ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        
        plt.title('Confusion Matrix', pad=20)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Fake (0)', 'Real (1)'])
        ax.set_yticklabels(['Fake (0)', 'Real (1)'])
        ax.xaxis.set_ticks_position('bottom')
        
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300)
        print("Saved confusion matrix image to 'confusion_matrix.png'")
    except ImportError:
        print("[!] matplotlib is not installed. Could not save confusion matrix image.")
        print("[!] Install it using: pip install matplotlib")

    # --- 5. Save the Model ---
    torch.save(model.state_dict(), "resnet50.pth")
    print("\nModel saved to 'resnet50.pth'")
