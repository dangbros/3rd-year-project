import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# --- 1. Define the Final Model Architecture ---
# This is the "brain" that does the personality prediction
class PersonalityModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PersonalityModel, self).__init__()
        # input_size is 1536 (768 from text + 768 from audio)
        self.layer1 = nn.Linear(input_size, 128)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)

    def forward(self, combined_features):
        x = self.layer1(combined_features)
        x = self.activation(x)
        output = self.output_layer(x)
        return output


def set_seed(seed: int) -> None:
    """Make runs reproducible across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Main execution ---
if __name__ == "__main__":
    # --- Reproducibility ---
    RANDOM_STATE = 42
    set_seed(RANDOM_STATE)

    # --- 2. Load Preprocessed Data ---
    print("Loading preprocessed data...")
    try:
        features = torch.load("train_features.pt").float()
        labels = torch.load("train_labels.pt").float()
        print("Loaded features and labels.")
        print(f"Feature shape: {features.shape}")
        print(f"Label shape: {labels.shape}")
    except FileNotFoundError:
        print("Error: 'train_features.pt' or 'train_labels.pt' not found.")
        print("Please 'git pull' or ask your friend to generate and push them.")
        exit()

    # --- 3. Split Data into Train / Validation / Test ---
    indices = np.arange(features.shape[0])

    # First split: Train vs Temp(Validation + Test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Second split: Temp into Validation and Test (15% each overall)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    print(f"Train size: {len(x_train)}")
    print(f"Validation size: {len(x_val)}")
    print(f"Test size: {len(x_test)}")

    # --- 4. Wrap in TensorDataset + DataLoader ---
    BATCH_SIZE = 32

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. Setup Model, Loss, and Optimizer ---
    INPUT_SIZE = features.shape[1]  # Get size from data (should be 1536)
    OUTPUT_SIZE = labels.shape[1]  # Get size from data (should be 5)

    model = PersonalityModel(INPUT_SIZE, OUTPUT_SIZE)

    # Mean Squared Error is used for regression (predicting a score)
    loss_function = nn.MSELoss()

    # Adam is a standard, effective optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- 6. Training Loop with Early Stopping ---
    print("Starting training...")
    num_epochs = 500
    patience = 20
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_sum = 0.0

        for batch_features, batch_labels in train_loader:
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_features.size(0)

        avg_train_loss = train_loss_sum / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                val_predictions = model(batch_features)
                val_loss = loss_function(val_predictions, batch_labels)
                val_loss_sum += val_loss.item() * batch_features.size(0)

        avg_val_loss = val_loss_sum / len(val_dataset)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # Save best checkpoint and check early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "personality_model_best.pth")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best validation loss: {best_val_loss:.6f}"
            )
            break

    print("Training complete!")

    # --- 7. Load best checkpoint and evaluate on test set ---
    model.load_state_dict(torch.load("personality_model_best.pth"))
    model.eval()

    test_loss_sum = 0.0
    abs_error_sum = 0.0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            test_predictions = model(batch_features)
            batch_mse = loss_function(test_predictions, batch_labels)
            test_loss_sum += batch_mse.item() * batch_features.size(0)

            abs_error_sum += torch.sum(torch.abs(test_predictions - batch_labels)).item()

    test_mse = test_loss_sum / len(test_dataset)
    test_mae = abs_error_sum / (len(test_dataset) * OUTPUT_SIZE)

    print(f"Final Test MSE: {test_mse:.6f}")
    print(f"Final Test MAE: {test_mae:.6f}")
    print("Saved best model to personality_model_best.pth")
