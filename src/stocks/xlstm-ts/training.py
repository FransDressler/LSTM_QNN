import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from .model import xLSTMTS
from .preprocessing import StockDataLoader, train_test_split_temporal
from .metrics import ModelMetrics


class xLSTMTrainer:
    def __init__(self, model: xLSTMTS, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.metrics = ModelMetrics()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def prepare_data(self, symbol: str, batch_size: int = 16, period: str = "2y",
                    interval: str = "1d", test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        data_loader = StockDataLoader(sequence_length=150)
        X, y, _ = data_loader.prepare_dataset(symbol, period, interval)

        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size)

        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += batch_y.size(0)
            correct_predictions += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += batch_y.size(0)
                correct_predictions += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, patience: int = 20) -> Dict:
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        print(f"Training xLSTM-TS model for {epochs} epochs...")
        print(f"Device: {self.device}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss
        }

    def evaluate(self, test_loader: DataLoader) -> Dict:
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        metrics = self.metrics.calculate_all_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_outputs)
        )

        return metrics

    def plot_training_history(self, save_path: str = None):
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])