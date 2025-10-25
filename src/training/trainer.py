import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import os
import sys
from datetime import datetime

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Trainer:
    def __init__(self, model, device, experiment_name="food_classification"):
        self.model = model
        self.device = device
        self.experiment_name = experiment_name
        
        mlflow.set_experiment(experiment_name)
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())
        
        avg_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy())
        
        avg_loss = running_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001, 
              optimizer_name='adam', scheduler_name='step', save_path=None):
        
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Оптимизатор {optimizer_name} не поддерживается")
        
        if scheduler_name.lower() == 'step':
            step_size = max(1, epochs//3) if epochs > 1 else epochs
            scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
        elif scheduler_name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None
        
        criterion = nn.CrossEntropyLoss()
        
        model_params = {
            'model_name': self.model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'epochs': epochs,
            'learning_rate': lr,
            'optimizer': optimizer_name,
            'scheduler': scheduler_name,
            'batch_size': train_loader.batch_size,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset)
        }
        
        best_val_accuracy = 0.0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        with mlflow.start_run():
            mlflow.log_params(model_params)
            
            for epoch in range(epochs):
                print(f"\nЭпоха {epoch+1}/{epochs}")
                
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
                val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = self.validate(val_loader, criterion)
                
                if scheduler:
                    scheduler.step()
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'epoch': epoch + 1
                })
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
                
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    if save_path:
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'val_accuracy': val_acc,
                            'model_params': model_params
                        }, save_path)
                        mlflow.log_artifact(save_path)
            
            final_val_loss, final_val_acc, final_val_precision, final_val_recall, final_val_f1, final_val_preds, final_val_lbls = self.validate(val_loader, criterion)
            
            final_results = {
                'best_val_accuracy': best_val_accuracy,
                'final_train_accuracy': train_accuracies[-1],
                'final_val_accuracy': val_accuracies[-1],
                'final_val_precision': final_val_precision,
                'final_val_recall': final_val_recall,
                'final_val_f1': final_val_f1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'confusion_matrix': confusion_matrix(val_labels, val_preds).tolist()
            }
            
            mlflow.pytorch.log_model(self.model, "model")
            
            return final_results


def train_model(model_name, train_loader, val_loader, device, 
                epochs=10, lr=0.001, optimizer='adam', scheduler='step'):
    from src.models.models import get_model
    
    model = get_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)
    
    trainer = Trainer(model, device, experiment_name=f"food_classification_{model_name}")
    
    save_path = f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    os.makedirs("models", exist_ok=True)
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        optimizer_name=optimizer,
        scheduler_name=scheduler,
        save_path=save_path
    )
    
    return results
