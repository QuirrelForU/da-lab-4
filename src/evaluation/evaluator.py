"""
Модуль для оценки качества моделей
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
import os
from datetime import datetime


class ModelEvaluator:
    """Класс для оценки качества моделей"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate(self, data_loader):
        """
        Полная оценка модели на датасете
        
        Args:
            data_loader: DataLoader с данными для оценки
            
        Returns:
            dict: Словарь с метриками
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Получаем предсказания и вероятности
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Вычисляем метрики
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # ROC AUC для бинарной классификации
        if len(np.unique(all_labels)) == 2:
            roc_auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        else:
            roc_auc = None
        
        # Матрица ошибок
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Отчет о классификации
        report = classification_report(all_labels, all_predictions, 
                                     target_names=['Non-Food', 'Food'], 
                                     output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, class_names=['Non-Food', 'Food'], save_path=None):
        """Построение матрицы ошибок"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, labels, probabilities, save_path=None):
        """Построение ROC кривой"""
        if len(np.unique(labels)) != 2:
            print("ROC кривая доступна только для бинарной классификации")
            return
        
        fpr, tpr, _ = roc_curve(labels, np.array(probabilities)[:, 1])
        auc = roc_auc_score(labels, np.array(probabilities)[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs, save_path=None):
        """Построение графиков истории обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # График точности
        ax2.plot(train_accs, label='Train Accuracy', color='blue')
        ax2.plot(val_accs, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def compare_models(results_dict, save_path=None):
    """
    Сравнение результатов нескольких моделей
    
    Args:
        results_dict: Словарь с результатами моделей
        save_path: Путь для сохранения графика
    """
    model_names = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Подготовка данных
    data = []
    for model_name in model_names:
        for metric in metrics:
            data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': results_dict[model_name][metric]
            })
    
    # Построение графика
    plt.figure(figsize=(12, 8))
    sns.barplot(data=data, x='Model', y='Value', hue='Metric')
    plt.title('Comparison of Model Performance')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_report(results_dict, output_path="reports/model_comparison_report.md"):
    """
    Генерация отчета о сравнении моделей
    
    Args:
        results_dict: Словарь с результатами моделей
        output_path: Путь для сохранения отчета
    """
    os.makedirs("reports", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Отчет о сравнении моделей классификации изображений еды\n\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Сводка результатов\n\n")
        f.write("| Модель | Точность | Точность (Precision) | Полнота (Recall) | F1-Score |\n")
        f.write("|--------|----------|---------------------|------------------|----------|\n")
        
        for model_name, results in results_dict.items():
            f.write(f"| {model_name} | {results['accuracy']:.4f} | "
                   f"{results['precision']:.4f} | {results['recall']:.4f} | "
                   f"{results['f1_score']:.4f} |\n")
        
        f.write("\n## Детальный анализ\n\n")
        
        for model_name, results in results_dict.items():
            f.write(f"### {model_name}\n\n")
            f.write(f"- **Точность**: {results['accuracy']:.4f}\n")
            f.write(f"- **Точность (Precision)**: {results['precision']:.4f}\n")
            f.write(f"- **Полнота (Recall)**: {results['recall']:.4f}\n")
            f.write(f"- **F1-Score**: {results['f1_score']:.4f}\n")
            
            if results.get('roc_auc'):
                f.write(f"- **ROC AUC**: {results['roc_auc']:.4f}\n")
            
            f.write("\n")
        
        # Определение лучшей модели
        best_model = max(results_dict.keys(), 
                        key=lambda x: results_dict[x]['accuracy'])
        f.write(f"## Лучшая модель\n\n")
        f.write(f"По метрике точности лучшей является модель **{best_model}** "
               f"с точностью {results_dict[best_model]['accuracy']:.4f}.\n\n")
        
        f.write("## Рекомендации\n\n")
        f.write("1. Для продакшена рекомендуется использовать модель с наивысшей точностью\n")
        f.write("2. При необходимости баланса между точностью и скоростью можно рассмотреть более легкие модели\n")
        f.write("3. Для улучшения результатов рекомендуется:\n")
        f.write("   - Увеличить количество обучающих данных\n")
        f.write("   - Применить аугментацию данных\n")
        f.write("   - Настроить гиперпараметры\n")
        f.write("   - Использовать ансамбли моделей\n")
