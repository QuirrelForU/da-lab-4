import torch
import torch.nn as nn
import os
import sys
import argparse
from datetime import datetime

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.data_loader import create_data_loaders, augment_data
from src.models.models import get_model
from src.training.trainer import train_model


def main():
    parser = argparse.ArgumentParser(description='Обучение моделей классификации изображений еды')
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--augmented_data_dir', type=str, default='data/augmented')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--augment_factor', type=int, default=2)
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Используется устройство: {device}")
    
    models_to_train = ['resnet18', 'efficientnet', 'convnet']
    all_results = {}
    
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ НА ИСХОДНЫХ ДАННЫХ")
    print("="*50)
    
    if os.path.exists(args.data_dir):
        train_loader, val_loader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            test_size=0.2,
            random_state=42
        )
        
        print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
        print(f"Размер валидационной выборки: {len(val_loader.dataset)}")
        
        for model_name in models_to_train:
            print(f"\nОбучение модели {model_name}...")
            
            try:
                results = train_model(
                    model_name=model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    optimizer='adam',
                    scheduler='step'
                )
                
                all_results[f"{model_name}_original"] = {
                    'accuracy': results['best_val_accuracy'],
                    'precision': results.get('final_val_precision', 0),
                    'recall': results.get('final_val_recall', 0),
                    'f1_score': results.get('final_val_f1', 0),
                    'dataset': 'original'
                }
                
                print(f"Модель {model_name} обучена. Лучшая точность: {results['best_val_accuracy']:.4f}")
                
            except Exception as e:
                print(f"Ошибка при обучении модели {model_name}: {e}")
                continue
    else:
        print(f"Директория {args.data_dir} не найдена!")
    
    if args.augment:
        print("\n" + "="*50)
        print("АУГМЕНТАЦИЯ ДАННЫХ")
        print("="*50)
        
        if os.path.exists(args.data_dir):
            print(f"Выполняется аугментация данных из {args.data_dir}...")
            augment_data(
                data_dir=args.data_dir,
                output_dir=args.augmented_data_dir,
                augment_factor=args.augment_factor
            )
            print(f"Аугментированные данные сохранены в {args.augmented_data_dir}")
        else:
            print(f"Директория {args.data_dir} не найдена!")
    
    if os.path.exists(args.augmented_data_dir):
        print("\n" + "="*50)
        print("ОБУЧЕНИЕ НА АУГМЕНТИРОВАННЫХ ДАННЫХ")
        print("="*50)
        
        train_loader_aug, val_loader_aug = create_data_loaders(
            data_dir=args.augmented_data_dir,
            batch_size=args.batch_size,
            test_size=0.2,
            random_state=42
        )
        
        print(f"Размер обучающей выборки (аугментированной): {len(train_loader_aug.dataset)}")
        print(f"Размер валидационной выборки (аугментированной): {len(val_loader_aug.dataset)}")
        
        for model_name in models_to_train:
            print(f"\nОбучение модели {model_name} на аугментированных данных...")
            
            try:
                results = train_model(
                    model_name=model_name,
                    train_loader=train_loader_aug,
                    val_loader=val_loader_aug,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    optimizer='adam',
                    scheduler='step'
                )
                
                all_results[f"{model_name}_augmented"] = {
                    'accuracy': results['best_val_accuracy'],
                    'precision': results.get('final_val_precision', 0),
                    'recall': results.get('final_val_recall', 0),
                    'f1_score': results.get('final_val_f1', 0),
                    'dataset': 'augmented'
                }
                
                print(f"Модель {model_name} обучена на аугментированных данных. Лучшая точность: {results['best_val_accuracy']:.4f}")
                
            except Exception as e:
                print(f"Ошибка при обучении модели {model_name} на аугментированных данных: {e}")
                continue
    
    if all_results:
        print("\n" + "="*50)
        print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*50)
        
        print("\nСводка результатов:")
        print("-" * 80)
        print(f"{'Модель':<20} {'Датасет':<12} {'Точность':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        
        for model_key, results in all_results.items():
            model_name, dataset = model_key.rsplit('_', 1)
            print(f"{model_name:<20} {dataset:<12} {results['accuracy']:<10.4f} "
                 f"{results['precision']:<10.4f} {results['recall']:<10.4f} {results['f1_score']:<10.4f}")
        
        print("-" * 80)
        
        best_original = max([(k, v) for k, v in all_results.items() if v['dataset'] == 'original'], 
                          key=lambda x: x[1]['accuracy'], default=None)
        best_augmented = max([(k, v) for k, v in all_results.items() if v['dataset'] == 'augmented'], 
                            key=lambda x: x[1]['accuracy'], default=None)
        
        if best_original:
            print(f"\nЛучшая модель на исходных данных: {best_original[0]} (точность: {best_original[1]['accuracy']:.4f})")
        
        if best_augmented:
            print(f"Лучшая модель на аугментированных данных: {best_augmented[0]} (точность: {best_augmented[1]['accuracy']:.4f})")
        
        print("\nВлияние аугментации:")
        for model_name in models_to_train:
            original_key = f"{model_name}_original"
            augmented_key = f"{model_name}_augmented"
            
            if original_key in all_results and augmented_key in all_results:
                orig_acc = all_results[original_key]['accuracy']
                aug_acc = all_results[augmented_key]['accuracy']
                improvement = aug_acc - orig_acc
                print(f"{model_name}: {orig_acc:.4f} → {aug_acc:.4f} ({improvement:+.4f})")
    
    else:
        print("Не удалось обучить ни одной модели!")


if __name__ == "__main__":
    main()
