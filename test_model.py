import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.models import get_model
from src.data.data_loader import get_transforms


def predict_image(model, image_path, device, transform):
    """Предсказание для одного изображения"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence


def test_model_on_dataset(model_name, model_path, test_dir, device, num_samples=10):
    """Тестирует модель на случайных изображениях из датасета"""
    
    model = get_model(model_name, num_classes=2, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    _, val_transform = get_transforms()
    
    if 'non_food' in test_dir:
        expected_class_name = 'non_food'
        expected_label = 0
    else:
        expected_class_name = 'food'
        expected_label = 1
    
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) == 0:
        print(f"В директории {test_dir} нет изображений")
        return
    
    num_samples = min(num_samples, len(images))
    selected_images = np.random.choice(images, num_samples, replace=False)
    
    correct = 0
    total = len(selected_images)
    
    print(f"\nTesting model {model_name}:")
    print(f"Model: {model_path}")
    print(f"Test directory: {test_dir}")
    print(f"Expected class: {expected_class_name}")
    print("-" * 80)
    
    for img_name in selected_images:
        img_path = os.path.join(test_dir, img_name)
        
        try:
            prediction, confidence = predict_image(model, img_path, device, val_transform)
            
            is_correct = prediction == expected_label
            
            if is_correct:
                correct += 1
            
            status = "OK" if is_correct else "FAIL"
            pred_class = "food" if prediction == 1 else "non_food"
            
            print(f"{status} {img_name:<30} Prediction: {pred_class:<10} Confidence: {confidence:.4f}")
            
        except Exception as e:
            print(f"FAIL Error processing {img_name}: {e}")
    
    accuracy = correct / total
    print("-" * 80)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(description='Тестирование обученных моделей')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Имя модели: resnet18, efficientnet, или convnet')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Путь к сохраненной модели (.pth файл)')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Директория с тестовыми изображениями')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Количество изображений для тестирования')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    if not os.path.exists(args.model_path):
        print(f"Ошибка: Модель {args.model_path} не найдена")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"Ошибка: Директория {args.test_dir} не найдена")
        return
    
    test_model_on_dataset(
        model_name=args.model_name,
        model_path=args.model_path,
        test_dir=args.test_dir,
        device=device,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
