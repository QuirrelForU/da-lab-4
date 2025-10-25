# Food vs Non-Food Image Classification

MLOps проект для классификации изображений еды с использованием PyTorch и MLflow.

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Подготовка данных
Создайте следующую структуру:
```
data/raw/
├── food/          # Изображения еды
└── non_food/      # Изображения не еды
```

### 3. Обучение моделей
```bash
# Обучение на исходных данных с аугментацией
python train.py --data_dir data/raw --augment --epochs 10

# Быстрый тест
python train.py --data_dir data/raw --epochs 1
```

### 4. Просмотр результатов в MLflow
```bash
mlflow ui
```
Откройте http://localhost:5000 в браузере

### 5. Тестирование моделей
```bash
# Тестирование модели на датасете
python test_model.py --model_name efficientnet --model_path models/efficientnet_YYYYMMDD_HHMMSS.pth --test_dir data/augmented/food

# Пример
python test_model.py --model_name efficientnet --model_path models/efficientnet_20251026_074018.pth --test_dir data/augmented/food --num_samples 20
```

## Используемые модели

- **ResNet18**
- **EfficientNet-B0**
- **Custom CNN**