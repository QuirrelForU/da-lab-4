#!/usr/bin/env python3

import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем и запускаем основной скрипт
from src.training.train_models import main

if __name__ == "__main__":
    main()
