# ОПРЕДЕЛЕНИЕ ПОРОД СОБАК
## Структура проекта


```
.
├── commands.py
├── config
│   └── config.yaml
├── data
├── dataclass.py
├── data.dvc
├── Dockerfile
├── dog_mlops
│   ├── dataclass.py
│   ├── export_model.py
│   ├── infer.py
│   ├── __init__.py
│   ├── model.py
│   ├── __pycache__
│   │   ├── dataclass.cpython-310.pyc
│   │   ├── infer.cpython-310.pyc
│   │   ├── train.cpython-310.pyc
│   │   ├── export_model.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   └── model.cpython-310.pyc
│   ├── train.py
│   └── utils.py
├── dog_model.ckpt
├── Dogs_predictions.csv
├── label_encoder.pkl
├── poetry.lock
├── project_structure.txt
├── pyproject.toml
├── README.md
└── setup.cfg
```

## Настройка

1. Клонировать репозиторий: https://github.com/korotkovaliza/MlOps.git
2. Установка зависимостей с помощью poetry: poetry install

3. Запуск обучения, инференса и экспорта модели onnx: python3 commands.py 
## Формулировка задачи
Создание модели классификации изображений для определения породы собаки на фотографии. Это нужно для автоматизации процесса идентификации пород собак на изображениях, что может быть полезно в приютах, ветеринарных клиниках или в различных приложениях для животноводов.

## Данные
Для обучения модели планируется использовать набор фотографий собак с метками пород. Данные будут собираться из надежных открытых источников, таких как Kaggle (https://www.kaggle.com/c/dog-breed-identification/data)

## Подход к моделированию
Использование предварительно обученной модели ResNet-18 для классификации изображений. 
1. Вход: изображение собаки
2. Предобработка: нормализация, аугментация
3. Модель: ResNet-18 (предварительно обученная на ImageNet)
4. Обучение: оптимизатор AdamW, функция потерь CrossEntropyLoss
5. Выход: порода собаки (классификационный вывод)

## Способ предсказания
После обучения модели, необходимо обернуть ее в продакшен пайплайн.
Шаги:
1. Загрузка обученной модели.
2. Предобработка новых изображений собак перед передачей модели.
3. Предсказание породы собаки с помощью обученной модели ResNet-18.
4. Отображение результата предсказания или интеграция модели в интерфейс или процесс, который требует классификации пород собак.
Финальное применение: После успешного предсказания породы собаки, модель может быть интегрирована в мобильное приложение для определения пород собак на фотографиях или использована в системе мониторинга видеокамер для автоматической идентификации пород собак в реальном времени.
