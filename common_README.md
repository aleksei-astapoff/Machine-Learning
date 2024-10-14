
# Учебный курс по машинному обучению

Этот проект представляет собой учебный курс по машинному обучению, состоящий из двух разделов:
1. **Intro to Machine Learning** — материалы для начинающих.
2. **Intermediate Machine Learning** — материалы для продвинутого уровня.

## Как запустить проект

1. Склонируйте репозиторий:

   ```
   git clone https://github.com/aleksei-astapoff/intro-to-machine-learning.git
   ```

2. Создайте и активируйте виртуальное окружение:
   
   Команда для установки виртуального окружения на Mac или Linux:
   ```
   python3 -m venv venv

   source venv/bin/activate
   ```

   Команда для установки виртуального окружения на Windows:
   ```
   python -m venv venv

   source venv/Scripts/activate
   ```

3. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```

4. Перейдите в нужную папку (в зависимости от уровня вашего обучения) и запустите файлы:

   Для начального уровня:
   ```
   cd intro_to_machine_learning

   ```

   Для продвинутого уровня:
   ```
   cd intermediate_machine_learning
  
   ```

## Структура проекта

- **intro_to_machine_learning**: содержит файлы и примеры для изучения базовых концепций машинного обучения.
- **intermediate_machine_learning**: содержит углублённые темы и методы, такие как XGBoost, работа с пропусками в данных, кросс-валидация и пайплайны.

## Требования

Проект использует следующие библиотеки:
- `pandas`
- `scikit-learn`
- `XGBoost`
