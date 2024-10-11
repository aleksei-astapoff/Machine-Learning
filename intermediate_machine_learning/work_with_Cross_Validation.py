import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score


data_path = './data/melb_data.csv'
# Загрузка данных в Pandas DataFrame с помощью read_csv()
data = pd.read_csv(data_path)

# Выбор подмножества предикторов (признаков)
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

# # Создание нового DataFrame X,
# который содержит только выбранные столбцы (признаки)
# из исходного набора данных data.
X = data[cols_to_use]

# Определение целевой переменной
y = data.Price

# Создаем конвейер (Pipeline) для автоматической обработки
my_pipeline = Pipeline(
    steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))
    ]
)

# Умножаем на -1, так как scikit-learn возвращает
# отрицательное значение MAE (Mean Absolute Error).
# По умолчанию функция cross_val_score при использовании метрики
# 'neg_mean_absolute_error' возвращает отрицательные значения,
# чтобы стандартно интерпретировать метрику как максимизируемую
# (то есть лучшее значение - меньшее по модулю отрицательное число).
# Мы умножаем результат на -1, чтобы получить положительные значения для MAE,
# что привычнее для интерпретации.
scores = -1 * cross_val_score(
    my_pipeline, X, y,
    cv=5, # 5-кратная перекрестная проверка (5-fold cross-validation)
    # Используем отрицательную среднюю абсолютную ошибку для оценки модели
    scoring='neg_mean_absolute_error')

# Выводим MAE для каждой из 5 итераций перекрестной проверки
print("MAE scores:\n", scores)
print()

print("Average MAE score (across experiments):")
# Среднее значение MAE для 5 итераций перекрестной проверки
print(scores.mean())
