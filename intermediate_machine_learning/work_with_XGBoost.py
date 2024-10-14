import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

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

# Разделиv данные на обучающие и проверочные наборы
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# Инициализация модели XGBoost для задачи регрессии.
# XGBRegressor — это модель градиентного бустинга,
# оптимизированная для задач регрессии.
# Она строит несколько деревьев решений,
# последовательно улучшая ошибки предыдущих деревьев.
my_model_first = XGBRegressor()
my_model_first.fit(X_train, y_train)

predictions_first = my_model_first.predict(X_valid)
print()
print("Mean Absolute Error: " + str(
    mean_absolute_error(predictions_first, y_valid)
    )
)
print()

# Инициализация модели XGBoost для задачи регрессии
# с настройками гиперпараметров.
# n_estimators=1000 — количество деревьев в модели (или итераций бустинга).
# learning_rate=0.05 — шаг обучения;
# меньшее значение делает обучение медленнее, но может улучшить результат.
# n_jobs=4 — количество параллельных потоков,
# которые будут использоваться для ускорения вычислений
# (полезно на многоядерных процессорах).
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

# Обучение модели XGBoost на обучающих данных (X_train, y_train).
# early_stopping_rounds=5 — остановка обучения,
# если результат не улучшится за 5 итераций.
# eval_set — данные для оценки модели на каждом шаге обучения
# (валидационный набор).
# eval_set=[(X_valid, y_valid)] — валидационные данные,
# которые будут использоваться для отслеживания прогресса.
# verbose=False — отключает вывод промежуточных результатов на экран
# (чтобы обучение было без детального отображения).
my_model.fit(
    X_train, y_train,
    # Остановка, если ошибка не улучшается 5 итераций подряд
    early_stopping_rounds=5,
    # Использование валидационной выборки для оценки во время обучения
    eval_set=[(X_valid, y_valid)],
    # Отключение вывода информации о ходе обучения
    verbose=False
)

predictions = my_model.predict(X_valid)
print()
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
print()
