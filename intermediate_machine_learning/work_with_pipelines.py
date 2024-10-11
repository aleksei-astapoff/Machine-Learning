import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


data_path = './data/melb_data.csv'
# Загрузка данных в Pandas DataFrame с помощью read_csv()
data = pd.read_csv(data_path)

# Предсказание цены на квартиру в Мельбурне выбор цели предсказания,
# цена - столбец 'Price'
y = data.Price

# Обучающая выборка
X = data.drop(['Price'], axis=1)

# Разделение данных на обучающую (80%) и валидационную (20%) выборки
# train_size=0.8 означает, что 80% данных будет использоваться для обучения,
# а 20% для проверки random_state=0 используется для воспроизводимости
# результатов, чтобы всегда получать одинаковые выборки
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )

# "Cardinality" означает количество уникальных значений в столбце.
# В этом коде мы выбираем категориальные столбцы
# с относительно низкой кардинальностью
# (количество уникальных значений в столбце менее 10).
# Это удобно для работы с категориальными переменными,
# так как такие переменные можно легко закодировать
# (например, с помощью one-hot encoding).
categorical_cols = [
    # Проходим по всем столбцам в наборе данных X_train_full
    cname for cname in X_train_full.columns
    # Оставляем только те столбцы, где количество уникальных значений меньше 10
    if X_train_full[cname].nunique() < 10
    # Также проверяем, что столбец является категориальным (тип данных object)
    and X_train_full[cname].dtype == "object"
]

# Выбираем числовые столбцы
# Столбцы, тип данных которых 'int64' или 'float64',
# считаются числовыми (integer и floating point числа).
# Этот код создает список имен столбцов, которые содержат числовые данные,
# что удобно для работы с моделями машинного обучения,
# так как модели обычно требуют числовые данные для обучения.
numerical_cols = [
    # Проходим по всем столбцам в наборе данных X_train_full
    cname for cname in X_train_full.columns
    # Оставляем только те столбцы, у которых тип данных int64 или float64
    if X_train_full[cname].dtype in ['int64', 'float64']
]

# Объединяем списки столбцов с низкой кардинальностью и числовых столбцов
my_cols = categorical_cols + numerical_cols

# Копируем данные для обучения с только что выбранными столбцами
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Трансформер для числовых признаков.
# SimpleImputer заменяет пропущенные значения в числовых столбцах.
# strategy='constant' означает,
# что все пропущенные значения будут заменены
# на константное значение (по умолчанию 0).
numerical_transformer = SimpleImputer(strategy='constant')

# Трансформер для категориальных признаков, использующий конвейер (Pipeline).
# Pipeline позволяет последовательно применять
# несколько шагов обработки данных.
categorical_transformer = Pipeline(steps=[
    # Шаг 1: Замена пропущенных значений в категориальных столбцах.
    # SimpleImputer со стратегией 'most_frequent'
    # заменяет пропуски на наиболее часто встречающееся значение в столбце.
    ('imputer', SimpleImputer(strategy='most_frequent')),

    # Шаг 2: Применение One-Hot Encoding.
    # OneHotEncoder преобразует категориальные значения в числовые
    # (добавляет отдельный столбец для каждой уникальной категории).
    # handle_unknown='ignore' указывает,
    # что нужно игнорировать неизвестные категории,
    # которые могут встретиться в новых данных.
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Создаем объект ColumnTransformer для предварительной обработки данных.
# ColumnTransformer позволяет применять различные преобразования
# к разным столбцам данных.
preprocessor = ColumnTransformer(
    transformers=[
        # Применение числового трансформера (SimpleImputer)
        # к числовым столбцам.
        # 'num' — имя трансформера, numerical_transformer — трансформер
        # для обработки числовых столбцов,
        # numerical_cols — список числовых столбцов,
        # к которым будет применяться этот трансформер.
        ('num', numerical_transformer, numerical_cols),

        # Применение категориального трансформера (Pipeline)
        # к категориальным столбцам.
        # 'cat' — имя трансформера, categorical_transformer
        # — трансформер для обработки категориальных столбцов,
        # categorical_cols — список категориальных столбцов,
        # к которым будет применяться этот трансформер.
        ('cat', categorical_transformer, categorical_cols)
    ]
)
# Инициализируем модель случайного леса
model = RandomForestRegressor(n_estimators=100, random_state=0)


# Создаем конвейер (Pipeline) для автоматической обработки
# данных и обучения модели.
# Pipeline объединяет несколько шагов обработки данных и моделирования,
# что делает код чище и проще.
my_pipeline = Pipeline(
    steps=[
        # Шаг 1: Применение предварительной обработки данных.
        # 'preprocessor' — это объект ColumnTransformer,
        # который был создан ранее.
        # Он выполняет все нужные трансформации
        # для числовых и категориальных данных.
        ('preprocessor', preprocessor),

        # Шаг 2: Применение модели.
        # 'model' — это обучаемая модель
        # (например, RandomForestRegressor или любая другая модель).
        # После предобработки данных модель будет
        # обучена на преобразованных данных.
        ('model', model)
    ]
)

# Обучение конвейера на данных
# Метод fit выполняет все шаги конвейера:
# 1. Применяет шаг 'preprocessor' (предварительная обработка данных).
#    Это включает в себя импутацию (заполнение пропусков),
#    применение One-Hot Encoding и любые другие трансформации,
#    которые были определены в объекте preprocessor
#    для числовых и категориальных столбцов.
# 2. После предобработки данных обучает модель (шаг 'model')
#    на преобразованных данных X_train и y_train.
my_pipeline.fit(X_train, y_train)

# Использование обученного конвейера для предсказаний на валидационных данных.
# Метод predict автоматически применяет шаги предобработки данных,
# аналогичные тем, что были выполнены при обучении
# (импутация, One-Hot Encoding и т.д.),
# и затем использует обученную модель для предсказания значений
# на основе предобработанных данных X_valid.
preds = my_pipeline.predict(X_valid)

score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
