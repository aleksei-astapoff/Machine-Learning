import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

data_path = './data/melb_data.csv'
# Загрузка данных в Pandas DataFrame с помощью read_csv()
melbourne_data = pd.read_csv(data_path)

# Целевая переменная - это цена на жилье ('Price'),
# которую мы будем предсказывать с помощью моделей
y = melbourne_data.Price

# Чтобы упростить задачу и работать только с числовыми данными,
# удаляем столбец 'Price', и оставляем только признаки (предикторы),
# которые будут использованы для предсказания
X = melbourne_data.drop(['Price'], axis=1)

# Разделение данных на обучающую (80%) и валидационную (20%) выборки
# train_size=0.8 означает, что 80% данных будет использоваться для обучения,
# а 20% для проверки random_state=0 используется для воспроизводимости
# результатов, чтобы всегда получать одинаковые выборки
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
    )

# Удаление столбцов с пропущенными значениями (простейший подход)
cols_with_missing = [
    col for col in X_train_full.columns if X_train_full[col].isnull().any()
    ]
# Если inplace=True, изменения вносятся непосредственно в исходный объект,
# и метод не возвращает новый объект.
# Если inplace=False (по умолчанию), исходный объект не изменяется,
# а создается его модифицированная копия,
# которую вы можете сохранить в новую переменную.
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" означает количество уникальных значений в столбце.
# В этом коде мы выбираем категориальные столбцы
# с относительно низкой кардинальностью
# (количество уникальных значений в столбце менее 10).
# Это удобно для работы с категориальными переменными,
# так как такие переменные можно легко закодировать
# (например, с помощью one-hot encoding).
low_cardinality_cols = [
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

# Оставляем только выбранные столбцы
# (категориальные с низкой кардинальностью и числовые столбцы)
# Объединяем два списка:
# столбцы с низкой кардинальностью и числовые столбцы.
# Затем создаем копии данных, содержащие только
# эти столбцы для обучения (X_train) и проверки (X_valid).

# Объединяем списки столбцов с низкой кардинальностью и числовых столбцов
my_cols = low_cardinality_cols + numerical_cols

# Создаем новые DataFrame для обучающей и валидационной выборок,
# содержащие только выбранные столбцы

# Копируем данные для обучения с только что выбранными столбцами
X_train = X_train_full[my_cols].copy()
# Копируем данные для валидации с только что выбранными столбцами
X_valid = X_valid_full[my_cols].copy()

print()
print(X_train.head())
print()

# Получаем список категориальных переменных (столбцов)
# Проверяем, какие столбцы в X_train имеют тип данных 'object'
# (обычно это текстовые данные).
# Создаем серию булевых значений,
# где True означает, что столбец имеет тип 'object'.
# Далее, выбираем имена этих столбцов и сохраняем их в список object_cols.

# Булевая серия: True для столбцов, которые имеют тип данных 'object'
s = (X_train.dtypes == 'object')

# Извлекаем имена столбцов с типом 'object' и сохраняем их в список
# s — это булевая серия, где для каждого столбца указывается,
# является ли его тип данных 'object'.
# s[s] — фильтрация, оставляющая только те элементы,
# где значение True (то есть, где столбец имеет тип 'object').
# .index — возвращает метки этих элементов, то есть имена столбцов.
object_cols = list(s[s].index)

# Выводим список категориальных переменных (тех, которые имеют тип 'object')
print("Categorical variables:")
print(object_cols)
print()


# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    """
    Функция для оценки модели на основе обучающей и валидационной выборок.

    Parameters:
    X_train: Обучающие данные (признаки)
    X_valid: Валидационные данные (признаки)
    y_train: Целевая переменная для обучающей выборки (цены на жилье)
    y_valid: Целевая переменная для валидационной выборки (цены на жилье)

    Return:
    Среднюю абсолютную ошибку (MAE) для модели случайного леса,
    обученной на данных X_train и y_train, и проверенной на X_valid и y_valid.
    """

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Оценка из подхода 1 (сбросить категориарные переменные)
# Мы отбрасываем столбцы объектов с помощью метода select_dtypes().
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
print()

# Оценка из подхода 2 (порядковая кодировка)
# Scikit-learn имеет класс OrdinalEncoder,
# который можно использовать для получения порядковых кодировок.
# Мы зацикливаем категориальные переменные
# и применяем порядковый кодировщик отдельно к каждому столбцу.

# Сделайте копию, чтобы избежать изменения исходных данных
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Примените порядковый кодер к каждому столбцу с категориальными данными
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(
    X_train[object_cols]
    )
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
print()


# Оценка из подхода 3 (One-Hot Encoding)
# Мы используем класс OneHotEncoder от scikit-learn,
# чтобы получить одногорячие кодировки. Существует ряд параметров,
# которые можно использовать для настройки его поведения.
# Мы устанавливаем handle_unknown='ignore', чтобы избежать ошибок,
# когда данные проверки содержат классы,
# которые не представлены в обучающих данных, и
# Setting sparse=False гарантирует, что закодированные столбцы возвращаются
# в виде массива numpy (вместо разреженной матрицы).

# Применяем one-hot encoding к каждому категориальному столбцу
# OneHotEncoder преобразует категориальные столбцы в числовые
# путем создания новых столбцов для каждой категории.
# handle_unknown='ignore' используется, чтобы игнорировать
# неизвестные категории в новых данных (например, в валидационных данных),
# sparse=False означает,
# что результат будет возвращен в виде плотной матрицы (не разреженной).
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Чтобы использовать кодировщик,
# мы предоставляем только категориальные столбцы,
# которые мы хотим закодировать в одногорячем порядке.
# Например, для кодирования тренировочных данных
# мы предоставляем X_train[object_cols].
# (object_cols в ячейке кода ниже представляет собой список названий столбцов
# с категориальными данными,
# и поэтому X_train[object_cols] содержит все
# категориальные данные в тренировочном наборе.)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding удалила индекс; верните его обратно
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Удалим категориальные столбцы (заменит на with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Добавим одногорячие закодированные столбцы к числовым характеристикам
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Убедимся что все столбцы имеют тип 'str'
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
print()

print('Обучение модели на лучшей метрике:')
# Объединяем обучающие и валидационные данные в одну выборку
final_X_train = pd.concat([OH_X_train, OH_X_valid], axis=0)
final_y_train = pd.concat([y_train, y_valid], axis=0)

final_model = RandomForestRegressor(n_estimators=100, random_state=0)
final_model.fit(final_X_train, final_y_train)

print('Модель обучена')
