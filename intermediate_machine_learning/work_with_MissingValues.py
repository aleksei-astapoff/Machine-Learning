import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


data_path = './data/melb_data.csv'
# Загрузка данных в Pandas DataFrame с помощью read_csv()
melbourne_data = pd.read_csv(data_path)


# Целевая переменная - это цена на жилье ('Price'),
# которую мы будем предсказывать с помощью моделей
y = melbourne_data.Price

# Чтобы упростить задачу и работать только с числовыми данными,
# удаляем столбец 'Price', и оставляем только признаки (предикторы),
# которые будут использованы для предсказания
melb_predictors = melbourne_data.drop(['Price'], axis=1)

# Выбираем только числовые признаки для построения модели,
# исключая категориальные данные ('object')
X = melb_predictors.select_dtypes(exclude=['object'])

# Разделение данных на обучающую (80%) и валидационную (20%) выборки
# train_size=0.8 означает, что 80% данных будет использоваться для обучения,
# а 20% для проверки random_state=0 используется для воспроизводимости
# результатов, чтобы всегда получать одинаковые выборки
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
    )


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

    # Инициализация модели случайного леса с 10 деревьями
    # и фиксированным random_state
    # RandomForestRegressor является ансамблем деревьев решений
    # для повышения точности модели
    model = RandomForestRegressor(n_estimators=10, random_state=0)

    # Обучение модели на обучающей выборке
    model.fit(X_train, y_train)

    # Прогнозирование значений для валидационной выборки
    preds = model.predict(X_valid)

    # Возвращение средней абсолютной ошибки (MAE)
    # между фактическими и предсказанными значениями
    return mean_absolute_error(y_valid, preds)


# Подход первый: Удаление столбцов с пропущенными значениями
# Drop Columns with Missing Values

# Генертруем cписок столбцов с пропущенными значениями
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Удаление столбцов с пропущенными значениями,
# из обучающей и валидационной выборок
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print()
print('Подход первый, удаление столбцов с пропущенными значениями')
print()
print('MAE from Approach 1 (Drop columns with missing values):')
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))


# Подход второй: Заполнение пропущенных значений
# Импутация (Imputation) — это метод,
# используемый для заполнения пропущенных данных в наборе данных
# заполняет пропущенные значения,
# по умолчанию средним значением для каждого столбца

# Инициализация SimpleImputer
# По умолчанию SimpleImputer заполняет пропущенные значения
# средним (mean) значением для каждого столбца
my_imputer = SimpleImputer()

# Заполнение пропущенных значений для обучающей выборки:
# fit_transform() обучает Imputer на данных и сразу же применяет заполнение
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

# Заполнение пропущенных значений для валидационной выборки:
# transform() просто применяет уже обученный Imputer
# для заполнения пропущенных значений
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# При заполнении пропущенных значений методом SimpleImputer
# исходные имена столбцов теряются.
# Мы восстанавливаем имена столбцов,
# чтобы новые DataFrame имели такие же заголовки, как оригинальные данные
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print()
print('Подход второй, заполнение пропущенных значений')
print('MAE from Approach 2 (Imputation):')
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# Подход третий: Дополнение к вменению (импутации)
# В этом подходе мы не только заполняем пропущенные значения,
# но и создаем новые признаки, которые указывают,
# было ли исходное значение пропущено.
# Это помогает модели лучше различать случаи,
# где были пропуски, что может повысить точность.

# Копируем исходные данные для обучающей и валидационной выборок,
# чтобы работать с ними и не изменять оригинальные данные.
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Проход по всем столбцам с пропущенными значениями
for col in cols_with_missing:
    # Создаем новый столбец с суффиксом '_was_missing', который указывает,
    # было ли значение в этом столбце пропущено (True)
    # или нет (False) для обучающей выборки
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

    # Аналогично, создаем столбец для валидационной выборки
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Инициализация SimpleImputer для заполнения пропущенных значений
my_imputer = SimpleImputer()

# Заполняем пропущенные значения в обучающей выборке
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

# Заполняем пропущенные значения в валидационной выборке
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Восстанавливаем имена столбцов после вменения,
# так как они теряются после использования SimpleImputer
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print()
print('Подход третий: Дополнение к вменению (импутации)')
print('MAE from Approach 3 (An Extension to Imputation):')
print(
    score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)
    )
