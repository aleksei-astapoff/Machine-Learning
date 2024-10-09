import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split


data_path = './data/melb_data.csv'
# Загрузка данных в Pandas DataFrame с помощью read_csv()
melbourne_data = pd.read_csv(data_path)

# Вывод названий столбцов
melbourne_data.columns


melbourne_data = melbourne_data.dropna(axis=0)

# Предсказание цены на квартиру в Мельбурне выбор цели предсказания,
# цена - столбец 'Price'
y = melbourne_data.Price

# Дабавление признаков "Особенностей" столбцы используемые для прогназирования
melbourne_features = [
    'Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'
    ]

# Обучающая выборка со столбцами "Особенностей" Называются "X"
X = melbourne_data[melbourne_features]

# Определение модели: Дерево решений для задачи регрессии.
melbourne_model = DecisionTreeRegressor(random_state=1)

# Обучение модели
melbourne_model.fit(X, y)

# Используем обученную модель для предсказания цен на жилье
predicted_home_prices = melbourne_model.predict(X)

# Рассчитываем среднюю абсолютную ошибку (MAE)
# между фактическими ценами и предсказанными
# MAE измеряет среднюю разницу между предсказанными и фактическими значениями
print('Вывод средней абсолютной ошибки (MAE) на обучающей выборке:')
print(mean_absolute_error(y, predicted_home_prices))

# Разделение обучающей выборки на обучающую и валидационную
# random_state=0 для воспроизводимости результатов, гарантирует,
# что мы получаем одинаковое распределение каждый раз,
# когда мы запускаем скрипт
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# инициализация модели
melbourne_model = DecisionTreeRegressor(random_state=1)

# обучение модели на обучающей выборке
melbourne_model.fit(train_X, train_y)

# предсказание на валидационной выборке
val_predictions = melbourne_model.predict(val_X)

# вывод средней абсолютной ошибки (MAE) на валидационной выборке
print('Вывод средней абсолютной ошибки (MAE) на валидационной выборке:')
print(mean_absolute_error(val_y, val_predictions))


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    Функция для расчета средней абсолютной ошибки (MAE)
    parameters:
    max_leaf_nodes: максимальное количество листьев в дереве решений,
    train_X, val_X, train_y, val_y: обучающая, валидационная выборка
    return: средняя абсолютная ошибка (mae)
    """

    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0
        )
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)


# Начальное значение средней абсолютной ошибки Максимальное значение
min_mae = float('inf')
# Значение количества листьев дерева
best_tree_size = None

# Поиск оптимального количества листьев дерева
# для сравнения с результатами работы с другими количествами листьев
# и вывод средней абсолютной ошибки (MAE)
# Оптимальное количество листьев, при котором средняя абсолютная ошибка
# на валидационной выборке будет минимальна
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(
        "Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (
            max_leaf_nodes, my_mae
            )
        )
    if my_mae < min_mae:
        min_mae = my_mae
        tree_size = max_leaf_nodes

# Инициализация модели с оптимальным количеством листьев
final_model = DecisionTreeRegressor(
    max_leaf_nodes=best_tree_size, random_state=1
    )
# Обучение модели с оптимальным количеством листьев на обучающей выборке
final_model.fit(X, y)
