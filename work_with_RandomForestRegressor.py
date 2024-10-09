import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


data_path = './data/melb_data.csv'
# Загрузка данных в Pandas DataFrame с помощью read_csv()
melbourne_data = pd.read_csv(data_path)

# Строки с пропусками в столбцах обрабатываются с помощью dropna()
# удаляет строки (по умолчанию)
# axis=0 соответствует строкам, а axis=1 столбцам.
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

# Разделение обучающей выборки на обучающую и валидационную
# random_state=0 для воспроизводимости результатов, гарантирует,
# что мы получаем одинаковое распределение каждый раз,
# когда мы запускаем скрипт
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


# инициализация модели случайного леса и обучение на обучающей выборке
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)


# Рассчитываем среднюю абсолютную ошибку (MAE)
print(mean_absolute_error(val_y, melb_preds))
