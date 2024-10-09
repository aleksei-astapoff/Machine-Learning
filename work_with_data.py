import pandas as pd

from sklearn.tree import DecisionTreeRegressor


data_path = './data/melb_data.csv'
# Загрузка данных в Pandas DataFrame с помощью read_csv()
melbourne_data = pd.read_csv(data_path)

# Вывод названий столбцов
melbourne_data.columns

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

# Определение модели: Дерево решений для задачи регрессии.
melbourne_model = DecisionTreeRegressor(random_state=1)

# Обучение модели
melbourne_model.fit(X, y)

# Предсказание цены на квартиру в Мельбурне вывод данных
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
