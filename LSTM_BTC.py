import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

from matplotlib import pyplot as plt

data = pd.read_csv(filepath_or_buffer='F:/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

# print(data.info())
# print(data.head(10))

# plt.plot(data['Weighted_Price'], label='Price')
# plt.ylabel('price')
# plt.legend()
# plt.show()

# print(data.isnull().sum())
data = data.dropna()
# print(data.isnull().sum())

# print((data == 0).astype(int).any())
data['Volume_(BTC)'].replace(0, np.nan, inplace=True)
data['Volume_(BTC)'].fillna(method='ffill', inplace=True)
data['Volume_(Currency)'].replace(0, np.nan, inplace=True)
data['Volume_(Currency)'].fillna(method='ffill', inplace=True)
# print((data == 0).astype(int).any())


# plt.plot(data['Weighted_Price'], label='Price')
# plt.ylabel('price')
# plt.legend()
# plt.show()


dataset = data.drop('Timestamp', axis=1).values
dataset = dataset.astype('float32')

mms = MinMaxScaler(feature_range=(0, 1))
dataset = mms.fit_transform(dataset)

ratio = 0.8
train_size = int(len(dataset)*ratio)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

def create_ds(data):
    window = 1
    label_idx = 6
    x, y = [], []
    for i in range(len(data) - window):
        x.append(data[i:(i+window), :])
        y.append(data[i+window, label_idx])
    return np.array(x), np.array(y)

x_train, y_train = create_ds(train)
x_test, y_test = create_ds(test)

#keras imple
def btc_lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    return model

model = btc_lstm()
# history = model.fit(x_train, y_train, epochs=80, batch_size=64, validation_data=(x_test, y_test), verbose=1, shuffle=False)
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=1, shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()