from math import *
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

def f(x):
    return sin(1.2*x + 0.5) + cos(2.5*x + 0.2)

train_size = 1900
x_train = []
y_train_cat = []
x_val_split=[]
y_val_split=[]
x_train_split = []
y_train_split = []
for i in range(train_size):
    xx = i/100
    yy = f(xx) # исследуемая функция
    x_train.append([xx])
    y_train_cat.append([yy])

size_val = 270  # размер выборки валидации
x_val_split = x_train[:size_val]  # выделяем первые наблюдения из обучающей выборки
y_val_split = y_train_cat[:size_val]  # в выборку валидации

x_train_split = x_train[size_val:train_size]  # выделяем последующие наблюдения для обучающей выборки
y_train_split = y_train_cat[size_val:train_size]

model = keras.Sequential()
model.add(Dense(300, input_dim=1, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(1))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['mse', 'mae', 'mape'])
history = model.fit(x_train, y_train_cat, batch_size=150, epochs=30000,validation_data=(x_val_split, y_val_split),verbose=0)
preds1 = model.predict(x_train_split)
preds2 = model.predict(x_val_split)
preds3 = model.predict([1.0001,5.0001,8.0001,10.0001,16.0001,17.521])
plt.figure(1)
plt.plot(x_train,y_train_cat,'b',linewidth='8')
plt.plot(x_train_split,preds1,'y*')
plt.plot(x_val_split,preds2,'r*')
plt.plot([1.0001,5.0001,8.0001,10.0001,16.0001,17.521],preds3,'m*', markersize = 20)
history_dict = history.history
print('\nhistory dict keys:', history_dict.keys())
print('\nhistory dict:', history_dict['mse'])
plt.figure(2)
plt.plot(history_dict['loss'], 'b', label='Training loss')
plt.plot(history_dict['val_loss'], 'r', label='Validation loss')
plt.grid(True)
plt.show()
