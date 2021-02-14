
# ライブラリのインポート
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import configparser
import datetime
from datetime import datetime, timedelta

df_train=pd.read_csv('/機械学習データサンプルN225ミニ_2015_60m_train.csv',encoding="cp932")
df_test=pd.read_csv('/機械学習データサンプルN225ミニ_2015_60m_test.csv',encoding="cp932")
#csvファイルを読み込む

del df_train['時間']
del df_train['日付']
train=df_train

del df_test['日付']
del df_test['時間']
test=df_test

# windowを設定
window_len = 24

# LSTMへの入力用に処理（訓練）
train_lstm_in = []
for i in range(len(train) - window_len):
    temp = train[i:(i + window_len)].copy()
    for col in train:
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    train_lstm_in.append(temp)
lstm_train_out = (train['終値'][window_len:].values / train['終値'][:-window_len].values)-1
 
# LSTMへの入力用に処理（テスト）
test_lstm_in = []
for i in range(len(test) - window_len):
    temp = test[i:(i + window_len)].copy()
    for col in test:
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    test_lstm_in.append(temp)
lstm_test_out = (test['終値'][window_len:].values / test['終値'][:-window_len].values)-1

# PandasのデータフレームからNumpy配列へ変換
train_lstm_in = [np.array(train_lstm_input) for train_lstm_input in train_lstm_in]
train_lstm_in = np.array(train_lstm_in)
 
test_lstm_in = [np.array(test_lstm_input) for test_lstm_input in test_lstm_in]
test_lstm_in = np.array(test_lstm_in)

# Kerasの使用するコンポーネントをインポートしましょう
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

# LSTMのモデルを設定
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
 
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
 
    model.compile(loss=loss, optimizer=optimizer)
    return model

# ランダムシードの設定
np.random.seed(202)
 
# 初期モデルの構築
yen_model = build_model(train_lstm_in, output_size=1, neurons = 20)
 
# データを流してフィッティングさせましょう
yen_history = yen_model.fit(train_lstm_in, lstm_train_out, 
                            epochs=50, batch_size=1, verbose=2, shuffle=True)

# MAEをプロットしてみよう
fig, ax1 = plt.subplots(1,1)
 
ax1.plot(yen_history.epoch, yen_history.history['loss'])
ax1.set_title('TrainingError')
 
if yen_model.loss == 'mae':
    ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
else:
    ax1.set_ylabel('Model Loss',fontsize=12)
ax1.set_xlabel('# Epochs',fontsize=12)
plt.show()

# 訓練データから予測をして正解レートと予測したレートをプロット
fig, ax1 = plt.subplots(1,1)
ax1.plot(train['終値'][window_len:], label='Actual', color='blue')
ax1.plot(((np.transpose(yen_model.predict(train_lstm_in))+1) * train['終値'].values[:-window_len])[0], 
         label='Predicted', color='red')

# テストデータを使って予測＆プロット
fig, ax1 = plt.subplots(1,1)
ax1.plot(test['終値'][window_len:], label='Actual', color='blue')
ax1.plot(((np.transpose(yen_model.predict(test_lstm_in))+1) * test['終値'].values[:-window_len])[0], 
         label='Predicted', color='red')
ax1.grid(True)