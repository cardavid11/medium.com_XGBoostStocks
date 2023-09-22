import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

# Get historical data
file_path = './data/MARA.csv'
drop_adj_close_col = ['Adj Close']
df = pd.read_csv(file_path, sep=',')
df = df.drop(drop_adj_close_col, axis=1)

df['Date'] = pd.to_datetime(df['Date'])
df = df[(df['Date'].dt.year >= 2015) & (df['Date'].dt.year <= 2020)].copy()
df.index = range(len(df))

df

# OHLC Chart
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Ohlc(x=df.Date,
                      open=df.Open,
                      high=df.High,
                      low=df.Low,
                      close=df.Close,
                      name='Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=df.Volume, name='Volume'), row=2, col=1)
fig.update(layout_xaxis_rangeslider_visible=False)
fig.show()
fig.add_trace(go.Scatter(x=df.Date, y=df.Volume, mode='markers', name='Volume'), row=2, col=1)

# Decomposition
df_close = df.copy()
df_close = df_close.set_index('Date')
df_close = df_close['Close']
df_close.head()
decomp = STL(df_close, period=365).fit()
fig = decomp.plot()
fig.set_size_inches(20, 8)

# Technical indicators
df['EMA_9'] = df['Close'].ewm(9).mean().shift()
df['SMA_5'] = df['Close'].rolling(5).mean().shift()
df['SMA_10'] = df['Close'].rolling(10).mean().shift()
df['SMA_15'] = df['Close'].rolling(15).mean().shift()
df['SMA_30'] = df['Close'].rolling(30).mean().shift()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.Date, y=df.EMA_9, name='EMA 9'))
fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_5, name='SMA 5'))
fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_10, name='SMA 10'))
fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_15, name='SMA 15'))
fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_30, name='SMA 30'))
fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close', opacity=0.2))
fig.show()

# Relative Strength Index
def relative_strength_idx(df, n=14):
    """
    Calculate the n-period Relative Strength Index.
    :param df: DataFrame with 'Close' column
    :param n: Number of periods for RSI calculation
    :return: Series of RSI values
    """
    # Ensure 'Close' exists in df
    if 'Close' not in df:
        raise ValueError("DataFrame missing 'Close' column")

    # Calculate price differences
    delta = df['Close'].diff().dropna()

    # Separate the up and down movements in price
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate rolling means
    roll_up = up.rolling(window=n).mean()
    roll_down = down.abs().rolling(window=n).mean()

    # Calculate RS
    rs = roll_up / roll_down

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


df['RSI'] = relative_strength_idx(df).fillna(0)
fig = go.Figure(go.Scatter(x=df.Date, y=df.RSI, name='RSI'))
fig.show()

# MACD
EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
df['MACD'] = pd.Series(EMA_12 - EMA_26)
df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=df['MACD'], name='MACD'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=df['MACD_signal'], name='Signal line'), row=2, col=1)
fig.show()


# Shift label column
df['Close'] = df['Close'].shift(-1)

# Drop invalid samples
df = df.iloc[33:] # Because of moving averages and MACD line
df = df[:-1]      # Because of shifting close price
df.index = range(len(df))

# split stock data frame into three subsets: training ( 70%), validation ( 15%) and test ( 15%) sets. 
test_size  = 0.15
valid_size = 0.15
test_split_idx  = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))
train_df  = df.loc[:valid_split_idx].copy()
valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
test_df   = df.loc[test_split_idx+1:].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.Close, name='Training'))
fig.add_trace(go.Scatter(x=valid_df.Date, y=valid_df.Close, name='Validation'))
fig.add_trace(go.Scatter(x=test_df.Date,  y=test_df.Close,  name='Test'))
fig.show()

# Drop unnecessary columns
drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'OpenInt']

for col in drop_cols:
    if col in train_df.columns:
        train_df = train_df.drop(col, axis=1)
    if col in valid_df.columns:
        valid_df = valid_df.drop(col, axis=1)
    if col in test_df.columns:
        test_df = test_df.drop(col, axis=1)


# Split into features and labels
y_train = train_df['Close'].copy()
X_train = train_df.drop(['Close'], axis=1)

y_valid = valid_df['Close'].copy()
X_valid = valid_df.drop(['Close'], axis=1)

y_test  = test_df['Close'].copy()
X_test  = test_df.drop(['Close'], axis=1)

X_train.info()

# Fine-tune XGBoostRegressor
parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}

# Another set of parameters
# parameters = {
#     'n_estimators': [100, 300, 500],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'max_depth': [3, 5, 8, 10],
#     'gamma': [0.001, 0.01, 0.1],
#     'min_child_weight': [1, 5, 10],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'random_state': [42]
# }

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
clf = GridSearchCV(model, parameters)

clf.fit(X_train, y_train)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

plot_importance(model)

#Calculate and visualize predictions

y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')
print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')

predicted_prices = df.loc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.Date, y=df.Close,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=predicted_prices.Close,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=y_test,
                         name='Truth',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

fig.show()


