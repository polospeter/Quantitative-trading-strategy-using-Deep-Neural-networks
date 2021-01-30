
##### Technical indicators #####
import sys
import pandas as pd

# Most of the following technical analysis indicators require two inputs, df: a timeseries and x: size of window/time interval,
# which for the metric will be calculated, e.g a 14 day moving average of the Apple stock

"""
    -----------------------------------------------------------------------------
    Daily returns
    -----------------------------------------------------------------------------
"""
def dailyreturn(df):
    df['returns']=df['Close']/df['Close'].shift(1,fill_value=df['Close'][1])-1
    return df

"""
    -----------------------------------------------------------------------------
    Moving average
    -----------------------------------------------------------------------------
"""
def ma(df,x):
    df['ma_'+str(x)] = df['Close'].ewm(span=x,min_periods=0,adjust=True,ignore_na=False).mean() # exponential weighted functions
    df['ma_ratio_'+str(x)] = df['ma_'+str(x)] /  df['Close']
    df['ma_inc_'+str(x)] = df['ma_'+str(x)] / df['ma_'+str(x)].shift(1) -1
    return df

"""
    -----------------------------------------------------------------------------
    Bias
    -----------------------------------------------------------------------------
"""
def bias(df,x):
    df['bias_'+str(x)]=df['Close'].rolling(window=x, min_periods=0).mean()-df['Close']
    return df

"""
    -----------------------------------------------------------------------------
    Macd - Moving average Convergence and divergence
    -----------------------------------------------------------------------------
"""
def macd(df,x,y):
    df['macd'] = (df['Close'].ewm(span=x,min_periods=0,adjust=True,ignore_na=False).mean()) - (df['Close'].ewm(span=y,min_periods=0,adjust=True,ignore_na=False).mean())
    return df

"""
    -----------------------------------------------------------------------------
    Bollinger bands
    -----------------------------------------------------------------------------
"""
def bollinger(df,x):
    df['bb_up_'+str(x)] = df['Close'].rolling(window=x, min_periods=0).mean() + (df['Close'].rolling(window=x, min_periods=0).std() * 2)
    df['bb_down_'+str(x)] = df['Close'].rolling(window=x, min_periods=0).mean() - (df['Close'].rolling(window=x, min_periods=0).std() * 2)

    df['bb_up_'+str(x)] = df['bb_up_'+str(x)] / df['Close'] -1
    df['bb_down_'+str(x)] = df['bb_down_'+str(x)] / df['Close'] -1
    return df

"""
    -----------------------------------------------------------------------------
    Stochastic line
    -----------------------------------------------------------------------------
"""
def stochasticline(df,x):
    df['stochastic_k_'+str(x)]=100*(df['Close']-df['Low'].rolling(window=x, min_periods=0).min())/(df['High'].rolling(window=x, min_periods=0).max()-df['Low'].rolling(window=x, min_periods=0).min())
    df['stochastic_d_'+str(x)]=df['stochastic_k_'+str(x)].rolling(window=3, min_periods=0).mean()
    return df

#------------------------------------------------------------------------------------------------------
# Williams %R
def williamR(df,x):
    df['William_R%_'+str(x)]=100*(df['High'].rolling(window=x, min_periods=0).max()-df['Close'])/(df['High'].rolling(window=x, min_periods=0).max()-df['Low'].rolling(window=x, min_periods=0).min())
    return df

#-------------------------------------------------------------------------------------------------------
# Average True Range- ATR
def atr(df,x):
    data1=df['High']-df['Low']
    data2=abs(df['Low']-df['Close'].shift(1,fill_value=df['Close'][1]))
    data3=abs(df['High']-df['Close'].shift(1,fill_value=df['Close'][1]))

    bigdata = pd.concat([data1,data2,data3], ignore_index=False,axis=1)
    bigdatamax=bigdata.max(axis=1)
    df['ATR_'+str(x)]=bigdatamax.rolling(window=x, min_periods=0).mean()
    return df

#-------------------------------------------------------------------------------------------------------
# Commodity Channel Index
def cci(df,x):
    tp=(df['Close']+df['High']+df['Low'])/3
    df['CCI_'+str(x)]=(tp-tp.rolling(window=x, min_periods=0).mean())/(0.015*tp.rolling(x, min_periods=0).std())
    return df

#--------------------------------------------------------------------------------------------------------
# RSI- Relative Strength Index
def RSI(series,xx):
    delta = series.diff().dropna() # daily price change
    period = xx
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0] # up moves/gains
    d[delta < 0] = -delta[delta < 0] # down moves/losses

    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])

    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])

    # Calculate the EWMA
    roll_up1 = u.rolling(period).mean()
    roll_down1 = d.rolling(period).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    return RSI1

def rsi(df,x):
    df['RSI_'+str(x)] = df[['Close']].apply(RSI,xx=x)
    return df

#--------------------------------------------------------------------------------------------------------
# Aroon Oscillator
def aroonosc(df,x):
    arohigh=100
    arolow=0
    df['Aroon Oscillator_'+str(x)] =arohigh-arolow
    return df
--------------------------------------------------------------------------------------------------------

# Formula for the Dice Loss function: --------------------------------------------------------------------

import keras.backend as K

def dice_coef(y_true, y_pred, smooth, thresh):
    y_pred = y_pred > thresh
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(smooth, thresh):
  def dice(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred, smooth, thresh)
  return dice

#-------------------------------------------------------------------------------------------------
#########################################################################################################

from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

##########################################################################################################

def dice_coefic(y_true, y_pred):
    smooth=1e-5
    y_true_f = (y_true)
    y_pred_f = (y_pred)
    intersection = K.sum(y_true_f * y_pred_f) # I was missing so far the squares for the formula??!!!!
    return (2. * intersection + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)

def dice_coef_losss(y_true, y_pred):
    return 1-dice_coefic(y_true, y_pred)
