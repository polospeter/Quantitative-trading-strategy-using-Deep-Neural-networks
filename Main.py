

import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential # neural network
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
import datetime
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import LeakyReLU
import urllib.request, json
import os
import pandas_datareader as pdr
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import time
import datetime #solution for datetime error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import CuDNNLSTM

##########################################################################################################################

#-------------------------------------------- Importing dataset ----------------------------------------------------------------------------------
#datetime(2013, 1, 1)
#datetime(2019, 1, 1)

start_date = '2010-01-01'
end_date = '2018-12-31'

# Period 2:
start_date =datetime.datetime(2012, 11, 1)

end_date =datetime.datetime(2018, 12, 31)

tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','XOM','GS','HD','IBM','INTC','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NKE','PFE','PG','TRV','UNH','UTX','VZ','WMT','WBA','DIS'] # all these stocks have data from 2001

# ETFs:
tickers=['SPY','QQQ','XLU','XLE','XLP','XLY','EWZ','EWH','XLF']
#===========================================================================================
dailyreturn(stock)
#______________________________________________________________________________________________________
# Step 1--Import stocks -------------------------------------------------------------------------------
stock=pdr.get_data_yahoo(symbols=tickers[0], start=start_date, end=end_date)

asset=pdr.get_data_yahoo(symbols=tickers[0], start=start_date, end=end_date)

#______________________________________________________________________________________________________
# Step 2--Labeling ------------------------------------------------------------------------------------
mylabels=labelmethod(stock,25)

# plot the labels:
plotlabels(stock[0:400],mylabels[0:400])
plt.savefig('foo.png') # save it as an image

#-----------------------
# (Optional) Convert to continous labels:
tradingsignal2=transformtocontinuos(mylabels)
plt.plot(tradingsignal2[0:100])

# Sell labels: -------------------------------------------------------------------------------------
sell_labels=onlysell(mylabels) #convert labels

# Buy labels: -------------------------------------------------------------------------------------
buy_labels=onlybuy(mylabels)

# Continuos labeling: ---------------- 
plt.plot(tradingsignal2)

#________________________________________________________________________________________________________
# Step 3--Model preparation -----------------------------------------------------------------------------

# Overall:
x_trainn,y_trainn,x_testt,y_testt=modelprep(stock,mylabels,start_date,end_date,0.8)

# Sell signals:
x_train1,y_train1,x_test1,y_test1=modelprep(stock,sell_labels,start_date,end_date,0.8)

# Buy signals:
x_train2,y_train2,x_test2,y_test2=modelprep(stock,buy_labels,start_date,end_date,0.8,"Buy")

# Continous labels: -----------------
x_train3,y_train3,x_test3,y_test3=modelprep(stock,tradingsignal2,start_date,end_date,0.8)

#____________________________________________________________________________________________________
# Step 4--Class weight for imbalanced classes -------------------------------------------------------

# Define the class_weights: 
from sklearn.utils import class_weight

y_integers = np.argmax(np.array(y_train), axis=1)
weights = class_weight.compute_class_weight('balanced',np.unique(y_integers),y_integers)

#... Run the Model::--------------------------------------------------------------------------------

# Create the Neural network for predicting the Sell signals
from keras import backend as K
        
         # Define Sell Model:
        sellmodel = Sequential()
        sellmodel.add(Dense(128, input_shape=x_train_sell["stock0-period-0"].shape[1:]))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(128))
        sellmodel.add(LeakyReLU(alpha=0.1))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(128))
        sellmodel.add(LeakyReLU(alpha=0.1))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(128))
        sellmodel.add(LeakyReLU(alpha=0.1))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(128))
        sellmodel.add(LeakyReLU(alpha=0.1))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(1, activation='sigmoid')) # 1 class
        
        opt= keras.optimizers.adam(lr=1e-4,beta_2=0.999) #
        sellmodel.compile(loss=dice_coef_losss, optimizer=opt, metrics=['accuracy'])
            
        #=============================================================================================================================
        #Sell:
        checkpoint = ModelCheckpoint(filepath='singlestock_sell_bestmodel.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
              
        #=======================================================================================================================
        # Sell model:   
        history = sellmodel.fit(x_train_sell["stock{}-period-{}".format(x, k)],y_train_sell["stock{}-period-{}".format(x, k)],epochs=200,batch_size=100,validation_split=0.25,callbacks=callbacks_list)

        #========================================================================================================================
        # Load Model (have to pass the custom functions in the load_model)
        sell_model=load_model('singlestock_sell_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

        sellsignals["stock{0}".format(x)] = sell_model.predict(x_test_sell["stock{}-period-{}".format(x,k)]) 


#==========================================================================================================
buysignals={}

# Create the Neural network for predicting the BUY signals
        # Define Buy Model:        
        buymodel = Sequential()
        buymodel.add(Dense(128, input_shape=x_train_buy["stock0-period-0"].shape[1:]))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(128))
        buymodel.add(LeakyReLU(alpha=0.1))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(128))
        buymodel.add(LeakyReLU(alpha=0.1))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(128))
        buymodel.add(LeakyReLU(alpha=0.1))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(128))
        buymodel.add(LeakyReLU(alpha=0.1))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(1, activation='sigmoid')) # 1 classes
        
        opt= keras.optimizers.adam(lr=0.0001,beta_2=0.999) #
        buymodel.compile(loss=dice_coef_losss, optimizer=opt, metrics=['accuracy'])
        
        #buymodel.save_weights('buymodel.h5')

        #----------------------------------------------------------------------------------------------------------------------       
        #Buy:
        checkpoint = ModelCheckpoint(filepath='singlestock_buy_bestmodel.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
        #------------------------------------------------------------------------------------------------------------------------
        # Buy model:
        history = buymodel.fit(x_train_buy["stock{}-period-{}".format(x, k)],y_train_buy["stock{}-period-{}".format(x, k)],epochs=200,batch_size=100,validation_split=0.25,callbacks=callbacks_list)
       
        #-------------------------------------------------------------------------------------------------------------------------
        # Load buy model
        buy_model=load_model('singlestock_buy_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

        buysignals["stock{0}".format(x)]=buy_model.predict(x_test_buy["stock{}-period-{}".format(x,k)]) 
        
        K.clear_session() # clear up memory



#==========================================================================================================================
        
preds=model.predict(x_test1) #predictions

# Load Model (have to pass the custom functions in the load_model)
sell_model=load_model('freshstart_sell_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

# Load buy model
buy_model=load_model('freshstart_buy_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

#-----------------------------------------------------------------------------------------------------------------------
# Evaluation:

# Buy:
buysignals=modeleval(buy_model,stock,x_test2,y_test2,0.5)

buysignals=modelevalshort(buy_model,stock,x_test2,y_test2,0.5)
#----------------------------------------------------------------

# Sell:
sellsignals=modeleval(sell_model,stock,x_test1,y_test1,0.4)

sellsignals=modelevalshort(sell_model,stock,x_test1,y_test1,0.5)


threshold=0.5
    buysignals = buy_model.predict(x_test2) 
    buysignals[buysignals<threshold]=0 
    buysignals[buysignals>=threshold]=1
    buysignals=-buysignals # -1 is the label for buysignal
    preds_buy=pd.DataFrame(buysignals,index=y_test_lstm_buy["stock{}-period-{}".format(x,k)].index,columns=["label"])
    #plotlabels(dd["stock{0}".format(x)],preds_buy)
#______________________________________________________________________________________________________

        filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5" # in this case it would not only save the best version of the Model
        
        #model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) -- check the different methods later on!!
        checkpoint = ModelCheckpoint(filepath='keras_buy_gettingthere.hdf5',monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]



# Step 5--Model evalution ----------------------------------------------------------------------------
finalsignals=modeleval(model,stock,x_test1,y_test1,0.5)

# Sell model:
sellsignals=modeleval(sellmodel,stock,x_test1,y_test1,0.5)

#---------------------------------------------------------------------------------------------------

fig=plt.figure(figsize=(14,7)) 
plt.plot(stock[0:500])

# Clean up labels:
finalsignals=samelabels(finalsignals)

plotlabels(stock,finalsignals)

y_test[["Buy","Wait"]]


#________________________________________________________________________________________________________
# Step 5--Combine the Buy and Sell signal predictions:------------------------------------------------------------------------
comblabel=pd.concat([sellsignals,buysignals])

#-------------
comblabel=svm_y_test_pred_sell.label+svm_y_test_pred_buy.label

comblabel=pd.DataFrame(comblabel,columns=["label"])

finalsignals=samelabels(comblabel) # Remove same labels

# Plot final signals:
plotlabels(stock,finalsignals)


# Step 6--Model Backtesting -----------------------------------------------------------------------------
backtest(finalsignals,stock,initcapital=10000) # we set the initial capital for backtesting to be $10000

#_______________________________________________________________________________________________________
# Check predictions manually:
predictt=model.predict(x_test)
predictt=pd.DataFrame(predictt,index=x_test.index)
#####################################################################################################################
# Define the weights: ------------------------------------------------------------------------------------------------------
               
# Adding class_weights:
from sklearn.utils import class_weight

y_integers = np.argmax(np.array(y_train), axis=1)
weights = class_weight.compute_class_weight('balanced',np.unique(y_integers),y_integers)


# Weights for binary classfication:

weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

weights=[10,10,0.5,0.5]
#######################################################################################################

# Dow30 stocks:

tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','XOM','GS','HD','IBM','INTC','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NKE','PFE','PG','TRV','UNH','UTX','VZ','V','WMT','WBA','DIS'] # all these stocks have data from 2001

start_date =datetime.datetime(2012, 1, 3)

end_date =datetime.datetime(2018, 12, 31)

lookback=7 # for LSTM

time_periods=3

#################################################################################################x
  
d={}
dd={}
mylabels={}
mylabelss={}
buy={}
sell={}
x_train={}
y_train={}
x_test={}
y_test={}
x_train_all={}
y_train_all={}

# Buy:
x_train_buy={}
y_train_buy={}
x_test_buy={}
y_test_buy={}

# Sell:
x_train_sell={}
y_train_sell={}
x_test_sell={}
y_test_sell={}


X_train_lstm_buy={}
y_train_lstm_buy={}
X_test_lstm_buy={}
y_test_lstm_buy={}

X_train_lstm_sell={}
y_train_lstm_sell={}
X_test_lstm_sell={}
y_test_lstm_sell={}


lstm_train_buy_appended = {}
lstm_test_buy_appended = {}
lstm_train_sell_appended = {}
lstm_test_sell_appended = {}
lstm_ytrain_buy_appended={}
lstm_ytrain_sell_appended={}


# This case we import data of stocks from 2001 to 2018, we label it, and add features to it:
for x in range(len(tickers)):
    
    # Import the stock: -----------------------------------------------------------------------------------------------
    dd["stock{0}".format(x)]=pdr.get_data_yahoo(symbols=tickers[x], start=start_date, end=end_date)

    # Label the data: -------------------------------------------------------------------------------------------------
    mylabelss["stock{0}".format(x)]=labelmethod(dd["stock{0}".format(x)],14) # time interval parameter
    
    # Model prep, add features and scale data: ------------------------------------------------------------------------
    d["stock{0}".format(x)],mylabels["stock{0}".format(x)]=modelprepshort(dd["stock{0}".format(x)], mylabelss["stock{0}".format(x)])
  
    # Create binary labels: -------------------------------------------------------------------------------------------
    buy["stock{0}".format(x)]=onlybuy(mylabels["stock{0}".format(x)])
    sell["stock{0}".format(x)]=onlysell(mylabels["stock{0}".format(x)])
    
    # Add Period column to stocks: (later do not forget to remove it!)------------------------------------------------
    d["stock{0}".format(x)]['Period']=0.5
    buy["stock{0}".format(x)]['Period']=0.5
    sell["stock{0}".format(x)]['Period']=0.5
    
# We want evaluation of all stocks for all time periods:
  
    for k in range(time_periods): #Number of time periods:
        h=len(d["stock{0}".format(x)])    
        d["stock{0}".format(x)].Period.iloc[((k)*int(h/time_periods)):((k+1)*int(h/time_periods))]=k
        buy["stock{0}".format(x)].Period.iloc[((k)*int(h/time_periods)):((k+1)*int(h/time_periods))]=k
        sell["stock{0}".format(x)].Period.iloc[((k)*int(h/time_periods)):((k+1)*int(h/time_periods))]=k  
        
        # Split stock data into periods:
        d["stock{}-period-{}".format(x, k)]= d["stock{0}".format(x)][d["stock{0}".format(x)].Period==k]
        d["stock{}-period-{}".format(x, k)]=d["stock{}-period-{}".format(x, k)].drop('Period', axis=1) # Remove period column
        
        buy["stock{}-period-{}".format(x, k)]= buy["stock{0}".format(x)][buy["stock{0}".format(x)].Period==k]
        buy["stock{}-period-{}".format(x, k)]=buy["stock{}-period-{}".format(x, k)].drop('Period', axis=1) # Remove period column
     
        sell["stock{}-period-{}".format(x, k)]= sell["stock{0}".format(x)][sell["stock{0}".format(x)].Period==k]
        sell["stock{}-period-{}".format(x, k)]=sell["stock{}-period-{}".format(x, k)].drop('Period', axis=1) # Remove period column
    
    
for k in range(time_periods): #Number of time periods:
    
    for x in range(len(tickers)):       
        
        # Train and Test Split: ---------------------------------------------------------------------------------------
        x_train_buy["stock{}-period-{}".format(x, k)], x_test_buy["stock{}-period-{}".format(x, k)], y_train_buy["stock{}-period-{}".format(x, k)], y_test_buy["stock{}-period-{}".format(x, k)]= train_test_split(d["stock{}-period-{}".format(x, k)], buy["stock{}-period-{}".format(x, k)], test_size=0.17,shuffle=False)
        x_train_sell["stock{}-period-{}".format(x, k)], x_test_sell["stock{}-period-{}".format(x, k)], y_train_sell["stock{}-period-{}".format(x, k)], y_test_sell["stock{}-period-{}".format(x, k)]= train_test_split(d["stock{}-period-{}".format(x, k)],sell["stock{}-period-{}".format(x, k)], test_size=0.17,shuffle=False)
    
        # This should be an optional section for LSTM ==================================================================
      
        # Prepare Train data for LSTM: --------------------------------------------------------------------------------
        X_train_lstm_buy["stock{}-period-{}".format(x, k)], y_train_lstm_buy["stock{}-period-{}".format(x, k)], indices_lstm=prepdata(x_train_buy["stock{}-period-{}".format(x, k)],y_train_buy["stock{}-period-{}".format(x, k)],lookback) # look back is 7 days
        X_train_lstm_sell["stock{}-period-{}".format(x, k)], y_train_lstm_sell["stock{}-period-{}".format(x, k)], indices_lstm=prepdata(x_train_sell["stock{}-period-{}".format(x, k)],y_train_sell["stock{}-period-{}".format(x, k)],lookback) 
        
        # Prepare Test data for LSTM: ---------------------------------------------------------------------------------
        X_test_lstm_buy["stock{}-period-{}".format(x, k)], y_test_lstm_buy["stock{}-period-{}".format(x, k)], indices_lstm=prepdata(x_test_buy["stock{}-period-{}".format(x, k)],y_test_buy["stock{}-period-{}".format(x, k)],lookback)
        X_test_lstm_sell["stock{}-period-{}".format(x, k)], y_test_lstm_sell["stock{}-period-{}".format(x, k)], indices_lstm=prepdata(x_test_sell["stock{}-period-{}".format(x, k)],y_test_sell["stock{}-period-{}".format(x, k)],lookback)
        
        
#############################################################################################################################       

mlp_datacombined={}
mlp_datacombined1={}
mlp_datacombined2={}
mlp_datacombined3={}
mlp_datacombined4={}
mlp_datacombined5={}    

mlp_datacombined_buy_xtrain={}
mlp_datacombined_buy_ytrain={}
mlp_datacombined_buy_xtest={}
mlp_datacombined_sell_xtrain={}
mlp_datacombined_sell_ytrain={}
mlp_datacombined_sell_xtest={}
                 
for k in range(time_periods): #Number of time periods:

    mlp_col=[]
    mlp_col1=[]
    mlp_col2=[]
    mlp_col3=[]
    mlp_col4=[]
    mlp_col5=[]

    for x in range(len(tickers)): 
        # Buy:
        mlp_col.append(x_train_buy["stock{}-period-{}".format(x, k)])
        mlp_col1.append(y_train_buy["stock{}-period-{}".format(x, k)])
        mlp_col2.append(x_test_buy["stock{}-period-{}".format(x, k)])
        
        mlp_col3.append(x_train_sell["stock{}-period-{}".format(x, k)])
        mlp_col4.append(y_train_sell["stock{}-period-{}".format(x, k)])
        mlp_col5.append(x_test_sell["stock{}-period-{}".format(x, k)])
        
    # Buy:   
    mlp_datacombined["period-{0}".format(k)]=mlp_col
    mlp_datacombined1["period-{0}".format(k)]=mlp_col1
    mlp_datacombined2["period-{0}".format(k)]=mlp_col2
    
    # Sell:
    mlp_datacombined3["period-{0}".format(k)]=mlp_col3
    mlp_datacombined4["period-{0}".format(k)]=mlp_col4
    mlp_datacombined5["period-{0}".format(k)]=mlp_col5
                    
    #--------------------------------------------------------------------------------       
# Buy   
    mlp_datacombined_buy_xtrain["period-{0}".format(k)]=np.concatenate(mlp_datacombined["period-{0}".format(k)])
    mlp_datacombined_buy_ytrain["period-{0}".format(k)]=np.concatenate(mlp_datacombined1["period-{0}".format(k)])
    mlp_datacombined_buy_xtest["period-{0}".format(k)]=np.concatenate(mlp_datacombined2["period-{0}".format(k)])
 
# Sell
    mlp_datacombined_sell_xtrain["period-{0}".format(k)]=np.concatenate(mlp_datacombined3["period-{0}".format(k)])
    mlp_datacombined_sell_ytrain["period-{0}".format(k)]=np.concatenate(mlp_datacombined4["period-{0}".format(k)])
    mlp_datacombined_sell_xtest["period-{0}".format(k)]=np.concatenate(mlp_datacombined5["period-{0}".format(k)])
    #--------------------------------------------------------------------------------
    
###################################################################################################################

k=2

#... Run the Model::--------------------------------------------------------------------------------

       # Building the model-- here comes the model itself
        model = Sequential()
        model.add(Dense(128, input_shape=x_train1.shape[1:]))
          
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        
        # Reasoning why to use sigmoid incase of one class!!!!
        model.add(Dense(1, activation='sigmoid')) # 1 classes
        
        opt= keras.optimizers.adam(lr=0.0001,beta_2=0.999) #
        model.compile(loss=dice_coef_losss, optimizer=opt, metrics=['accuracy'])
        
#----------------------------------------------------------------------------------------------------------------------        
        #Sell:
        checkpoint = ModelCheckpoint(filepath='fresh_combined_sell_bestmodel.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
#----------------------------------------------------------------------------------------------------------------------       
        #Buy:
        checkpoint = ModelCheckpoint(filepath='fresh_combined_buy_bestmodel.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
#=======================================================================================================================
# Sell model:   
        history = model.fit(mlp_datacombined_sell_xtrain["period-{0}".format(k)],mlp_datacombined_sell_ytrain["period-{0}".format(k)],epochs=200,batch_size=150,validation_split=0.25,callbacks=callbacks_list)
#------------------------------------------------------------------------------------------------------------------------
# Buy model:
        history = model.fit(mlp_datacombined_buy_xtrain["period-{0}".format(k)],mlp_datacombined_buy_ytrain["period-{0}".format(k)],epochs=200,batch_size=150,validation_split=0.25,callbacks=callbacks_list)
#==========================================================================================================================
        
# Load Model (have to pass the custom functions in the load_model)
sell_model=load_model('freshstart_sell_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

#------------------------------------------------------------------------------------------------------------------
# Load buy model
buy_model=load_model('freshstart_buy_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

#######################################################################################################################x


finalsignals={}
a=[]
buyandholdreturns=[]
b=[]
mymodelreturns=[]

for k in range(2,3): # use time periods here later
    #for x in range(len(tickers)):  # all stock one by one
    for x in range(3):
        
        # Buy signals:------------------------------------------------------------------------------------------
        buysignals = buy_model.predict(x_test_buy["stock{}-period-{}".format(x,k)]) 
        buysignals[buysignals<thresholdbuy]=0 
        buysignals[buysignals>=thresholdbuy]=1
        buysignals=-buysignals # -1 is the label for buysignal
        preds_buy=pd.DataFrame(buysignals,index=x_test_buy["stock{}-period-{}".format(x,k)].index,columns=["label"])
        #plotlabels(dd["stock{0}".format(x)],preds_buy)
        
        # SELL:  ---------------------------------------
        sellsignals = sell_model.predict(x_test_sell["stock{}-period-{}".format(x,k)]) 
        sellsignals[sellsignals<thresholdsell]=0 
        sellsignals[sellsignals>=thresholdsell]=1
        preds_sell=pd.DataFrame(sellsignals,index=x_test_sell["stock{}-period-{}".format(x,k)].index,columns=["label"])
        #plotlabels(dd["stock{0}".format(x)],preds_sell)
        
        
         # Combine Buy and Sell: ------------------------
        finalsignals=preds_sell+preds_buy
        finalsignals=samelabels(finalsignals)
        
        a,b=backtest(finalsignals,dd["stock{0}".format(x)],initcapital=10000,tradingcost=0.005,name=tickers[x])
        
        buyandholdreturns.append(a)
        mymodelreturns.append(b)

# Collect all the returns in Dataframe:      
results=pd.DataFrame(np.column_stack([buyandholdreturns, mymodelreturns,tickers]), columns=['B&H returns', 'MLP model returns','Name'])
results.set_index('Name', inplace=True)

######################################################################################################################################x
#######################################################################################################################################


#                 Train each Model on only one stock data:

from keras import backend as K

finalsignals={}
a=[]
buyandholdreturns=[]
b=[]
mymodelreturns=[]
sellsignals={}

runtime = []


for k in range(1): # use time periods here later
    start_time = time.time()
    runtime.append(time.time()-start_time) #starting time
    
    #for x in range(len(tickers)):  # all stock one by one
    for x in range(len(tickers)):  # all stock one by one
    #for x in range(10):  # all stock one by one

        print("period ",k,"stock ",x)
        #sellmodel.load_weights('sellmodel.h5')
        
         # Define Model:
        sellmodel = Sequential()
        sellmodel.add(Dense(128, input_shape=x_train_sell["stock0-period-0"].shape[1:]))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(128))
        sellmodel.add(LeakyReLU(alpha=0.1))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(128))
        sellmodel.add(LeakyReLU(alpha=0.1))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(128))
        sellmodel.add(LeakyReLU(alpha=0.1))
        sellmodel.add(Dropout(0.1)) 
        
        sellmodel.add(Dense(128))
        sellmodel.add(LeakyReLU(alpha=0.1))
        sellmodel.add(Dropout(0.1)) 
        
        # Reasoning why to use sigmoid incase of one class!!!!
        sellmodel.add(Dense(1, activation='sigmoid')) # 1 classes
        
        opt= keras.optimizers.adam(lr=1e-4,beta_2=0.999) #
        sellmodel.compile(loss=dice_coef_losss, optimizer=opt, metrics=['accuracy'])
            
        # Save initial weights:
        #sellmodel.save_weights('sellmodel.h5')
        #=============================================================================================================================
        #Sell:
        checkpoint = ModelCheckpoint(filepath='singlestock_sell_bestmodel.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
              
        #=======================================================================================================================
        # Sell model:   
        history = sellmodel.fit(x_train_sell["stock{}-period-{}".format(x, k)],y_train_sell["stock{}-period-{}".format(x, k)],epochs=200,batch_size=100,validation_split=0.25,callbacks=callbacks_list)

        #========================================================================================================================
        # Load Model (have to pass the custom functions in the load_model)
        sell_model=load_model('singlestock_sell_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

        sellsignals["stock{0}".format(x)] = sell_model.predict(x_test_sell["stock{}-period-{}".format(x,k)]) 

        if x==1:    
            runtime.append(time.time()-start_time)
       
        if x==9:    
            runtime.append(time.time()-start_time)
        
        if x==19:    
            runtime.append(time.time()-start_time)
        
        K.get_session().close()
        K.set_session(tf.Session())
        K.get_session().run(tf.global_variables_initializer())
        
#==========================================================================================================
from keras import backend as K
buysignals={}

for k in range(1): # use time periods here later
        
    for x in range(len(tickers)):  # all stock one by one
    #for x in range(10):  # all stock one by one
        
        # Buy Model:
        
        print("period ",k,"stock ",x)
        #buymodel.load_weights('buymodel.h5')
        
        # Define Model:
        buymodel = Sequential()
        buymodel.add(Dense(128, input_shape=x_train_buy["stock0-period-0"].shape[1:]))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(128))
        buymodel.add(LeakyReLU(alpha=0.1))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(128))
        buymodel.add(LeakyReLU(alpha=0.1))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(128))
        buymodel.add(LeakyReLU(alpha=0.1))
        buymodel.add(Dropout(0.1))
        
        buymodel.add(Dense(128))
        buymodel.add(LeakyReLU(alpha=0.1))
        buymodel.add(Dropout(0.1))
        
        # Reasoning why to use sigmoid incase of one class!!!!
        buymodel.add(Dense(1, activation='sigmoid')) # 1 classes
        
        opt= keras.optimizers.adam(lr=0.0001,beta_2=0.999) #
        buymodel.compile(loss=dice_coef_losss, optimizer=opt, metrics=['accuracy'])
        
        #buymodel.save_weights('buymodel.h5')

        #----------------------------------------------------------------------------------------------------------------------       
        #Buy:
        checkpoint = ModelCheckpoint(filepath='singlestock_buy_bestmodel.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
        #------------------------------------------------------------------------------------------------------------------------
        # Buy model:
        history = buymodel.fit(x_train_buy["stock{}-period-{}".format(x, k)],y_train_buy["stock{}-period-{}".format(x, k)],epochs=200,batch_size=100,validation_split=0.25,callbacks=callbacks_list)
       
        #-------------------------------------------------------------------------------------------------------------------------
        # Load buy model
        buy_model=load_model('singlestock_buy_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

        buysignals["stock{0}".format(x)]=buy_model.predict(x_test_buy["stock{}-period-{}".format(x,k)]) 
        
        if x==9:    
            runtime.append(time.time()-start_time)
        
        if x==29:    
            runtime.append(time.time()-start_time)
        
        K.get_session().close()
        K.set_session(tf.Session())
        K.get_session().run(tf.global_variables_initializer())

#################################################################################################################

buy_array2006=np.zeros((len(buysignals["stock2"]),30))
sell_array2006=np.zeros((len(sellsignals["stock2"]),30))

buy_array2012=np.zeros((len(buysignals["stock2"]),30))
sell_array2012=np.zeros((len(sellsignals["stock2"]),30))

buy_array2018=np.zeros((len(buysignals["stock0"]),30))
sell_array2018=np.zeros((len(sellsignals["stock0"]),30))


for k in range(1): # use time periods here later
    for x in range(len(tickers)):  # all stock one by one   
        
        buy_array2018[:,x]=np.ravel(buysignals["stock{0}".format(x)])
        sell_array2018[:,x]=np.ravel(sellsignals["stock{0}".format(x)])
        
        
        buy_array2012[:,x]=np.ravel(buysignals["stock{0}".format(x)])
        sell_array2012[:,x]=np.ravel(sellsignals["stock{0}".format(x)])
        
        
        buy_array2006[:,x]=np.ravel(buysignals["stock{0}".format(x)])
        sell_array2006[:,x]=np.ravel(sellsignals["stock{0}".format(x)])



# Save them into a CSV:
np.savetxt("Sellsignals2006.csv", sell_array2006, delimiter=",")
np.savetxt("Buysignals2006.csv", buy_array2006, delimiter=",")

np.savetxt("Sellsignals2012.csv", sell_array2012, delimiter=",")
np.savetxt("Buysignals2012.csv", buy_array2012, delimiter=",")


np.savetxt("Sellsignals2018new.csv", sell_array2018, delimiter=",")
np.savetxt("Buysignals2018new.csv", buy_array2018, delimiter=",")

 
#===================================================================================================================

# Testing period 2006:
sell_array2006 = np.genfromtxt("Sellsignals2006.csv", delimiter=',')
buy_array2006= np.genfromtxt("Buysignals2006.csv", delimiter=',')

# Testing period 2012:
sell_array2012 = np.genfromtxt("Sellsignals2012.csv", delimiter=',') 
buy_array2012= np.genfromtxt("Buysignals2012.csv", delimiter=',')

# Testing period 2018:
sell_array2018 = np.genfromtxt("Sellsignals2018.csv", delimiter=',') 
buy_array2018= np.genfromtxt("Buysignals2018.csv", delimiter=',')
#=====================================================================================================================

     
a=[]
buyandholdreturns=[]
b=[]
mymodelreturns=[]  
finalsignals={}
     
for k in range(1): # use time periods here later
    for x in range(6,7):  # all stock one by one           
        
        # Buy signals:------------------------------------------------------------------------------------------
        thresholdbuy=0.5 #probability threshold
        buysig=buy_array2006[:,x]
        buysig[buysig<thresholdbuy]=0 
        buysig[buysig>=thresholdbuy]=-1
        
        # SELL:  ---------------------------------------
        thresholdsell=0.5
        sellsig = sell_array2006[:,x]
        sellsig[sellsig<thresholdsell]=0 
        sellsig[sellsig>=thresholdsell]=1
       
         # Combine Buy and Sell: ------------------------
        combinedlabel=buysig+sellsig
        finalsignal=pd.DataFrame(combinedlabel,index=y_test_buy["stock{}-period-{}".format(x,k)].index,columns=["label"])    
        finalsignals["stock{0}".format(x)]=samelabels(finalsignal)

        plotlabels(dd["stock{0}".format(x)],finalsignals["stock{0}".format(x)])
        
        # Backtest:
        a,b=backtest(finalsignals["stock{0}".format(x)],dd["stock{0}".format(x)],initcapital=10000,tradingcost=0.005,name=tickers[x])
        
        buyandholdreturns.append(a)
        mymodelreturns.append(b)
        
        
print(np.mean(buyandholdreturns))       
print(np.mean(mymodelreturns))

############################################################################################################
from sklearn.metrics import confusion_matrix


predicts=pd.concat(finalsignals.values(), ignore_index=True)

targetbuy=pd.concat(y_test_buy.values(), ignore_index=True)
targetsell=pd.concat(y_test_sell.values(), ignore_index=True)
targets=targetsell-targetbuy


confusion_matrix(targets, predicts)

labels=[-1,0,1]

# Classification report:---------------------------------------------------------------
print(classification_report(targets,predicts,labels=labels))
    
    
#Confusion matrix ---------------------------------------------------------------------
mat = confusion_matrix(ytestma, predlabelss.label)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,xticklabels=labels,yticklabels=labels) 
plt.xlabel('True label')
plt.ylabel('Predicted label')



############################################################################################################

# Backtest for William R and RSI strategy:

williamR(df,14)

finalsignals={}
a=[]
buyandholdreturns=[]
b=[]
mymodelreturns=[]  

for k in range(1): # use time periods here later
    for x in range(len(tickers)):  # all stock one by one  
        wdata=rsi(dd["stock{0}".format(x)],14)
        wdata= wdata.iloc[:,6]    
        wdata[wdata<=30]=-1
        wdata[wdata>=70]=1
        wdata[wdata>1]=0
        wdata=wdata.to_frame()
        wdata.columns=["label"]
        wdata=wdata.tail(251)
        
        finalsignals["stock{0}".format(x)]=samelabels(wdata)
        
        a,b=backtest(finalsignals["stock{0}".format(x)],dd["stock{0}".format(x)],initcapital=10000,tradingcost=0.005,name=tickers[x])
        
        buyandholdreturns.append(a)
        mymodelreturns.append(b)

# Collect all the returns in Dataframe:    
results=pd.DataFrame(np.column_stack([buyandholdreturns, mymodelreturns,tickers]), columns=['B&H returns', 'MLP model returns','Name'])
results.set_index('Name', inplace=True)

#Import to Excel
results.to_excel(r'C:\Users\Peter\Desktop\rsssi2006.xlsx')
        
print(np.mean(buyandholdreturns))       
print(np.mean(mymodelreturns))    
        
