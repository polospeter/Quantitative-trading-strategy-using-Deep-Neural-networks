from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import time
import seaborn as sns
import pandas as pd
import random

#########################################################################################################################
# FINAL VERSION OF THE GROUP BACKTEST FUNCTION
#########################################################################################################################

# It is the version of the Backtesting function that can take multiple stocks as inputs for a given time period,
# and will run the backtest for each one of them.

# DOW30 stocks
tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','XOM','GS','HD','IBM','INTC','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NKE','PFE','PG','TRV','UNH','UTX','VZ','WMT','WBA','DIS'] # all these stocks have data from 2001

start_date =datetime.datetime(2012, 11,23)

end_date =datetime.datetime(2018, 12, 31)

lookback=7 # for LSTM

time_periods=1

#---------------------------------------------------------------------------------------------------------------------
# Here we initialize the necessary train and test container datasets for the Buy and Sell signals.

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

# Buy for LSTM mode:
X_train_lstm_buy={}
y_train_lstm_buy={}
X_test_lstm_buy={}
y_test_lstm_buy={}

# Sell for LSTM mode:
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

#######################################################################################################################
# MODEL PREPARATION
#######################################################################################################################

# Here we import data of stocks for the given time period, we label it, and add features to it:
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
    
    # Add Period column to stocks: ------------------------------------------------------------------------------------
    d["stock{0}".format(x)]['Period']=0
    buy["stock{0}".format(x)]['Period']=0
    sell["stock{0}".format(x)]['Period']=0
    
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
        x_train_buy["stock{}-period-{}".format(x, k)], x_test_buy["stock{}-period-{}".format(x, k)], y_train_buy["stock{}-period-{}".format(x, k)], y_test_buy["stock{}-period-{}".format(x, k)]= train_test_split(d["stock{}-period-{}".format(x, k)], buy["stock{}-period-{}".format(x, k)], test_size=0.167,shuffle=False)
        x_train_sell["stock{}-period-{}".format(x, k)], x_test_sell["stock{}-period-{}".format(x, k)], y_train_sell["stock{}-period-{}".format(x, k)], y_test_sell["stock{}-period-{}".format(x, k)]= train_test_split(d["stock{}-period-{}".format(x, k)],sell["stock{}-period-{}".format(x, k)], test_size=0.167,shuffle=False)
    
        # This is an optional section for LSTM: ======================================================================
        
        # Prepare Train data for LSTM: --------------------------------------------------------------------------------
        X_train_lstm_buy["stock{}-period-{}".format(x, k)], y_train_lstm_buy["stock{}-period-{}".format(x, k)], indices_lstm=prepdata(x_train_buy["stock{}-period-{}".format(x, k)],y_train_buy["stock{}-period-{}".format(x, k)],lookback) # look back is 7 days
        X_train_lstm_sell["stock{}-period-{}".format(x, k)], y_train_lstm_sell["stock{}-period-{}".format(x, k)], indices_lstm=prepdata(x_train_sell["stock{}-period-{}".format(x, k)],y_train_sell["stock{}-period-{}".format(x, k)],lookback) 
        
        # Prepare Test data for LSTM: ---------------------------------------------------------------------------------
        X_test_lstm_buy["stock{}-period-{}".format(x, k)], y_test_lstm_buy["stock{}-period-{}".format(x, k)], indices_lstm=prepdata(x_test_buy["stock{}-period-{}".format(x, k)],y_test_buy["stock{}-period-{}".format(x, k)],lookback)
        X_test_lstm_sell["stock{}-period-{}".format(x, k)], y_test_lstm_sell["stock{}-period-{}".format(x, k)], indices_lstm=prepdata(x_test_sell["stock{}-period-{}".format(x, k)],y_test_sell["stock{}-period-{}".format(x, k)],lookback)
        

#######################################################################################################################
# LSTM MODEL PREPARATION AND TRAINING
#######################################################################################################################

# The LSTM model requires a special input data structure.

# Define the class_weights for Weighted Binary Cross-entropy function: 
from sklearn.utils import class_weight

#y_integers = np.argmax(np.array(y_train_lstm_sell["stock{}-period-{}".format(x, k)]), axis=0)
y_integers =np.array(y_train_lstm_sell["stock{}-period-{}".format(x, k)])
weights = class_weight.compute_class_weight('balanced',np.unique(y_integers),y_integers)

weights =np.ones(2)
weights[0,]=[(len(y_integers)-sum(y_integers))/sum(y_integers),1]

random.seed(1234)

# Train each Model on only one stock data: --------------------------------------------

sell_saved = sell_lstm.get_weights()
buy_saved = buy_lstm.get_weights()


from keras import backend as K

finalsignals={}
buysignals_lstm={}
sellsignals_lstm={}
a=[]
buyandholdreturns=[]
b=[]
mymodelreturns=[]

# Long-Short Term Memory Model:

for k in range(1): # use time periods here later
    
    #for x in range(len(tickers)):  # all stock one by one
    for x in range(len(tickers)):  # all stock one by one
        
        print("period ",k,"stock ",x)
        
        sell_lstm = Sequential()

        sell_lstm.add(CuDNNLSTM(units = 128, return_sequences = True, input_shape = (X_train_lstm_sell["stock0-period-0"].shape[1],X_train_lstm_sell["stock0-period-0"].shape[2])))
        sell_lstm.add(Dropout(0.2))
        
        sell_lstm.add(CuDNNLSTM(units = 128, return_sequences = True))
        sell_lstm.add(Dropout(0.2))

        sell_lstm.add(Dropout(0.2))
                
        sell_lstm.add(CuDNNLSTM(units = 128))
        sell_lstm.add(Dropout(0.2))
                        
        sell_lstm.add(Dense(units = 1, activation='sigmoid'))
        
        opt = keras.optimizers.adam(lr=1e-4,beta_2=0.999)
        sell_lstm.compile(optimizer = opt, loss =dice_coef_losss, metrics=['accuracy'] )
               
        #=============================================================================================================================
        #Sell:
        checkpoint = ModelCheckpoint(filepath='lstmstock_sell_bestmodel.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
              
        #=======================================================================================================================
        # Sell model:   
        history = sell_lstm.fit(X_train_lstm_sell["stock{}-period-{}".format(x, k)],y_train_lstm_sell["stock{}-period-{}".format(x, k)],epochs=200,batch_size=110,validation_split=0.25,callbacks=callbacks_list)

        #========================================================================================================================
        # Load Model (have to pass the custom functions in the load_model)
        sell_lstm_model=load_model('lstmstock_sell_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})

        # SELL:  ---------------------------------------
        sellsignals_lstm["stock{0}".format(x)] = sell_lstm_model.predict(X_test_lstm_sell["stock{}-period-{}".format(x,k)]) 

        K.get_session().close()
        K.set_session(tf.Session())
        K.get_session().run(tf.global_variables_initializer())
        
        
#==========================================================================================================

from keras import backend as K

buysignals_lstm={}

for k in range(1): # use time periods here later
    #for x in range(len(tickers)):  # all stock one by one
    for x in range(len(tickers)):  # all stock one by one
        
        # Buy Model:
        print("period ",k,"stock ",x)
      
        buy_lstm = Sequential()

        buy_lstm.add(CuDNNLSTM(units = 128, return_sequences = True, input_shape = (X_train_lstm_buy["stock0-period-0"].shape[1],X_train_lstm_buy["stock0-period-0"].shape[2])))
        buy_lstm.add(Dropout(0.2))

        buy_lstm.add(CuDNNLSTM(units = 128, return_sequences = True))
        buy_lstm.add(Dropout(0.2))
        
        buy_lstm.add(CuDNNLSTM(units = 128))
        buy_lstm.add(Dropout(0.2))

        buy_lstm.add(Dense(units = 1, activation='sigmoid'))
        
        opt = keras.optimizers.adam(lr=1e-4,beta_2=0.999)
        buy_lstm.compile(optimizer = opt, loss =dice_coef_losss, metrics=['accuracy'] )
       
        #----------------------------------------------------------------------------------------------------------------------       
        #Buy:
        checkpoint = ModelCheckpoint(filepath='lstmstock_buy_bestmodel.hdf5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        
        #------------------------------------------------------------------------------------------------------------------------
        # Buy model:
        history = buy_lstm.fit(X_train_lstm_buy["stock{}-period-{}".format(x, k)],y_train_lstm_buy["stock{}-period-{}".format(x, k)],epochs=200,batch_size=110,validation_split=0.25,callbacks=callbacks_list,class_weight=None)
       
        #-------------------------------------------------------------------------------------------------------------------------
        # Load buy model
        buy_lstm_model=load_model('lstmstock_buy_bestmodel.hdf5',custom_objects={'dice_coef_losss':dice_coef_losss})
        
        # Buy signals:------------------------------------------------------------------------------------------
        buysignals_lstm["stock{0}".format(x)] = buy_lstm_model.predict(X_test_lstm_buy["stock{}-period-{}".format(x,k)]) 
        
        K.get_session().close()
        K.set_session(tf.Session())
        K.get_session().run(tf.global_variables_initializer())
    
   
#####################################################################################################################


lstm_buy_array2006=np.zeros((len(buysignals["stock2"]),30))
lstm_sell_array2006=np.zeros((len(sellsignals["stock2"]),30))

lstm_buy_array2012=np.zeros((len(buysignals["stock2"]),30))
lstm_sell_array2012=np.zeros((len(sellsignals["stock2"]),30))

lstm_buy_array2018=np.zeros((len(buysignals["stock2"]),29))
lstm_sell_array2018=np.zeros((len(sellsignals_lstm["stock2"]),29))


for k in range(1): # use time periods here later
    for x in range(len(tickers)):  # all stock one by one   
        
        lstm_buy_array2018[:,x]=np.ravel(buysignals["stock{0}".format(x)])
        lstm_sell_array2018[:,x]=np.ravel(sellsignals_lstm["stock{0}".format(x)])
        
        
        lstm_buy_array2012[:,x]=np.ravel(buysignals["stock{0}".format(x)])
        lstm_sell_array2012[:,x]=np.ravel(sellsignals["stock{0}".format(x)])
        
        
        lstm_buy_array2006[:,x]=np.ravel(buysignals["stock{0}".format(x)])
        lstm_sell_array2006[:,x]=np.ravel(sellsignals["stock{0}".format(x)])


###########################################################################################################
# SAVE THE RESULTS INTO A CSV
###########################################################################################################

np.savetxt("Sellsignals2006.csv", sell_array2006, delimiter=",")
np.savetxt("Buysignals2006.csv", buy_array2006, delimiter=",")

np.savetxt("Sellsignals2012.csv", sell_array2012, delimiter=",")
np.savetxt("Buysignals2012.csv", buy_array2012, delimiter=",")

np.savetxt("Sellsignals_lstm2018.csv", lstm_sell_array2018, delimiter=",")
np.savetxt("Buysignals_lstm2018.csv", lstm_buy_array2018, delimiter=",")

selldata = pd.read_csv("Sellsignals2012.csv",header ='False') 

import csv

with open("Sellsignals2012.csv", newline='') as csvfile:
    selldata = list(csv.reader(csvfile))
 
    np.genfromtxt(r'C:\Users\Peter\Desktop\Master thesis\table18.csv', delimiter=',')
 C:\Users\Peter\Desktop\University\Thesis\Python code\Sellsignals2012.csv  

 
# Testing period 2012: 
sell_array2012 = np.genfromtxt(r"C:\Users\Peter\Desktop\University\Thesis\Python code\Sellsignals2012.csv", delimiter=',') # Okay this one works
buy_array2012= np.genfromtxt(r"C:\Users\Peter\Desktop\University\Thesis\Python code\Buysignals2012.csv", delimiter=',')

# Testing period 2006:
sell_array2006 = np.genfromtxt(r"C:\Users\Peter\Desktop\University\Thesis\Python code\Sellsignals2006.csv", delimiter=',') # Okay this one works
buy_array2006= np.genfromtxt(r"C:\Users\Peter\Desktop\University\Thesis\Python code\Buysignals2006.csv", delimiter=',')

# Testing period 2018:
sell_array2018 = np.genfromtxt("Sellsignals_lstm2018.csv", delimiter=',') # Okay this one works
buy_array2018= np.genfromtxt("Buysignals_lstm2018.csv", delimiter=',')
#=====================================================================================================================
  
 
a=[]
buyandholdreturns=[]
b=[]
mymodelreturns=[]  
finalsignal=[]
     
for k in range(1): # use time periods here later
    for x in range(len(tickers)):  # all stock one by one           
        
        # Buy signals:------------------------------------------------------------------------------------------
        thresholdbuy=0.5 #probability threshold
        buysig=buy_array2012[:,x]
        buysig[buysig<thresholdbuy]=0 
        buysig[buysig>=thresholdbuy]=-1

        # SELL:  ---------------------------------------
        thresholdsell=0.5
        sellsig = sell_array2012[:,x]
        sellsig[sellsig<thresholdsell]=0 
        sellsig[sellsig>=thresholdsell]=1

         # Combine Buy and Sell: ------------------------
        combinedlabel=buysig+sellsig
        finalsignal=pd.DataFrame(combinedlabel,index=(x_test_buy["stock{}-period-{}".format(x,k)].index,columns=["label"])    
        finalsignals["stock{0}".format(x)]=samelabels(finalsignal)
        
        # Backtest:
        a,b=backtest(finalsignals["stock{0}".format(x)],dd["stock{0}".format(x)],initcapital=10000,tradingcost=0.005,name=tickers[x])
        
        buyandholdreturns.append(a)
        mymodelreturns.append(b)
        
        
print(np.mean(buyandholdreturns))       
print(np.mean(mymodelreturns))
    
    
###########################################################################################################
# QUICK BACKTEST:
###########################################################################################################
                                 
for k in range(2,3): # use time periods here later
    for x in range(len(tickers)):  # all stock one by one
        
        a,b=backtest(finalsignals["stock{0}".format(x)],dd["stock{0}".format(x)],initcapital=10000,tradingcost=0.005,name=tickers[x])
        
        buyandholdreturns.append(a)
        mymodelreturns.append(b)

# Collect all the returns in Dataframe:    
results=pd.DataFrame(np.column_stack([buyandholdreturns, mymodelreturns,tickers]), columns=['B&H returns', 'MLP model returns','Name'])
results.set_index('Name', inplace=True)

#Import to Excel
results.to_excel(r'C:\Users\Peter\Desktop\Resulttables2012.xlsx')

#############################################################################################################
# Create Violin Plots for the Rates of return: 
#############################################################################################################

sns.set(style="whitegrid")
sns.set(font_scale=1.3)

#----------------------------------------------------------------------------------------------------
tablets6= np.genfromtxt(r'C:\Users\Peter\Desktop\Master thesis\table6.csv', delimiter=',')
tablets6 = np.delete(tablets6, (0), axis=0)
finaltablets6=pd.DataFrame(tablets6,columns=["MLP model","Buy&Hold","RSI","William R"])    

# Violin plots:
fig=plt.figure(figsize=(20,10)) 
ax = sns.violinplot(data=finaltablets6)
plt.ylabel('Rate of Return', fontsize=14)
plt.title("Violin plot for the rates of return of different strategies in 2006",fontsize=14)

#----------------------------------------------------------------------------------------------------
tablets12= np.genfromtxt(r'C:\Users\Peter\Desktop\Master thesis\table12.csv', delimiter=',')
tablets12 = np.delete(tablets12, (0), axis=0)
finaltablets12=pd.DataFrame(tablets12,columns=["MLP model","Buy&Hold","RSI","William R"])    

# Violin plots:
fig=plt.figure(figsize=(14,7)) 
ax = sns.violinplot(data=finaltablets12)
plt.ylabel('Rate of Return', fontsize=14)
plt.title("Violin plot for the rates of return of different strategies in 2012",fontsize=15)

#----------------------------------------------------------------------------------------------------
tablets18= np.genfromtxt(r'C:\Users\Peter\Desktop\Master thesis\table18.csv', delimiter=',')
tablets18 = np.delete(tablets18, (0), axis=0)
finaltablets18=pd.DataFrame(tablets18,columns=["MLP model","Buy&Hold","RSI","William R"])    

# Violin plots:
fig=plt.figure(figsize=(14,7)) 
ax = sns.violinplot(data=finaltablets18)
plt.ylabel('Rate of Return', fontsize=14)
plt.title("Violin plot for the rates of return of different strategies in 2018",fontsize=15)

#---------------------------------------------------------------------------------------------------

# Violin plots for the year 2006: ------------------------------------------------------------------
fig=plt.figure(figsize=(20,10)) 
ax = sns.violinplot(data=finaltablets6,linewidth=2.5)
plt.ylabel('Rate of Return', fontsize=14)
plt.title("Violin plot for the rates of return of different trading strategies in 2006",fontsize=14)
plt.savefig('violin2006new.png')

# Violin plots for the year 2012: ------------------------------------------------------------------
fig=plt.figure(figsize=(20,10)) 
ax = sns.violinplot(data=finaltablets12,linewidth=2.5)
plt.ylabel('Rate of Return', fontsize=14)
plt.title("Violin plot for the rates of return of different trading strategies in 2012",fontsize=14)
plt.savefig('violin2012new.png')

# Violin plots for the year 2018: ------------------------------------------------------------------
fig=plt.figure(figsize=(20,10)) 
ax = sns.violinplot(data=finaltablets18,linewidth=2.5)
plt.ylabel('Rate of Return', fontsize=14)
plt.title("Violin plot for the rates of return of different trading strategies in 2018",fontsize=14)
plt.savefig('violin2018new.png')
