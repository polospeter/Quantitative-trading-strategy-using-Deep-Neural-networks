
import numpy as np
import datetime
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing
import urllib.request, json
import os
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, fbeta_score
from sklearn.metrics import average_precision_score
import seaborn as sns
from keras.models import Sequential # neural network
from keras import initializers

###########################################################################################################################
#------ Model preparation -------------------------------------------------------------------------------------------------
###########################################################################################################################

def modelprep(stockname,labels,startdate,enddate,trainratio,signalname="Sell"):
    
        """ This function calculates the most important technical analysis indicators based on a stock time series data for a given period. Furthermore it normalizes these
        features, and splits the data training-validation-testing sets as well, so they could be easily fed into the model.
    
    Parameters
    ----------
    stockname : string
        name of stock
    labels : pandas.Series with float values
        buy and sell labels provided for the stock
    startdate : datetime format
        starting date of stock data
    enddate : datetime format
        finaldate of stock data
    trainration : float
        ratio of train-test split    
    signalname : string
        buy or sell  
    
    Returns
    -------
    pandas dataframe
        final input features for the model
        
    """
    
    stock= stockname
    
    vol=stock['Volume'] # change placement of Volume column
    stock=stock.drop('Volume', axis=1)
    
    # Labeling -- add options to choose from different label methods:
  
    stock[['buy_signal']]=labels
    stock['Volume']=vol
    
    
     #---------------------------Define trainig period ----------------(later cross validation rather)------
    step=datetime.timedelta(days=1) # one day jump ahead
    
    hossz=len(stock.index) # number of indices,before removing NAN-s
    
    # Ratio of train/validation/test
    val1=trainratio
    
    # Training period -----------------------------------------------------------------------------------------------
    train_start = stock.index.min()
    train_end = stock.index[round(hossz*val1)]
    
    # Testing period ------------------------------------------------------------------------------------------------
    test_start = train_end+step   
    test_end = stock.index.max()
    
    print("Training for: "+str(train_start)+" - "+str(train_end)+
              " / Testing for: "+str(test_start)+" - "+str(test_end))   
    
    ##### Technical indicators #####
    #Add all the parameters to the dataframe(also use different time windows)
    
    #intervals=[3,6,9,14,21] # should we use all these intervals?
    intervals=[14]
    
    for i in range(len(intervals)):
        ma(stock,intervals[i])
        bias(stock,intervals[i])
        macd(stock,12,26)
        #bollinger(stock,intervals[i])
        stochasticline(stock,intervals[i]) #this is 2 parameters
        williamR(stock,intervals[i])
        atr(stock,intervals[i])
        cci(stock,intervals[i])
        rsi(stock,intervals[i])
    
    #stock[['buy_signal']]=stock[['buy_signal']].shift(-1) no need for daily shift
    
    stock.dropna(inplace=True) #!!!!!!!!!!!!!!
           
    ######################################################################################################################

    #--- Scaling of the dataset and features--------------------------------------------------------------------------------------------

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)  # these is other type scaler as well
    #scaler = StandardScaler(feature_range=(0, 1), copy=False) need changes
    #scaler = StandardScaler(copy=False) #need changes   

    numsig=len(set(stock.buy_signal))
    
    if numsig>4:
        scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)

    # INPUT variables:
    # we know that the first 5 are the price values of the stock
    
    # Train set =======================================================================================================
    
    X_train = stock[stock.columns[6:len(stock.columns)]][(stock.index>=train_start) & (stock.index<=train_end)]  # Input parameters for training:
    
    X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns) # scale every column
    
    y_train = stock[['buy_signal']][(stock.index>=train_start) & (stock.index<=train_end)] # signals
    
    # Test set ========================================================================================================
    
    X_test = stock[stock.columns[6:len(stock.columns)]][(stock.index>=test_start) & (stock.index<=test_end)]
    
    X_test = pd.DataFrame(scaler.fit_transform(X_test),columns=X_test.columns)
    
    y_test = stock[['buy_signal']][(stock.index>=test_start) & (stock.index<=test_end)]
    
    #=======================================================================================================================    
    #--------------------------------LABELING ----------------------------------------------------------------------
    # Convert Labels into independent columns

    # Count number of unique signals first:
    numsig=len(set(stock.buy_signal))

    #----------------------------------------------------------------------------------    
    if numsig==4:
        wait=(stock.buy_signal==0)*1
        buy=(stock.buy_signal==-1)*1
        sell=(stock.buy_signal==1)*1
        hold=(stock.buy_signal==2)*1
    
        combsignal=pd.concat([buy,sell,wait,hold],ignore_index=False,axis=1)  #convert to Panda dataframe
        combsignal.columns=["Buy", "Sell","Wait","Hold"]
    #---------------------------------------------------------------------------------- 
    if numsig==3:
        wait=(stock.buy_signal==0)*1
        buy=(stock.buy_signal==-1)*1
        sell=(stock.buy_signal==1)*1
        hold=(stock.buy_signal==2)*1
    
        combsignal=pd.concat([buy,sell,wait],ignore_index=False,axis=1)  #convert to Panda dataframe
        combsignal.columns=["Buy", "Sell","Hold"]       
    #----------------------------------------------------------------------------------   
    if numsig==2 and signalname=="Buy": # Buy and Wait
        others=(stock.buy_signal==0)*1
        buy=(stock.buy_signal==1)*1

        combsignal=pd.DataFrame(buy)

        combsignal.columns=["Buy"] # for binary model, one column is enough

    #---------------------------------------------------------------------------------      
    if numsig==2 and signalname=="Sell":# Sell and Hold (Binary labeling)
        sell=(stock.buy_signal==1)*1
        hold=(stock.buy_signal==0)*1 #not only hold in reality
    
        combsignal=pd.DataFrame(sell)

        combsignal.columns=["Sell"]
 
    #------------------------------------------------------------------------------------
    # In case of continuos labelling:
    if numsig>4: # since it is contiouns it takes up many values between 0 and 1
        combsignal=stock[['buy_signal']]
      
    #-------------------------------------------------------------------------------------        
        
    #===============================================================================================
    # Train--------------------------
    y_train=combsignal[train_start:train_end]
    #x_train= pd.DataFrame(X_train,index=y_train.index)
    x_train= X_train
    x_train.index=y_train.index
    
    # Test -------------------------  
    y_test=combsignal[test_start:test_end]
    x_test = X_test
    x_test.index=y_test.index
    #-------------------------------------------------------------------------------------
    # In case of Continuos labels:
    #targets_train=combsignal[train_start:train_end]
    #-------------------------------------------------------------------------------------
    
    return x_train,y_train,x_test,y_test



######################################################################################################################
#----------------------- EVALUATION ---------------------------------------------------------------------------------
######################################################################################################################

def modeleval(model,stockname,x_test,y_test,threshold=0.5):

    # Transform Predicted signals to dataframe------------------------------------------------------------------------
    predictions = model.predict(x_test)
    
    a,b=predictions.shape
    
    placemax=np.argmax(predictions,axis=1)
       
#==========================================================================================================================    
    # When the predictions are only one column incase of Binary classifier:
    if b==1: # How do we know whether it is buy or sell model?? 
        
        predlabels=np.zeros(a)
            
        for i in range(a):
            if (predictions[i,0]>=threshold):
                predlabels[i]=1 # Sell
                                
        predlabelss=pd.DataFrame(predlabels)      
        predlabelss.columns=['label']    
        
#         #Use the written labels instead of the numbers:
        predlabelss[predlabelss.label==1]=y_test.columns[0]
        predlabelss[predlabelss.label==0]="Others"
        
        
    if y_test.columns[0]=='Sell':
        labels = ["Sell","Others"]
            
    if y_test.columns[0]=='Buy':
        labels = ["Buy","Others"]
   
#==========================================================================================================           
    #y_testmax=pd.DataFrame(y_test.idxmax(axis=1, skipna=True)) # ??? what is this used for???

    ytestma=y_test
    ytestma=pd.DataFrame(y_test)
    #ytestma[ytestma==-1]=ytestma.columns[0]
    ytestma[ytestma==1]=ytestma.columns[0]
    ytestma[ytestma==0]="Others"
    
    # Classification report:---------------------------------------------------------------
    print(classification_report(ytestma,predlabelss.label,labels=labels))
    
    
   #Confusion matrix ---------------------------------------------------------------------
    mat = confusion_matrix(ytestma, predlabelss.label)
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,xticklabels=labels,yticklabels=labels) 
    plt.xlabel('true label')
    plt.ylabel('predicted label')

#================================================================================================================
     # Visualize the predicted labels --------------------------------------------------------------------
    stock=stockname
    test_start=y_test.index[0]
    test_end=y_test.index[-1]
    stocktest=stock[(stock.index>=test_start) & (stock.index<=test_end)]

#================================================================================================================
    if b==1: # Incase of Sell and Hold-- Binary (one column only)
      
        predlabels=np.zeros(a)
         
        for i in range(a):
            if predictions[i,0]>=threshold:
                if y_test.columns[0]=='Sell':  # make it if option for Sell and Buy
                    predlabels[i]=1 # Sell
                else:
                        predlabels[i]=-1 # Buy
            else:
                    predlabels[i]=0
             
    predlabelss=pd.DataFrame(predlabels)
    predlabelss.index=x_test.index
    predlabelss.columns=['label']
     
    d = pd.DataFrame(0, index=stocktest.index, columns=['label'])
     
    if y_test.columns[0]=='Sell':
        d.loc[predlabelss[predlabelss.label==1].index]=1
    else:
        d.loc[predlabelss[predlabelss.label==-1].index]=-1
         
    plotlabels(stocktest,d)
         
    return d 

#########################################################################################################################
#########################################################################################################################


# It is a cleaned up version of the ModelEval function, focusing of Binary predicting models:

# Model Eval short:

def modelevalshort(model,stockname,x_test,y_test,threshold=0.5):
        
    # Transform Predicted signals to dataframe------------------------------------------------------------------------
    predictions = model.predict(x_test)
    
    predictions[predictions<threshold]=0 
    predictions[predictions>=threshold]=1
                        
    predlabelss=pd.DataFrame(predictions,columns=['label'],index=y_test.index) 
    predictions2=pd.DataFrame(predictions,columns=['label'],index=y_test.index) 
    #predlabelss.columns=['label']    
        
    #Use the written labels instead of the numbers:
    predlabelss[predlabelss.label==1]=y_test.columns[0]
    predlabelss[predlabelss.label==0]="Others"
              
    if y_test.columns[0]=='Sell':
        labels = ["Sell","Others"]
            
    if y_test.columns[0]=='Buy':
        labels = ["Buy","Others"]
        
    #==========================================================================================================           
    #y_testmax=pd.DataFrame(y_test.idxmax(axis=1, skipna=True)) # ??? what is this used for???

    ytestma=y_test  
    ytestma=pd.DataFrame(y_test)
    ytestma[ytestma==1]=ytestma.columns[0]
    ytestma[ytestma==0]="Others"
    
    # Classification report:---------------------------------------------------------------
    print(classification_report(ytestma,predlabelss.label))
      
    #Confusion matrix ---------------------------------------------------------------------
    mat = confusion_matrix(ytestma,predlabelss.label,labels=labels)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,xticklabels=labels,yticklabels=labels) 
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    #=============================================================================
     # Visualize the predicted labels -------------------------------------------------------------------
    stock=stockname
    test_start=y_test.index[0]
    test_end=y_test.index[-1]
    stocktest=stock[(stock.index>=test_start) & (stock.index<=test_end)]  
          
    if y_test.columns[0]=='Buy':
        predictions2=predictions2*-1
        
    plotlabels(stocktest,predictions2)
         
    return predictions2


################################################################################################################################
################################################################################################################################


#----------------------- BACKTESTING ---------------------------------------------------------------------------


def backtest(labels,stock,initcapital=10000,tradingcost=0.005,name="Stock"):
    # In this version the trading cost is simply a percentage
    
    predlabelss=labels
    
    # The stock in the time frame of the test:
    stocktest=stock[(stock.index>=predlabelss.index[0]) & (stock.index<=predlabelss.index[-1])] 
    
    # Time points where we buy new stocks:
    buysignals=predlabelss.index[predlabelss.label==-1].tolist() # I think I do not even use these!!
    
    # Time points where we sell new stocks:
    sellsignals=predlabelss.index[predlabelss.label==1].tolist()

    t=len(predlabelss)
    
    positions=[0] * t
    numstocks=[0] * t # number of stocks
        
    numberstocks=0 # initial we do not own any stocks
    
    # Set the initial capital
    capital=initcapital

    for i in range(t):
        
        # Buy
        if predlabelss.label[i]==-1:
            numberstocks=int(capital/stocktest['Close'][i])
            capital=capital-numberstocks*stocktest['Close'][i]*(1+tradingcost) # we can adjust here by non-integer size of number of stocks
    
        # Sell
        if predlabelss.label[i]==1:
            capital=capital+numberstocks*stocktest['Close'][i]*(1-tradingcost)
            numberstocks=0
    
            
        positions[i]=capital+numberstocks*stocktest['Close'][i]
    
    posit=pd.DataFrame(data=positions,index=predlabelss.index,columns=['Close'])

    bh=stocktest['Close']*int(initcapital/stocktest['Close'][0])
    const=initcapital-bh[0]
    bh=bh+const
    
#    # Plot strategies--------------------------------------------------------------------------------
#    
    fig=plt.figure(figsize=(14,7)) 
    ax1 = fig.add_subplot(111)
    plt.plot(bh,label="Buy and Hold",lw=2.)
    plt.plot(posit,label="Model",lw=2.)
    
    ax1 = fig.add_subplot(111, ylabel='Closing price')
     
     # Plot the "Sell" trades against the equity curve
    ax1.plot(predlabelss.loc[predlabelss.label == 1.0].index, 
               bh[predlabelss.label == 1.0],
              'v', markersize=10, color='red',label="Sell")
     
     # Plot the "Buy" trades against the equity curve
    ax1.plot(predlabelss.loc[predlabelss.label  == -1.0].index, 
               bh[predlabelss.label == -1.0],
              '^', markersize=10, color='limegreen',label="Buy")
 #=============================================================================

    plt.grid()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Portfolio value in $', fontsize=14)
    plt.legend(fontsize=12)
    plt.title(name,fontsize=15)
    
    # The profits from the two different strategies-------------------------------------------------
    bhrevenue=stocktest['Close'][-1]*initcapital*(1-tradingcost)/stocktest['Close'][0]  # I think it should be stocktest??
    modelrevenue=posit['Close'][-1]
    
    # Returns: ----------------------------------------------------------------------------------------------
    bhreturn=bhrevenue/initcapital
    modelreturn=modelrevenue/initcapital
    plt.savefig('portfolio.png')
    
    return bhreturn-1, modelreturn-1

#########################################################################################################
    

def modelprepshort(stockname,labels):
    
    #stockname=d["stock{0}".format(x)]
    #labels=buy["buylabels{0}".format(x)]
    
    stock= stockname
    vol=stock['Volume'] # change placement of Volume column
    stock=stock.drop('Volume', axis=1)
    
    # Labeling -- add options to choose from different label methods:
    stock[['label']]=labels
    stock['Volume']=vol
    
    train_start = stock.index.min()
    train_end = stock.index.max()
    
     #---------------------------Define trainig period ----------------(later cross validation rather)------
  
    hossz=len(stock.index) # number of indices,before removing NAN-s
        
    #intervals=[3,6,9,14,21] 
    intervals=[14]
    
    for i in range(len(intervals)):
        ma(stock,intervals[i])
        bias(stock,intervals[i])
        macd(stock,12,26)
        #bollinger(stock,intervals[i])
        stochasticline(stock,intervals[i]) #this is 2 parameters
        williamR(stock,intervals[i])
        atr(stock,intervals[i])
        cci(stock,intervals[i])
        rsi(stock,intervals[i])
        dailyreturn(stock)
        
    stock.dropna(inplace=True) #!!!!!
    stock =stock[~stock.isin([np.nan, np.inf, -np.inf]).any(1)]
    
    # Optional:
    #stock=stock.drop('Volume', axis=1)
    
#--- Scaling of the dataset and features--------------------------------------------------------------------------------------------

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)  # these is other type scaler as well
    #scaler = StandardScaler(copy=False) #need changes   
    
    # Train set =======================================================================================================
    X_train = stock[stock.columns[6:len(stock.columns)]][(stock.index>=train_start) & (stock.index<=train_end)]  # Input parameters for training:
    
    X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns) # scale every column
    
    y_train = stock[['label']][(stock.index>=train_start) & (stock.index<=train_end)] # signals
    
    # Train--------------------------
    x_train= X_train
    x_train.index=y_train.index

    return X_train, y_train    

#########################################################################################################################
    
# Creating LSTM models:-------------------------------------------------------------------------------------

def prepdata(x_test2,y_test2,sequence_length):
    X_test_lstm = []
    y_test_lstm = []
    indices_lstm=[]

    for i in range(sequence_length, len(x_test2)):
        X_test_lstm.append(x_test2.values[i-sequence_length:i, :])
        y_test_lstm.append(y_test2.values[i,0])
        indices_lstm.append(y_test2.index[i])
    
    X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)

    y_test_lstm=pd.DataFrame( y_test_lstm,index=indices_lstm,columns=["Buy"]) # have to add Sell option here too
    
    return X_test_lstm,y_test_lstm,indices_lstm
