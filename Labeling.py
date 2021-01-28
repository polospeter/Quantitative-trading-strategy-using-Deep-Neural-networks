
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import matplotlib.lines as mlines


# For our automatic trading strategy is essential that we provide labeled historical dataset for our model to be able to
# learn on and use that knowledge to predict optimal future trading times. Therefore we are going to label our stock data
# into 3 possible classes/labels: BUY-SELL-HOLD. Various methods will be proposed here how we could categorize our past
# time series into this 3 investment actions.

#####################################################################################################################
#--METHOD 1--Discrete labeling---------------------------------------------------------------------------------------
#####################################################################################################################

def labelmethod(ts,window):
    t=len(ts)
    labels = pd.DataFrame(index=ts.index).fillna(0.0)
    labels['label'] = 0.0
    # labels['label'][0] = 1.0

    # look for optimal first buying point as well
    k=ts['Close'][1:(window)].idxmin(axis=0, skipna=True)
    labels['label'][k]=-1

    for i in range(t-window-1):
            
        if labels['label'][i]==-1:
            l=ts['Close'][i+1:(i+window)].idxmax(axis=0, skipna=True)
            #index of the maximum
            labels['label'][l]=1  # sell signal

        elif labels['label'][i]==1:
            k=ts['Close'][i+1:(i+window)].idxmin(axis=0, skipna=True)
             #index of the minimum
            labels['label'][k]=-1   # buy signal

    print("Buy:",labels[labels.label == -1].shape[0],
          "Sell:",labels[labels.label == 1].shape[0],
          "Hold:",labels[labels.label == 0].shape[0])

    return labels

# A faster implementation of the method:
def labelmethodfast(ts,window):
    t=len(ts)
    labels = pd.DataFrame(index=ts.index).fillna(0.0)
    labels['label'] = 0.0
    
    i=ts['Close'][1:(window)].idxmin(axis=0, skipna=True)
    labels.loc[i]['label']=-1
    
    datetime.timedelta(days=1)
    
    while i<labels.index.values[t-window-1]:
        
        if labels['label'][i]==-1:
            l=ts['Close'][i+datetime.timedelta(days=1):(i+datetime.timedelta(days=window))].idxmax(axis=0, skipna=True)
            #index of the maximum
            labels.loc[l]['label']=1  # sell signal
            i=l

        elif labels['label'][i]==1:
            k=ts['Close'][i+datetime.timedelta(days=1):(i+datetime.timedelta(days=window))].idxmin(axis=0, skipna=True)
             #index of the minimum
            labels.loc[k]['label']=-1   # buy signal
            i=k

    print("Buy:",labels[labels.label == -1].shape[0],
          "Sell:",labels[labels.label == 1].shape[0],
          "Hold:",labels[labels.label == 0].shape[0],
          )

    return labels

labelmethodfast(stock,25)

#--------------------------------------------------------------------------------------------------

def transformtocontinuos(labels):

    labelss=np.array(labels.label)
    newlabels=labelss

    for i in range(2,len(labels)-2):

        if labelss[i]==1:
            newlabels[i-1]=0.7
            newlabels[i+1]=0.7
            newlabels[i+2]=0.3
            newlabels[i-2]=0.3

        if labelss[i]==-1:
            newlabels[i-1]=-0.7
            newlabels[i+1]=-0.7
            newlabels[i+2]=-0.3
            newlabels[i-2]=-0.3

        newlabel=pd.DataFrame(newlabels,index=labels.index,columns=["label"])
    return newlabel

#####################################################################################################################
#--METHOD 2--PRInvestor Labeling-------------------------------------------------------------------------------------
#####################################################################################################################

# Function Buy finds the first optimal buy index
def Buy(x,buy_id,charge):
    sell_id=buy_id+1
    while sell_id<len(x) and x['Close'][sell_id]<=x['Close'][buy_id]*charge:
        if x['Close'][sell_id]<=x['Close'][buy_id]:
            buy_id=sell_id
        sell_id=sell_id+1;
    return buy_id,sell_id

# =============================================================================

# Function Sell finds the first optimal sell index
def Sell(x,sell_id,charge):
    buy_id=sell_id+1
    while buy_id<len(x) and x['Close'][buy_id]>=x['Close'][sell_id]/charge:
        if x['Close'][buy_id]>=x['Close'][sell_id]:
            sell_id=buy_id
        buy_id=buy_id+1;
    return sell_id,buy_id

# Labelling method from the PR.investor research paper

def OptimalLabelling(x,charge):
    new_buy_id=1
    n=len(x)
    labels=np.zeros((n,1))
    charge=(1+charge)/(1-charge)
    while new_buy_id<n:
        buy_id,sell_id=Buy(x,new_buy_id,charge) # getting optimal buy ID
        sell_id, new_buy_id=Sell(x,sell_id,charge) # getting optimal sell ID

        if sell_id==n and x['Close'][sell_id]<x['Close'][buy_id]*charge:
            break

        labels[buy_id]=-1 #buy signal
        labels[(buy_id+1):(sell_id)]=2 #hold signal
        labels[sell_id]=1 #sell signal

    labels = pd.DataFrame(data=labels,columns=["label"],index=x.index)

    print("Buy:",labels[labels.label == -1].shape[0],
          "Sell:",labels[labels.label == 1].shape[0],
          "Wait:",labels[labels.label == 0].shape[0],
          "Hold:",labels[labels.label == 2].shape[0])

    return labels

#--------------------------------------------------------------------------------------------------------
# Individual Buy and Sell signals (every other signal is zero)

def onlybuy(signals):
    buy=signals[signals==-1]*-1
    buy['label'].fillna(0, inplace=True)
    return buy


def onlysell(signals):
    sell=signals[signals==1]*1
    sell['label'].fillna(0, inplace=True)
    return sell

#---------------------------------------------------------------------------------------------------------

def onlybuyandwait(signals):
    buy=signals[signals.label==-1]
    wait=signals[signals.label==0]
    df=pd.concat([buy,wait])
    return df.sort_index()

def onlysellandhold(signals):
    sell=signals[signals.label==1]
    hold=signals[signals.label==2]*0
    df=pd.concat([sell,hold])
    return df.sort_index()


def sellandhold(x_test,y_test):
    sell=y_test.Sell[y_test.Sell==1]
    hold=y_test.Hold[y_test.Hold==1]*0

    df=pd.concat([sell,hold])
    df=df.sort_index()

    tra=x_test.ix[df.index]

    pd.DataFrame(x_test,index=df.index)

    return tra,df


def buyandwait(x_test,y_test):
    buy=y_test.Sell[y_test.Buy==1]
    wait=y_test.Hold[y_test.Wait==1]*0

    df=pd.concat([buy,wait])
    df=df.sort_index()

    tra=x_test.ix[df.index]

    pd.DataFrame(x_test,index=df.index)

    return tra,df

#####################################################################################################################
#--METHOD 3--Continuos Labeling-------------------------------------------------------------------------------------
#####################################################################################################################

# Trend labeling: (from the study A hybrid stock trading framework integrating technical analysis... )
def trendlabel(stock):
    stock['ma']=stock['Close'].rolling(window=15).mean()
    #stock['ma'] = stock['close'].ewm(span=15,min_periods=0,adjust=True,ignore_na=False).mean() # exponential weighted functions

    stock['ma'].diff()
    stock['diff'] = stock['ma'].diff() < 0 # if true then decreasing

    uptrend=stock[(stock['diff']==False)&(stock['Close']>stock['ma'])] # Up trend
    downtrend=stock[(stock['diff']==True)&(stock['Close']<stock['ma'])] # Down trend

    stock['Trend']="No"
    stock.loc[uptrend.index,'Trend']="Up"
    stock.loc[downtrend.index,'Trend']="Down"

    stock['Min']=stock['Close'].rolling(window=3).min().shift(-2)
    stock['Max']=stock['Close'].rolling(window=3).max().shift(-2)
    stock['Trend value']=(stock['Close']-stock['Min'])/(stock['Max']-stock['Min'])*0.5
    stock.loc[uptrend.index,'Trend value']=stock['Trend value']+0.5

    return stock['Trend value'].fillna(0)

#-------------------------------------------------------------------------------------------------------------------

def converttosignal(label):
    up=label>0.5
    down=label<=0.5
    #label.loc[up]="Up"
    #label.loc[down]="Down"
    label.loc[up]=1
    label.loc[down]=0
    df = pd.DataFrame(label)
    df['Signal']=label.diff()*(-1)
    return df  # we get a value in the in range of 0 to 1

#####################################################################################################################
#--METHOD 4--Piecewise Linear representain Method for Labelling -----------------------------------------------------
#####################################################################################################################

def plrlabeling(stock,threshold):

    stock=stock['Close']

    stock=np.array(stock)
    x=np.array(range(0,stock.size))

    coord=np.zeros((2, 2))
    coord[0,0]=0
    coord[0,1]=stock[1]
    coord[1,0]=stock.size
    coord[1,1]=stock[-1]

    #linepart=np.zeros(stock.size)
    line=[]
    maxdist=[]
    maxdist.append(threshold+1)
    j=0

    while  threshold < maxdist[j]:

        j=len(coord)-1
    #for j in range(1,100):

        diffs=np.diff(coord,axis=0)
        slope=diffs[:,1]/diffs[:,0]

        for i in range(j):
            s = slice(int(coord[i,0]),int(coord[i+1,0])) # make them integers to be acceptable for indices
            #x[s]=range(int(coord[i+1,0])-int(coord[i,0]))
            line[s]=(x[s]-coord[i,0])*slope[i]+coord[i,1]

        distances=abs(stock-line)
        maxdtime=np.argmax(distances) # time point of max distance
        maxdist.append(max(distances[1:(maxdtime-1)]))

        coord=np.append(coord,[[maxdtime,stock[maxdtime]]],axis=0) # adding new breaking point to the line
        #coord=np.sort(coord, axis=0)
        coord=coord[coord[:,0].argsort()] # sort element by increasing time

            #P3 perpendicular to a line drawn between P1 and P2.
            # distances=norm(np.cross(line[-1]-line[1],line[1]-stock))/norm(line[-1]-line[1])

    # Lets visualize these trend
    plt.figure(figsize=(15,8))
    plt.plot(stock[0:200]) #!!
    plt.plot(x[0:200],line[0:200])
    plt.title('PLR of Stock')
    plt.xlabel('Time')
    plt.ylabel('Price')

    return line,coord

#-----------------------------------------------------------------------------------------------------------

def plrtosignal(stock,threshold):
    stockpd=stock
    #stock=stock['Close']
    #stock=np.array(stock)

    trading=np.zeros(shape=(len(stock),1))
    trading[0]=1

    plrline,maxindex=plrlabeling(stock,threshold)

    trend=np.diff( maxindex,axis=0) # length of window and trend
    mask1=trend[:,1]>=0
    mask2=trend[:,1]<0
    combmask=np.array(mask1*1+mask2*(-1))
    ind=np.cumsum(trend[:,0]) # there was a -1 here
    ind=np.insert(ind, 0, 0)
    result=np.column_stack((trend, combmask,ind[1:]))

    for i in range(0,len(ind)-2):

        if result[i,2]==1:
            trading[int(result[i,3])]=1

        if result[i,2]==-1:
            trading[int(result[i,3])]=-1

    for i in range(0,len(ind)-2):

        for j in range(1,int(result[i+1,0])): # length of a trend period

            if result[i,2]==1 and result[i+1,2]==-1:

                trading[int(result[i,3]+j)]=trading[int(result[i,3]+j)-1]-2/(int(result[i+1,0]))

            if result[i,2]==-1 and result[i+1,2]==1:

                trading[int(result[i,3]+j)]=trading[int(result[i,3]+j)-1]+2/(int(result[i+1,0]))

            if (result[i,2]==1 and result[i+1,2]==1 and (j/result[i+1,0])<=0.5):

                trading[int(result[i,3]+j)]=trading[int(result[i,3]+j)-1]-1/(int(result[i+1,0]/2))

            if (result[i,2]==1 and result[i+1,2]==1 and (j/result[i+1,0])>0.5):

                trading[int(result[i,3]+j)]=trading[int(result[i,3]+j)-1]+2/result[i+1,0]

            if (result[i,2]==-1 and result[i+1,2]==-1 and (j/result[i+1,0])<=0.5):

                trading[int(result[i,3]+j)]=trading[int(result[i,3]+j)-1]+1/(int(result[i+1,0]/2))

            if (result[i,2]==-1 and result[i+1,2]==-1 and (j/result[i+1,0])>0.5):

                trading[int(result[i,3]+j)]=trading[int(result[i,3]+j)-1]-2/result[i+1,0]

    tradingpd=pd.DataFrame(trading,index=stockpd.index) # I have removed stock indices, cause there are weekends when there are not stock prices, so thats why the lines were uneven

    return tradingpd

#--------------------------------------------------------------------------------------------------------

# Final buy and sell signal predictions:
def exponensmooth(model,x_test,alpha,bound):

    predictions = model.predict(x_test)

    series=predictions
    ft=np.zeros(len(series))
    bar=np.zeros(len(series))
    ft[0]=series[0]

    for i in range(len(series)-1):
        ft[i+1]=alpha*series[i+1]+(1-alpha)*ft[i]
       
    # New approach:
    ran=max(predictions)-min(predictions)

    upper=ft+bound*ran
    lower=ft-bound*ran

    tradingsignals=np.zeros(len(series))
    count=0

    for i in range(len(series)):

        if series[i]>upper[i]:
            count=count+1
            tradingsignals[i]=1

        if series[i]<lower[i]:
            count=count+1
            tradingsignals[i]=-1

    # Plot:
    fig=plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(111, ylabel='Predicted trading signal')
    plt.plot(series,color='black',label="NN output")
    plt.plot(upper,linestyle='dashed',color='red',label="Upper bound")
    plt.plot(lower,linestyle='dotted',color='red',label="Lower bound")

    # Plot the "Sell" trades against the equity curve
    ax1.plot(np.where(tradingsignals== 1.0)[0],
             series[tradingsignals== 1.0],
             's', markersize=7, color='dodgerblue')

    # Plot the "Buy" trades against the equity curve
    ax1.plot(np.where(tradingsignals== -1.0)[0],
             series[tradingsignals == -1.0],
             's', markersize=7, color='blue')

    plt.legend(fontsize=12)

    tradingsignal=pd.DataFrame(tradingsignals,columns=['label'],index=x_test.index)

    clearedsignals=samelabels(tradingsignal) # Clear the labels
    
    return clearedsignals


#####################################################################################################################
#--VISUALIZE THE LABELS ---------------------------------------------------------------------------------------------
#####################################################################################################################

def plotlabels(stock,labels):

    if len(stock)!=len(labels):
        #stock=stock.ix[labels.index]
        stock=stock[labels.index[0]:labels.index[-1]]

    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(111, ylabel='Closing price')
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=16)
    # Plot the equity curve in dollars
    stock['Close'].plot(ax=ax1, lw=2.,grid=True)

    # Plot the "Sell" trades against the equity curve
    ax1.plot(labels.loc[labels.label == 1.0].index,
             stock['Close'][labels.label == 1.0],
             'v', markersize=10, color='red')

    # Plot the "Buy" trades against the equity curve
    ax1.plot(labels.loc[labels.label  == -1.0].index,
             stock['Close'][labels.label == -1.0],
             '^', markersize=10, color='limegreen')

    #Legend
    red_square = mlines.Line2D([], [], color='red', marker='v', linestyle='None',
                          markersize=14, label='Sell')
    green_star = mlines.Line2D([], [], color='limegreen', marker='^', linestyle='None',
                          markersize=14, label='Buy')

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Closing price in $', fontsize=14)
    plt.legend(handles=[green_star,red_square],fontsize=14)
    plt.title("CSCO",fontsize=15)
    plt.savefig('newstock.png')

    # Show the plot
    plt.show()

    return

#----------------------------------------------------------------------------------------------------

# Clear up consecutive labels:

def samelabels(predlabelss):
    # The input data would be a panda Dataframe

    t=len(predlabelss)
    newlabel=np.zeros(t)
    my_list=predlabelss.label

    buy = np.where(my_list == -1)[0] # all occurences of Buy labels
    sell = np.where(my_list == 1)[0] # all occurences of Sell labels

    firstsell=0
    firstbuy=0

    newlabelpd=pd.DataFrame(newlabel,index=predlabelss.index)
    newlabelpd.columns=["label"]

    if len(buy)>0:

        newlabel[buy[0]]=-1 # first buy signal

        if len(sell)>0:

            while buy[firstbuy]<sell[len(sell)-1] and buy[len(buy)-1]>sell[firstsell]:


                list=sell-buy[firstbuy]
                firstsell = np.where(list>0)[0][0]

                newlabel[sell[firstsell]]=1

                if buy[len(buy)-1]>sell[firstsell]:

                    list=buy-sell[firstsell]
                    firstbuy = np.where(list>0)[0][0]

                    newlabel[buy[firstbuy]]=-1

            newlabelpd=pd.DataFrame(newlabel,index=predlabelss.index)
            newlabelpd.columns=["label"]

    return newlabelpd

#######################################################################################################################

def plrtosignal(stock,threshold,my=0):
    stockpd=stock
    stock=stock['Close']
    stock=np.array(stock)

    trading=np.zeros(shape=(len(stock),1))
    trading[0]=0.5

    plrline,maxindex=plrlabeling(stock,threshold)

    trend=np.diff( maxindex,axis=0) # length of window and trend
    mask1=trend[:,1]>=0
    mask2=trend[:,1]<0
    combmask=np.array(mask1*1+mask2*(-1))
    ind=np.cumsum(trend[:,0])-1
    ind=np.insert(ind, 0, 0)
    result=np.column_stack((trend, combmask))

    for i in range(0,len(ind)-1):
        trading[int(ind[i+1])]=0.5

        for j in range(1,int(result[i,0])):

            if result[i,2]==1 and (j/result[i,0])<=0.5:
                trading[int(ind[i]+j)]=trading[int(ind[i]+j)-1]-1/(int(result[i,0]/2)*2)

            if result[i,2]==1 and (j/result[i,0])>0.5:
                trading[int(ind[i]+j)]=trading[int(ind[i]+j)-1]+1/result[i,0]

            if result[i,2]==-1 and (j/result[i,0])<=0.5:
                trading[int(ind[i]+j)]=trading[int(ind[i]+j)-1]+1/(int(result[i,0]/2)*2)

            if result[i,2]==-1 and (j/result[i,0])>0.5:
                trading[int(ind[i]+j)]=trading[int(ind[i]+j)-1]-1/result[i,0]

    tradingpd=pd.DataFrame(trading,index=stockpd.index)

    return tradingpd

########################################################################################################################

# Clear up consecutive same labels (version 2)

def samelabelsv2(predlabelss,stock):

    # Adding stock prices in the time frame of the test:
    stocktest=stock[(stock.index>=predlabelss.index[0]) & (stock.index<=predlabelss.index[-1])]

    # The input data would be a panda Dataframe
    t=len(predlabelss)
    newlabel=np.zeros(t)
    my_list=predlabelss.label

    buy = np.where(my_list == -1)[0] # all occurences of Buy labels
    sell = np.where(my_list == 1)[0] # all occurences of Sell labels

    firstsell=0
    firstbuy=0

    newlabelpd=pd.DataFrame(newlabel,index=predlabelss.index)
    newlabelpd.columns=["label"]

    if len(buy)>0:

        newlabel[buy[0]]=-1 # first buy signal

        if len(sell)>0:

            while buy[firstbuy]<sell[len(sell)-1] and buy[len(buy)-1]>sell[firstsell]:

                list=sell-buy[firstbuy]
                firstsell = np.where(list>0)[0][0]

                while stocktest['Close'][sell[firstsell]]<stocktest['Close'][buy[firstbuy]] and firstsell<(len(sell)-1):
                    firstsell=firstsell+1

                newlabel[sell[firstsell]]=1

                if buy[len(buy)-1]>sell[firstsell]:

                    list=buy-sell[firstsell]
                    firstbuy = np.where(list>0)[0][0]

                    newlabel[buy[firstbuy]]=-1

            newlabelpd=pd.DataFrame(newlabel,index=predlabelss.index)
            newlabelpd.columns=["label"]

    return newlabelpd
