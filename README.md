# MSc-thesis

This repository includes all the files about my research during my master studies. The goal of my thesis was to create a deep neural network based trading strategy to predict optimal trading times for stocks.

# Abstract:
The main focus of my study was to predict the buy and sell decision points for stocks. I will do that by creating a classification model to capture trading signals that are hidden in historical data and learn from them, how to automatically categorise the future time series into optimal investment actions.

First, I need to convert the daily time series of stock data into a series of buy-sell-hold trigger labels/signals. For each time point an asset would be labelled either Buy, Sell or Hold. The method for this would be to use a specific time frame, like a 25-day moving window, where the local minimum of the closing prices in each of these periods would be labelled as Buy points and the local maximums as Sell points and every other time point between the two would be Hold labels.

After labelling, I would calculate for each stock on a daily basis various technical indicators (e.g. RSI, MACD, Williams \%R,...). Later I will use these indicators as input features for my deep neural network model to predict the buy-sell-hold data points on the previously unseen datasets, based on the knowledge the model has learnt from historical data. I will also focus on improving the performance of these neural network models by experimenting with the structure of the networks and fine-tuning different hyperparameters of the models.

After I have my best models, I will compare the financial performance of my trading strategy based on the prediction from these models with the Buy and Hold strategy and other simple trading strategies. To make sure that the comparison is valid between my own trading strategy and other strategies, I will use multiple stocks over multiple periods for the performance evaluation and also consider factors like transaction costs.

{\bf In summary, the main purpose of my thesis was to create a completely automated trading system using deep learning algorithms, which is profitable and does not require any human intervention.}
