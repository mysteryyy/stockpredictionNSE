# stockpredictionNSE
For predicting stock market returns.
A LSTM neural network is used for predicting the future retuns of the stocks of NIFTY50.
In general the historical price data is taken as input,however various transformation of these prices have been tested out as inputs to the neural networks.   
Moreover autoencoders are used to reduce the dimensionality of data progressively so that traditional clustering techniques can be applied to identify persistent price patterns which can be further used for future directional returns prediction.
