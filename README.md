# Fire Prediction

## Model Architecture
This project include two kinds of models. The first is a normal Dense Neuron Network, while the other is our FirPRNN model. The first version of our FirPRNN model apply two tracks of RNN, respectively for long terms (days) and short terms (hours) data before the given datetime. We would then concatenate the two outputs arrays from the RNN tracks, and get the binary predicted result from the output of MLP with sigmoid activation function. The architecture is illustrated by the following figure.  

![alt text](https://github.com/daniel0321forever/fire_prediction/blob/main/plots/FirPRNN.png)  

However, as the data API we are currently using for our App cannot receive the most recent data from current time, we design another architecture that temporily drop out the short terms track of RNN, so that we can make prediction only based on the long terms (daily) data before.  

![alt text](https://github.com/daniel0321forever/fire_prediction/blob/main/plots/normalRNN.png)  

## Data and API
The data we use to train our model is mainly extracted from NASA POWER API and FIRMS API.
https://power.larc.nasa.gov/docs/services/api/  
https://firms.modaps.eosdis.nasa.gov/
