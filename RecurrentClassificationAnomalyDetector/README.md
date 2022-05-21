
# Introduction to RecurrentClassificationAnomalyDetector

This is machine learning model to detect anomalies in time series dataset using combination of clustering, binary/ternary search, and ad-hoc algorithms.
The model is capable of detecting both segment anomalies (based on pattern match) and point (based on spikes/dips) anomalies.



## Input Dataset

Time series datset.

## Documentation

This model works by splitting dataset into segments and using them as datapoints.
As the objective is to find both segment and point anomalies. 
Segmentation is done starting with high segment length and reducing it by half on each iteration.





