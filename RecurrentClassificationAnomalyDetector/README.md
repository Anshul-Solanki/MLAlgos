
# Introduction to RecurrentClassificationAnomalyDetector

This is machine learning model to detect anomalies in time series dataset using combination of clustering, binary/ternary search, and ad-hoc algorithms.
The model is capable of detecting both segment anomalies (based on pattern match) and point (based on spikes/dips) anomalies.


## Input Dataset

Time series datset.

## Data Preparation

This model works by splitting dataset into segments and using them as datapoints.

## Hight Level Description and Objective of this Model

Main idea is to perform clustering with dataset of segments.
Here each cluster would indicate set of segments with similar pattern.
And segments with anomaly will get separated into clusters of minimal size. 
So our aim is to find an **Optimal Cluster Distribution** which can result anomaly clusters. 
This cluster distribution is determined by various factors
and in most cases does not result correct anomaly clusters, or even result false anomalies. 
These factors include selection of segment length, 
computing distance between segments, 
and threshold used to classify segments to a cluster based on distance.
Large segment length can result pattern anomalies, 
and small segment length can result point anomalies. 
Manhattan distance can be used to calculate distance between segments. 
But this might not work as expected when segments have similar pattern, but are shifted by some length.
Accounting for this shift in length is important to get correct distance. 
Threshold will be used as parameter to compare the distance, 
and segment gets classified to cluster only if distance is less than threshold. 
When classifying a segment, it is possible that distance is less than threshold to multiple clusters. 
Best way to address this conflict is by sorting clusters by their sizes 
and compare distance of segments to clusters one at a time, starting with cluster with smallest size. 
This will ensure clusters with minimal size are real anomalies. 
Finally we still need to filter false anomalies, as anomaly clusters might be formed with segments processed in last iterations 
which have only slightly more distance from non-anomaly clusters.
The overall algorithm also needs good time optimization using binary/ternary search techniques to find optimal cluster distributions and 
calculating segment distances with shift.

## Terminologies

| Term        | Description |
| ----------- | ----------- |
| dataset                       | Dataset representing list of segments prepared by segmentation of time-series data |
| cluster distribution          | Set of cluster sizes and its centers after performing clustering of complete dataset        |
| optimal cluster distribution  | Ideal cluster distribution where non-anomaly datapoints are classified into clusters of large size, and anomaly clusters of minimal size  |
| anomalyRatio                  | Used to identify anomaly cluster size as ratio of size to average cluster size in cluster distribution  |
| threshold                     | maximum distance to the cluster center for segment to be classified into that cluster |
| optimal threshold             | optimal value of threshold which gives optimal cluster distribution on clustering for given segment length |

## Segment length selection

As the objective is to find both segment and point anomalies. 
Segmentation is done starting with high segment length and reducing it by half on each iteration. 
So this is iterative algorithm to find optimal cluster distribution for given segment length and aggregating results for all segment lengths.

## Clustering algorithm

Given fixed values of: segment length, threshold, and complete dataset.
This algorithm is inspired by standard K-Means / KNN clustering. 
But as anomaly detector is based on repeated classification. 
Our focus is to make this very efficient with average case time complexity O(N).  

The entier dataset is traversed only once through segments separated by sliding length. 
For example if length of dataset is 100. 
Traversal can result into following segments with segment length of 10.  
S1=[0:10], S2=[10:20], S3=[15:25], S4=[23:33], .. ., Sn=[90:100]  
Si indicates ith segment  
and segments can be separated with variable sliding length 
to account for pattern match with shift (which is discussed in further sections)  
For example if this results into two clusters:  
[S1, S3, .. Sn] [S2, S4 .. Sm]  
The reason why S3 starts from 15 instead of 20 is because pattern [0:10] matches with [15:20]   
When shifting segment [20:30] backwards to [15:20] prevents forming false clusters and in turn prevents forming false anomaly clusters  

Clustering algorithm works as per below steps to achieve time complexity O(N*C), where N is number of segments, and C is number of clusters  

**Starting with first segment**  
first segment S1 = [0 : SegmentLength] is used as center of first cluster  

**Classification at ith iteration**  
Compare ith segment (i from 2 to N) with existing centers 
starting with clusters sorted with ascending order. 
This way classification is priotized more for clusters with anomaly size to prevent false anomaly.  

**Maintaining sorted order of clusters**  
Sorting of clusters should not impact time complexity. 
If clusters are sorted in ascending order of size. 
We start with single cluster with S1 on first iteration. 
On ith iteration if new segment gets classified into cluster at index k  
Then size of cluster[K] will increase by one, 
and size of cluster[K+1] can be either equal or more than size of cluste[K], as original cluster distribution is sorted  
So, we check if size of cluster[K] is increased and swap the two cluster positions along with their respective centers 
This way sorting order is maintained throught iterations without any extra overhead. 
This can be illustrated well with below example:  
Consider set of segments [a,b,a,b,b,c,c,c,c], each character representing segments of particulat pattern.  
| Iteration        | Cluster Distribution |
| ----------- | ----------- |
| 1           | [a]         |
| 2           | [a], [b]    |
| 3           | [b], [a,a]  |
| 4           | [b,b], [a,a] |
| 5           | [a,a], [b,b,b] |
| 6           | [c], [a,a], [b,b,b]|
| 7           | [c,c], [a,a], [b,b,b] |
| 8           | [a,a], [c,c,c], [b,b,b] |
| 9           | [a,a], [b,b,b], [c,c,c,c] | 

Notice cluster swapping at iteration 3, 5, 8 and 9

And we do not really store complete list of segment in each cluster in the memory. 
Instead we just store centers and the size of clusters as cluster distribution to achieve good memory optimization. 

## Optimal Cluster Distribution

The result of clustering algorithm in previous section is a cluster distribution. 
And next step is to analyze cluster distribution to check whether anomalies can be extracted 
and changing threshold value in a way that clustering in next iteration has good chance of resulting an optimal cluster distribution.  

With simple observation, we know how changing threshold actually impacts cluster distribution. 
If threshold is 0, resulting cluster distribution has large number of clusters but with very less size. 
If threshold is maximum limit, resulting cluster distribution has single cluster with size of complete dataset. 
This maximum can be easily estimated by aggregating maximum distance of all segments from first segment 
because first segment is center of first cluster by default.

This shows us the general trend:  
Large threshold value results small number of clusters with large size  
Small threshold value results large number of clusters with small size  

None of the above scenario provides optimal cluster distribution  
This can be best illustrated with below table:  

For constant size of dataset, approx = 500
| Threshold        | Cluster Distribution | Average cluster size |
| ----------- | ----------- | ----------|
| 10          | [200, 150, 70, 20, 15, 15, 10, 5, 5, 5, 3, 1] | 38 |
| 1000        | [500]                                         | 500 |
| 100         | [300, 199, 1]                                 | 167 |

From above table, it is not possible to find anomaly clusters for threshold values 10 and 1000.  
But for threshold = 100, it shows 300 segments of similar pattern, and 199 segments of another similar pattern.  
The remaining cluster of size = 1 can be considered as anomaly.  
This is indeed the optimal cluster distribution we want to find.  

Hence, optimal cluster distribution cannot be found using very less or very large value of threshold. 
But can only be found using an intermediate optimal threshold value.  
To find this optimal threshold, we cannot simply iterate through all possible values of threshold and do clustering at each step. 
This will make algorithm too slow.
Binary search algorithm can be used to do this efficiently. 

## Binary Search to find Optimal Threshold

