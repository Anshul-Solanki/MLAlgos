Machine learning algorithms like - LogisticRegression, SVM finds best possible hyperplane (N-1 dimension) dividing dataset (N dimension) into correct classes, by using objective function and optimizing it based on gradient descent.  

SoftHyperplaneClassifier is a lightweight code to find approx hyperplane just by traversing dataset. Without using gradient descent.  
Hence timecomplexity is O(M) , M: is length of dataset  
Another advantage is this can be used with dynamic data. This means historical data can be processed only once by storing necessary state params and can be deleted. And new data can be processed to update the model params further. This means it can be dynamically trained with continuous generated data.  
This is in contrast to LogisticRegression as it needs complete data to run gradient descent.  

TODOs:  
> learn best way to edit readme file  
> test this model with various datasets (including the dataset generator code) and compare results with sklearn Logistic regression model (also try with erroneous dataset)  
> Add basic details of algorithm  
> Describe in details how this algo is effecient (how this fits well with generic data set, and not creating overfitted model)  
> How is this different than just finding centers of cluster and predicting new datapoint into class whose center is nearest (inverse ratio of projected distance from hyperplane is used for classification.. and how probability of classification is estimated)  
