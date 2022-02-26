Machine learning algorithms like - LogisticRegression, SVM finds best possible hyperplane (N-1 dimension) dividing dataset (N dimension) into correct classed, by using objective function and optimizing it based on gradient descent.

SoftHyperplaneClassifier is a lightweight code to find approx hyperplane just by traversing dataset. Without using gradient descent.
Hence timecomplexity is O(M) , M: is length of dataset
Another advantage is this can be used with dynamic data. This means historical data can be processed only once by storing necessary state params and can be deleted. And new data can be processed to update the model params further. This means it can be dynamically trained with continuous generated data.
This is in contrast to LogisticRegression as it needs complete data to run gradient descent.
