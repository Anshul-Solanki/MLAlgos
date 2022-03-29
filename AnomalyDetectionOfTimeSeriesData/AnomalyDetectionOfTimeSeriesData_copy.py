'''
write pseudo code first!!

# get timeseries data
# divide data into segments of length L and sliding length SL
# start with L = totalsize / 10 and SL = L / 5
# then keep reducing L / 10 till SL >= 1

# use soft k-means to cluster data, below algo details:
# select first datapoint as first center
# compare distance of new data point with existing centers, if distance is less than threshold then add this new data point into that center
# sort the centers in descreasing size when comparing with new data point, check below example:
# [a]
# [a] [b]
# [b,b] [a] (swap a and b as size of b > a)
# [b,b] [a] [c]
# [b,b] [c,c] [a]
# [c,c,c] [b,b] [a]
# this ensures that new data point has better chance of getting classified into clusters of maximum size and not be identified as anomaly
# in the above example, suppose cluster a represents anomaly
# if clusters are arranged with increasing size
# new datapoint x gets classified into a even if its distance from a and b was below threshold
# if clusters are arranged with decreasing size
# new datapoint x gets classified into a only if its distance from other centers was more then thershold

# In the process of finding optimal threshold, for each classification, just find the cluster distribution and its centers
# then do classification for last time and find anomaly points..

# Find max threshold which classifies data into single cluster using soft k-means algo
# soft k-mean algo

# Need to find optimal threshold where anomalies get separated into few clusters of very small size
# use binary search to achieve this
# rule of convergence with BS:
# Anomaly points are not lost if we reduce threshold:
# like 1 is included in both cluster distributions below:
# 300, 199, 1
# 300, 197, 2 ,1

(makes sure this converges correctly)
# if - meanClusterSize / smallestClusterSize >= 100  (optimal threshold)
	reduce threshold (by current threshold)
# if - totalSize / smallestClusterSize >= 100 (non optimal threshold)
	increase threshold (by current threshold + 1)
# if - totalSize / smallestClusterSize < 100 (non optimal threshold)
	reduce threshold (by current threshold)

once optimal threshold is converged, find anomaly points if present using below criteria -
all data points classified with cluster size < mean cluster size / 100

Filter anomaly points:
suppose we get following segments as anomaly
[55, 65] [57, 67] , [58, 68], [82,92]
here actual anomaly points could be 65 and 92 (as they are included in all segments)
if [x, x+10] is first anomaly segment, we skip all segments which include x+10
and return anomaly point as x+10

# Questions:
#1 is there more effective way of finding centers ranther then finding them on traversal?
#2 check is convergence can be improved and what is gurantee of finding anomaly if it exists?
#3 can we figureout no anomaly result just by looking at lowerbound and upperbound of threshold and its cluster ditributions?
#4 could optimal threshold be float -> No, because Manhattan distance is always integer

Aim is to find actual anomaly points as would be identified by looking dataset manually

3/27
New idea:
change algo to find distance bw segments
instead move segments to best position to align pattern then take distance, again using bs bw 0 - sl
'''

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock

class RecurrentClassificationAnomalyDetector () :

	@staticmethod
	# X: timeseries numerial data of type list
	def FindAnomalyPoints (X) :
		totalSize = len(X)
		segmentLen = int(totalSize / 10)
		anomalyPoints = []

		while (segmentLen >= 1) :

			#Evaluate cluster distribution for anomaly
			slidingLen = segmentLen #max(1, int(segmentLen/10))
			totalNumberOfSegments = totalSize / slidingLen

			anomalyRatio = 1 / math.sqrt(totalSize / segmentLen)  #1 / math.log2(totalSize / segmentLen) #0.01

			# use binary search to find optimal threshold
			minThresholdLimit = 0
			maxThresholdLimit = int(RecurrentClassificationAnomalyDetector.GetMaxThreshold(X, segmentLen))
			optimalThreshold = -1

			while (minThresholdLimit < maxThresholdLimit) :

				currentThreshold = int((minThresholdLimit + maxThresholdLimit) / 2)

				clusterSizes, centers = RecurrentClassificationAnomalyDetector.GetClusterDistribution(X, segmentLen, currentThreshold)

				avgClusterSize = sum(clusterSizes) / len(centers)
				smallestClusterSize = clusterSizes[-1]

				# optimal threshold condition -> reduce threshold to find more optimal threshold
				if (RecurrentClassificationAnomalyDetector.ClusterDistributionHasAnomalies(clusterSizes, segmentLen, anomalyRatio)) : 
					maxThresholdLimit = currentThreshold
					optimalThreshold = currentThreshold
				# non optimal threshold condition with many clusters of small sizes -> increase threshold
				elif (avgClusterSize < totalNumberOfSegments * anomalyRatio) :
					minThresholdLimit = currentThreshold + 1
				# non optimal threshold condition with large cluster sizes -> reduce threshold
				else :
					maxThresholdLimit = currentThreshold

			# check if anomaly is found
			if (optimalThreshold != -1) :
				nonAnomalyCenters = []

				clusterSizes, centers = RecurrentClassificationAnomalyDetector.GetClusterDistribution(X, segmentLen, optimalThreshold)

				#remove
				print("optimal threshold found for segment")
				print(segmentLen)
				print("threshold val")
				print(optimalThreshold)
				print("optimal cluster distribution is")
				print(*clusterSizes)
				print(*centers)

				avgClusterSize = sum(clusterSizes) / len(centers)

				centerIndex = 0
				while centerIndex < len(centers) :
					if ( clusterSizes[centerIndex] < avgClusterSize * anomalyRatio ) :
						break
					nonAnomalyCenters.append(centers[centerIndex])
					centerIndex += 1

				# add false anomaly centers into nonAnomalyCenters
				numberOfNonAnomalyClusters = len(nonAnomalyCenters)
				while centerIndex < len(centers) :
					if (RecurrentClassificationAnomalyDetector.IsFalseAnomalyCenter(X, segmentLen, centers, centerIndex, numberOfNonAnomalyClusters, optimalThreshold)) :
						nonAnomalyCenters.append(centers[centerIndex])
					centerIndex += 1

				anomalyPoints.append(RecurrentClassificationAnomalyDetector.GetAnomalyPoints(X, segmentLen, optimalThreshold, nonAnomalyCenters))

			# reduce segment length for next iteration
			segmentLen = segmentLen - 1 #int(segmentLen / 2) 

		return anomalyPoints



	@staticmethod
	def GetClusterDistribution (X, segmentLen, threshold) :
		slidingLen = segmentLen #max(1, int(segmentLen/10)) #int(segmentLen / 5)

		centers = []
		clusterSizes = []

		for start_pos in range(0, len(X), slidingLen) :
			end_pos = start_pos + segmentLen

			if (end_pos > len(X)) :
				break

			segment = X[start_pos : end_pos]

			centerIndex = 0

			while (centerIndex < len(centers)) :
				center_start_pos = centers[centerIndex]
				center_end_pos = center_start_pos + segmentLen

				if (RecurrentClassificationAnomalyDetector.CompareManhattanDist(X[center_start_pos : center_end_pos], segment, threshold)) : #(cityblock(X[center_start_pos : center_end_pos], segment) <= threshold) :
					break
				centerIndex = centerIndex + 1

			# check if segment is classified into existing center
			if (centerIndex < len(centers)) :
				clusterSizes[centerIndex] = clusterSizes[centerIndex] + 1

				#swap with previous center if size of this cluster is more
				while (centerIndex > 0 and clusterSizes[centerIndex] > clusterSizes[centerIndex-1]) :
					#swap centers
					temp = centers[centerIndex]
					centers[centerIndex] = centers[centerIndex-1]
					centers[centerIndex-1] = temp

					#swap cluster sizes
					temp = clusterSizes[centerIndex]
					clusterSizes[centerIndex] = clusterSizes[centerIndex-1]
					clusterSizes[centerIndex-1] = temp

					centerIndex -= 1
			else :
				centers.append(start_pos)
				clusterSizes.append(1)

		return clusterSizes, centers


	@staticmethod
	def GetMaxThreshold (X, segmentLen) :
		slidingLen = segmentLen #max(1, int(segmentLen/10)) #int(segmentLen / 5)
		firstSegment = X[0:segmentLen]

		maxThreshold = 0

		for i in range(slidingLen, len(X), slidingLen) :
			if (i+segmentLen > len(X)) :
				break

			segment = X[i : i+segmentLen]
			maxThreshold = max(maxThreshold, cityblock(firstSegment, segment))

		return maxThreshold

	@staticmethod
	def GetAnomalyPoints(X, segmentLen, threshold, nonAnomalyCenters) :
		anomalyPoints = []
		slidingLen = segmentLen #max(1, int(segmentLen/10)) #int(segmentLen / 5)

		for i in range(0, len(X), slidingLen) :
			if (i + segmentLen > len(X)) :
				break

			segment = X[i : i+segmentLen]
			isAnomalyPoint = True
			for center in nonAnomalyCenters :
				if (RecurrentClassificationAnomalyDetector.CompareManhattanDist(X[center : center+segmentLen], segment, threshold)) : #( cityblock(X[center : center+segmentLen], segment) <= threshold ) :
					isAnomalyPoint = False
					break

			if (isAnomalyPoint) :
				anomalyPoints.append(i)

		# Filter anomaly points
		filteredAnomalyPoints = []
		anomalyIndex = -1
		for anomalyPoint in anomalyPoints :
			if (anomalyIndex < anomalyPoint) :
				filteredAnomalyPoints.append(anomalyPoint + segmentLen / 2)
				anomalyIndex = anomalyPoint + segmentLen

		return filteredAnomalyPoints

	@staticmethod
	def CompareManhattanDist(seg1, seg2, threshold) :
		dist = 0

		for i in range(len(seg1)) :
			dist = dist + abs(seg1[i] - seg2[i]) # need to use [0] for csv dataset, check why?
			if (dist > threshold) :
				return False

		return True

	@staticmethod
	# NOTE: only compare non-anomaly centers
	def IsFalseAnomalyCenter(X, segmentLen, centers, centerIndex, nonAnomalyCenterLen, threshold) :
		result = False

		multiplier = 1.5 #1 + 1 / segmentLen

		print("number of nonAnomalyCenters")
		print(nonAnomalyCenterLen)

		for i in range( nonAnomalyCenterLen ) :
			if (i == centerIndex) :
				continue

			centerSegmentA = X[centers[centerIndex] : centers[centerIndex]+segmentLen]
			centerSegmentB = X[centers[i] : centers[i]+segmentLen]
			if (RecurrentClassificationAnomalyDetector.CompareManhattanDist(centerSegmentA, centerSegmentB, threshold*multiplier)) :
				result = True
				break

		return result

	# unused method
	@staticmethod
	def ClusterDistributionHasAnomalies(clusterSizes, segmentLen, anomalyRatio) :
		numberOfClusters = len(clusterSizes)
		avgClusterSize = sum(clusterSizes) / numberOfClusters
		anomalySize = avgClusterSize * anomalyRatio
		sumOfAnomalyClusterSizes = 0

		for clusterSize in clusterSizes :
			if (clusterSize <= anomalySize) :
				sumOfAnomalyClusterSizes += clusterSize
				# Do not consider anomaly as clusters
				# otherwise [30, 2] indicates anomalies but [30, 1, 1] does not
				numberOfClusters -= 1
			elif (clusterSize < avgClusterSize) :
				return False

		if (sumOfAnomalyClusterSizes == 0) :
			return False

		newAvgClusterSize = sum(clusterSizes) / numberOfClusters
		newAnomalySize = newAvgClusterSize * anomalyRatio

		if (sumOfAnomalyClusterSizes > newAnomalySize) :
			return False

		return True



# test code
def main() :
	#df = pd.read_csv ("anomalyDatasets\\art_daily_jumpsdown.csv") 
	#df = pd.read_csv ("anomalyDatasets\\ec2_cpu_utilization_5f5533.csv") # [1282.0] [2993.0] [2962.5] [2966.0] [1269.0, 2965.0]
	#df = pd.read_csv ("anomalyDatasets\\art_daily_flatmiddle.csv") # [2987.5] [3084.5] [1082.0, 1370.0] [3146.5] [1370.0]
	#df = pd.read_csv ("anomalyDatasets\\art_daily_jumpsup.csv") # [3001.5] [3124.5] [3078.0] [3083.0] [3090.5] [3100.0] [2987.0, 3095.0]

	#X = df.values.tolist() #[74.0] [79.0] [100.5] [102.0] [103.0]

	X = []
	
	for i in range(1000) :
		if (i > 100 and i < 105):
			X.append(25)
		elif (i > 200 and i < 205):
			X.append(75)
		elif (i%31 < 17) :
			X.append(80)
		else:
			X.append(20)
	'''
	Y = []

	for i in range(len(X)) :
		Y.append(X[i])

	for i in range(2000) :
		Y.append(X[i])

	for i in range(2000) :
		Y.append(X[i])	

	for i in range(2000) :
		Y.append(X[i])	

	for i in range(2000) :
		Y.append(X[i])	

	for i in range(2000) :
		Y.append(X[i])	
	'''

	anomalyPoints = RecurrentClassificationAnomalyDetector.FindAnomalyPoints(X)
	print(*anomalyPoints)

	# [321.5, 2041.5, 3481.5]
	
	plt.plot(X)
	plt.show()

if __name__ == "__main__" :	
	main()