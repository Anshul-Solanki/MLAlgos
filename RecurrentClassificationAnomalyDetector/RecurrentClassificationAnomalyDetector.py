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
#2 check if convergence can be improved and what is gurantee of finding anomaly if it exists?
#3 can we figureout no anomaly result just by looking at lowerbound and upperbound of threshold and its cluster ditributions?
#4 could optimal threshold be float -> No, because Manhattan distance is always integer

Aim is to find actual anomaly points as would be identified by looking dataset manually

3/27
New idea:
change algo to find distance bw segments
instead move segments to best position to align pattern then take distance, again using bs bw 0 - sl

3/7
finding optimal point where segments match
find 4 points at distance 0, sl/30, sl/15, sl/10 -> names a,b,c,d
here b,c are mid points, and a,d are end points
find smallest number
if a is smallest : means minima has to exist bw a-b    : ends a-b
if b is smallest : means minima has to exist bw a-b-c  : ends a-c
if c is smallest : means minima has to exist bw b-c-d  : ends b-d
if d is smallest : means minima has to exist bw c-d    : ends c-d

continue above algo with new ends, until it gets converged to single point

4/10
do not pick next segment just after end of present segment
but instead try to shift if inward to optimal point
for example with segLen, startPos of previous seg, endPos of previous seg
start with new startPos of next seg as = startPos + 0.9 * segLen
and find optimal shift, it could be positive/negative
if optimalShift < d, next seg is selected as seg shifted inward
if optimalShift == d, start next seg startPos = startPos + 0.8 * segLen
repeat previous step only till 0.5 * segLen, even then if optimal shift is not achieved, then finalize startPos = startPos + segLen

4/30
simplify above algo: by finding optimal minimum bw startpos + seglen/2 : startpos + seglen
use existing algo to find the minimum
Note: this would work even for small scale patterns as we will keep reducing window on each iteration to find more optimal minimum

5/18
maybe create cluster distribution arranged in increasing size of clusters? to ensure anomaly centers are real anomalies?
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

		while (segmentLen >= 1) : # make this 1

			print("segmentLen", segmentLen)

			# use binary search to find optimal threshold
			minThresholdLimit = 0
			maxThresholdLimit = int(RecurrentClassificationAnomalyDetector.GetMaxThreshold(X, segmentLen))
			optimalThreshold = -1

			while (minThresholdLimit < maxThresholdLimit) :

				currentThreshold = int((minThresholdLimit + maxThresholdLimit) / 2)

				clusterSizes, centers = RecurrentClassificationAnomalyDetector.GetClusterDistribution(X, segmentLen, currentThreshold)

				totalNumberOfSegments = sum(clusterSizes)
				anomalyRatio = 1 / math.sqrt(totalNumberOfSegments)

				'''
				print("present threshold", currentThreshold)
				print("anomalyratio", anomalyRatio)
				print("present cluster dis")
				print(*clusterSizes)
				print(*centers)
				'''

				avgClusterSize = totalNumberOfSegments / len(centers)

				# optimal threshold condition -> reduce threshold to find more optimal threshold
				if (RecurrentClassificationAnomalyDetector.ClusterDistributionHasAnomalies(clusterSizes, totalNumberOfSegments, segmentLen, anomalyRatio)) : 
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

				print("optimal cluster dis found")
				print("optimal threshold")
				print(optimalThreshold)
				print("cluster dis")
				print(*clusterSizes)
				print(*centers)

				totalNumberOfSegments = sum(clusterSizes)
				anomalyRatio = 1 / math.sqrt(totalNumberOfSegments)

				avgClusterSize = totalNumberOfSegments / len(centers)

				# for analysis
				print("avgClusterSize", avgClusterSize)
				print("anomalyRatio", anomalyRatio)
				print("anomalySize", avgClusterSize * anomalyRatio)

				centerIndex = 0
				while centerIndex < len(centers) :
					if ( clusterSizes[centerIndex] <= avgClusterSize * anomalyRatio ) :
						break
					nonAnomalyCenters.append(centers[centerIndex])
					centerIndex += 1

				# for analysis
				print("non anomaly centers before filtering false anomalies")
				print(*nonAnomalyCenters)

				# add false anomaly centers into nonAnomalyCenters
				numberOfNonAnomalyClusters = len(nonAnomalyCenters)
				while centerIndex < len(centers) :
					if (RecurrentClassificationAnomalyDetector.IsFalseAnomalyCenter(X, segmentLen, centers, centerIndex, numberOfNonAnomalyClusters, optimalThreshold, anomalyRatio)) :
						nonAnomalyCenters.append(centers[centerIndex])
					centerIndex += 1

				print("non anomaly centers")
				print(*nonAnomalyCenters)

				trueAnomalies = RecurrentClassificationAnomalyDetector.GetAnomalyPoints(X, segmentLen, optimalThreshold, nonAnomalyCenters)
				'''
				if len(trueAnomalies) > 0 :
					print("true anomalies found for segmentLen")
					print(segmentLen)
					print("cluster dis")
					print(*clusterSizes)
					print(*centers)
					print("nonAnomalyCenters")
					print(*nonAnomalyCenters)
					print("anomaly centers")
					print(*trueAnomalies)
				'''

				anomalyPoints.append(trueAnomalies)

			# reduce segment length for next iteration
			segmentLen = int(segmentLen * 0.5)

			print("\n")

		return anomalyPoints

	@staticmethod
	def GetClusterDistribution (X, segmentLen, threshold) :
		nonAnomalyCenters = []
		centers, clusterSizes, ingnore = RecurrentClassificationAnomalyDetector.GetClusterDistributionAndAnomalyPoints(X, segmentLen, threshold, nonAnomalyCenters)

		return centers, clusterSizes

	@staticmethod
	def GetClusterDistributionAndAnomalyPoints (X, segmentLen, threshold, nonAnomalyCenters) :
		centers = []
		clusterSizes = []

		anomalyPoints = []
		collectNonAnomalyCenters = False
		if (len(nonAnomalyCenters) > 0) :
			collectNonAnomalyCenters = True

		start_pos = 0
		# use next_start_pos to prevent false anomalies where first segment is left out and patterns are found for other intervals
		next_start_pos = 0

		while (start_pos < len(X) - segmentLen) :

			if (start_pos > next_start_pos) :
				start_pos = next_start_pos
				next_start_pos = next_start_pos + segmentLen
			elif (start_pos == next_start_pos) :
				next_start_pos = next_start_pos + segmentLen

			end_pos = start_pos + segmentLen
			centerIndex = len(centers) - 1
			mshift = 0
			mdist = -1

			while (centerIndex >= 0) :
				center_start_pos = centers[centerIndex]
				center_end_pos = center_start_pos + segmentLen
				
				dist, shift = RecurrentClassificationAnomalyDetector.GetManhattanDistAndShift(X, center_start_pos, start_pos, segmentLen)

				if (mdist == -1) :
					mdist = dist
					mshift = shift
				elif (dist < mdist) :
					mdist =dist
					mshift = shift

				if (dist <= threshold) :
					break

				centerIndex = centerIndex - 1

			start_pos = start_pos - mshift

			# check if segment is classified into existing center
			if (centerIndex >= 0) :
				clusterSizes[centerIndex] = clusterSizes[centerIndex] + 1

				# Add anomaly points
				if (centers[centerIndex] not in nonAnomalyCenters and collectNonAnomalyCenters == True) :
					anomalyPoints.append(start_pos)

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

				# add to anomaly points
				if (start_pos not in nonAnomalyCenters and collectNonAnomalyCenters == True) :
					anomalyPoints.append(start_pos)

			start_pos = start_pos + segmentLen

		return clusterSizes, centers, anomalyPoints

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

		ignore1, ignore2, anomalyPoints = RecurrentClassificationAnomalyDetector.GetClusterDistributionAndAnomalyPoints(X, segmentLen, threshold, nonAnomalyCenters)

		# Filter anomaly points
		filteredAnomalyPoints = []
		anomalyIndex = -1
		for anomalyPoint in anomalyPoints :
			if (anomalyIndex < anomalyPoint) :
				filteredAnomalyPoints.append(anomalyPoint + segmentLen / 2)
				anomalyIndex = anomalyPoint + segmentLen

		#TODO: this should be anomaly window instead  of single points 
		return filteredAnomalyPoints

	@staticmethod
	def CompareManhattanDist(X, seg1_start, seg2_start, segmentLen, threshold) :
		dist, ignore = RecurrentClassificationAnomalyDetector.GetManhattanDistAndShift(X, seg1_start, seg2_start, segmentLen)
		if (dist <= threshold) :
			return True
		else :
			return False

	# seg1: center
	@staticmethod
	def GetManhattanDistAndShift(X, seg1_start, seg2_start, segmentLen) :
		a = 0
		d = int(segmentLen * 0.5)

		optimalDist = 0

		# if segLen is 1
		if (a == d) :
			optimalDist = RecurrentClassificationAnomalyDetector.GetManhattanDist(X, seg1_start, seg2_start, segmentLen)
			return optimalDist, 0

		while a < d :

			tSegLen = d-a
			b = a + int(tSegLen/3)
			c = a + int(2*tSegLen/3)

			#print("abcdval")
			#print(a, b, c, d)

			aDist = 0
			bDist = 0
			cDist = 0
			dDist = 0

			aDist = RecurrentClassificationAnomalyDetector.GetManhattanDist(X, seg1_start, seg2_start, segmentLen, a)
			bDist = aDist

			#print("aDist")
			#print(aDist)

			if b > a :
				bDist = RecurrentClassificationAnomalyDetector.GetManhattanDist(X, seg1_start, seg2_start, segmentLen, b)
			cDist = bDist

			#print("bDist")
			#print(bDist)

			if c > b :
				cDist = RecurrentClassificationAnomalyDetector.GetManhattanDist(X, seg1_start, seg2_start, segmentLen, c)
			dDist = RecurrentClassificationAnomalyDetector.GetManhattanDist(X, seg1_start, seg2_start, segmentLen, d)

			#print("cDist")
			#print(cDist)

			#print("dDist")
			#print(dDist)

			#print("evaluated distances")
			#print(aDist, bDist, cDist, dDist)

			if a < b :
				# converge only based on minima found in between window
				# minimas at end of windows might be false positive in case window length is more than pattern length
				if bDist <= cDist :
					d = c
				else :
					a = b
			# in this case a = b
			# resulting values of a,b,c,d can be: a,a,a+1,a+2 OR a,a,a,a+1
			# hence we can safely converge to single point corresponding to minimum distance
			else :
				if bDist <= cDist and bDist <= dDist :
					d = a # converge to a
					optimalDist = bDist
				elif cDist <= dDist :
					a = c # converge to c
					d = c
					optimalDist = cDist
				else :
					a = d
					optimalDist = dDist

		return optimalDist, a

	@staticmethod
	def GetManhattanDist(X, seg1_start, seg2_start, segmentLen, shift=0) :
		dist = 0

		for i in range(segmentLen) :
			dist = dist + abs(X[seg1_start+i] - X[seg2_start+i-shift]) # when using df try not to use list

		return dist

	@staticmethod
	# NOTE: only compare non-anomaly centers
	def IsFalseAnomalyCenter(X, segmentLen, centers, centerIndex, nonAnomalyCenterLen, threshold, anomalyRatio) :
		result = False

		for i in range( nonAnomalyCenterLen ) :
			if (i == centerIndex) :
				continue

			c1 = min(centers[centerIndex], centers[i])
			c2 = max(centers[centerIndex], centers[i])
			if (RecurrentClassificationAnomalyDetector.CompareManhattanDist(X, c1, c2, segmentLen, threshold*(1 + anomalyRatio))) :
				print("found false anomaly", centers[i], centers[centerIndex], threshold, threshold*(1 + anomalyRatio))
				result = True
				break

		return result

	@staticmethod
	def ClusterDistributionHasAnomalies(clusterSizes, totalNumberOfSegments, segmentLen, anomalyRatio) :
		numberOfClusters = len(clusterSizes)
		avgClusterSize = totalNumberOfSegments / numberOfClusters
		anomalySize = avgClusterSize * anomalyRatio
		sumOfAnomalyClusterSizes = 0

		'''
		print("check ClusterDistributionHasAnomalies")
		print("avgClusterSize", avgClusterSize)
		print("anomalySize", anomalySize)
		'''

		# to avoid scenario where cluster dis is too scattered. For example sizes from 100, 99, .., 3 ,2,1
		minSizeOfNonAnomalyClusters = sum(clusterSizes) * anomalyRatio

		#print("avgClusterSize when checking clusted dis", avgClusterSize)
		#print("minSizeOfNonAnomalyClusters", minSizeOfNonAnomalyClusters)

		for clusterSize in clusterSizes :
			if (clusterSize <= anomalySize) :
				sumOfAnomalyClusterSizes += clusterSize
				# Do not consider anomaly as clusters
				# otherwise [30, 2] indicates anomalies but [30, 1, 1] does not
				numberOfClusters -= 1
			elif (clusterSize < minSizeOfNonAnomalyClusters) : 
				return False

		#print("sumOfAnomalyClusterSizes", sumOfAnomalyClusterSizes)

		if (sumOfAnomalyClusterSizes == 0) :
			return False

		newAvgClusterSize = totalNumberOfSegments / numberOfClusters
		newAnomalySize = newAvgClusterSize * anomalyRatio

		if (sumOfAnomalyClusterSizes > newAnomalySize) :
			return False

		return True

# test code
def main() :
	#df = pd.read_csv ("anomalyDatasets\\art_daily_jumpsdown.csv") 
	#df = pd.read_csv ("anomalyDatasets\\ec2_cpu_utilization_5f5533.csv") 
	#df = pd.read_csv ("anomalyDatasets\\art_daily_flatmiddle.csv") 
	#df = pd.read_csv ("anomalyDatasets\\art_daily_jumpsup.csv") 

	#X = df.values.tolist()

	X = []
	for i in range(1000) :
		if (i > 100 and i < 105):
			X.append(20)
		elif (i > 200 and i < 205):
			X.append(80)
		elif (i%31 < 17) :
			X.append(80)
		else:
			X.append(20)

	anomalyPoints = RecurrentClassificationAnomalyDetector.FindAnomalyPoints(X)
	print(*anomalyPoints)

	plt.plot(X)
	plt.show()

if __name__ == "__main__" :	
	main()