import numpy as np
import pandas as pd


class SoftHyperplaneClassifier() :

	# X: dataset with numeric feature values
	# Y: represents class of each datapoint, supported values 0/1
	def fit(self, X, Y) :

		self.X = X
		self.Y = Y

		# m: number of datapoints in dataset
		# n: number of features
		self.m, self.n = X.shape

		# objective is to find approx hyperplane equation separating dataset into correct classification based on Y
		self.W = np.zeros( self.n )
		self.b = 0

		# Construct hyperplane params with dataset

		# find centers of dataset classified into based on Y
		# we expect 2 centers with possible values of Y being 0/1
		# p1center, p2center will represent the two centers
		# these centers can be easily found by aggregating sum of datapoints based on class
		# then dividing by number of datapoints in class
		# p1cnt, p2cnt can represent number of datapoints in class = 1, class = 0 respectively

		p1_agg = np.zeros( self.n )
		p2_agg = np.zeros( self.n )
		p1cnt = 0
		p2cnt = 0

		for i in range( self.m ) :

			if (self.Y[i] == 1) :
				p1cnt += 1

				for j in range ( self.n ) :
					p1_agg[j] = p1_agg[j] + X[i][j]

			else :
				p2cnt += 1

				for j in range ( self.n ) :
					p2_agg[j] = p2_agg[j] + X[i][j]

		p1center = p1_agg / p1cnt
		p2center = p2_agg / p2cnt

		# At this point we know perpendicular vector to hyperplane as p1center - p2center
		# update weight as W = p1center - p2center
		# And equation of hyperplane will be WX + b = 0, W being the perpendicular vector 
		self.W = p1center - p2center

		# make W as unit vector
		w_len = np.sqrt( self.W.dot(self.W) )
		self.W = self.W / w_len

		# to find bias, we need a point through which hyperplane should pass
		# this could be a point between line joining p1center and p2center
		# best estimate of this point could be found by 
		# dividing line joining p1center to p2center by inverse ratio of average distance of datapoints to their respective centers
		# this average distance can be imagined as cylindrical radius of clusters of each class 
		# and this distance should be measured in direction of vector p1center - p2center, hence we use helper method called "getProjectionDist"

		# dist1: aggregate sum of projection distance from p1center of datapoints belonging to class=1 
		dist1_agg = 0 
		# dist2: aggregate sum of projection distance from p2center of datapoints belonging to class=0 
		dist2_agg = 0 

		for i in range ( self.m ) :
			if (Y[i] == 1) :
				dist1_agg += self.getProjectionDist(p1center, p2center, X[i])
			else :
				dist2_agg += self.getProjectionDist(p2center, p1center, X[i])

		# find cylindrical radius of clusters as average distance
		r1 = dist1_agg / p1cnt
		r2 = dist2_agg / p2cnt

		# find point on line joining centers, dividing line by inverse ratio of radius
		p_middle = ( p1center*r2 + p2center*r1 ) / (r1 + r2)

		# put this point in the hyperplane equation to find bias value
		self.b = -1 * self.W.dot(p_middle)

		return self

	# px: datapoint for which distance has to be measured
	# p1: center where datapoint has same classification
	# p2: other center with different classification
	# This method finds distance of px from p1 along vector starting p2 to p1
	def getProjectionDist(self, p1, p2, px) :
		p2p1vec = p1 - p2

		# find unit vector in direction of p2p1vec to find distance of px from p1 using dot product
		p2p1vec_len = np.sqrt( p2p1vec.dot(p2p1vec) )

		# unit vector
		p2p1unitvec = p2p1vec / p2p1vec_len

		# vector joining px to p1
		pxp1vec = p1 - px

		# return distance of px from p1 in direction of p2 to p1
		return abs( p2p1unitvec.dot(pxp1vec) )

	# X: datapoints to be predicted
	# Using sigmoid funtion to do the prediction (similar to logistic regression)
	def predict(self, X) :
		probability_class1 = 1 / ( 1 + np.exp( -1 * (X.dot(self.W) + self.b) ) )
		results = np.where( probability_class1 >= 0.5, 1, 0 )
		return results