# this can be used to generate dataset which is linearly separable
# for testing ML classification algorithms

import random
import numpy as np
import pandas as pd

class LinearlySeparableDataSetGenerator :

	def __init__ (self, numberOfFeatures=5, length=5000, featureMinVal=0, featureMaxVal=100) :
		self.n = numberOfFeatures
		self.m = length
		self.featureMinVal = featureMinVal
		self.featureMaxVal = featureMaxVal

	def GenerateAndGetDatSet(self) :
		dataSetSuccessfullyGenerated = False

		maxRetries = 20
		dataSet_WithClassValues = None

		while dataSetSuccessfullyGenerated == False and maxRetries > 0 :

			# use random coefficients between 0 and 1
			# And a random constant between 0 and 1
			coefficients = np.random.rand(self.n) - 0.5
			constant = random.uniform(0, 1) - 0.5

			dataSet = np.random.uniform(low=self.featureMinVal , high=self.featureMaxVal, size=(self.m, self.n) )

			# validate if dataset has appropriate classification ratio 
			# to avoid majority of datapoints in same class
			# assume number of datapoints in class=1 to be at-least 35% and at-most 65%

			# datapoint belongs to class=1 if projectedDataSetValues>0
			projectedDataSetValues = dataSet.dot(coefficients) + constant

			# number of datapoints in class=1
			dataSet_classValues = np.where(projectedDataSetValues > 0, 1, 0)
			class1DataPoints = dataSet_classValues.sum()

			dataSet_WithClassValues = np.column_stack((dataSet, dataSet_classValues))

			if (class1DataPoints > 0.35*self.m and class1DataPoints < 0.65*self.m) :
				print("dataset generated successfully")
				dataSetSuccessfullyGenerated = True
			else :
				print("dataset is skewed.. retrying again..")

			maxRetries -= 1

		if (dataSetSuccessfullyGenerated == False) :
			print("could not generate dataset after retrying many times")

		return dataSet_WithClassValues
