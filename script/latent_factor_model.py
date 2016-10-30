# -*- coding: utf-8 -*-

import random
import heapq
import math


class LatentFactorModel(object):

	def __init__(self, filePath):
		self._filePath = filePath
		self._splitData(5, 1, 0)

	def precisionAndRecall(self, N):
		hit = 0
		precision_all = 0
		recall_all = 0

		train = self._train
		test = self._test

		for user in train.iterkeys():
			test_set_user_items = test.get(user, None)
			if test_set_user_items is None:
				continue

			rank = self.recommend(user, N)
			for item, pui in rank:
				if item in test_set_user_items:
					hit += 1

			precision_all += N
			recall_all += len(test_set_user_items)
		return hit / float(precision_all), hit / float(recall_all)

	def coverage(self, N):
		recommend_items = set()
		all_items = set()
		train = self._train
		num = 0

		for user in train.iterkeys():
			for item in train[user].iterkeys():
				all_items.add(item)

			rank = self.recommend(user, N)
			num += len(rank)
			for item, _ in rank:
				recommend_items.add(item)
		print "coverage", num, len(recommend_items), len(all_items)
		return len(recommend_items) / float(len(all_items))

	def popularity(self, N):
		item_popularity = {}
		popularity = 0
		num = 0

		for user, items in self._train.iteritems():
			for item in items.iterkeys():
				item_popularity.setdefault(item, 0)
				item_popularity[item] += 1

		for user in self._train.iterkeys():
			rank = self.recommend(user, N)
			for item, _ in rank:
				popularity += math.log(1 + item_popularity[item])
				num += 1

		popularity /= float(num)
		return popularity

	def _splitData(self, totalSplitNum, kthAsTest, seed):
		data = open(self._filePath)
		random.seed(seed)
		train = {}  # {user : {item: score}}
		test = {}
		count = 0

		for line in data:
			user, item, score, _ = line.strip().split("::")
			if random.randint(1, totalSplitNum) == kthAsTest:
				test.setdefault(user, {})
				test[user][item] = int(score)
			else:
				train.setdefault(user, {})
				train[user][item] = int(score)

			count += 1
			if count == MAX_DATA_NUM:
				break
		print "data num:", count
		self._train = train
		self._test = test

	def _sortPopularityItem(self):
		itemsPopularity = {}
		for items in self._train.itervalues():
			for item in items:
				itemsPopularity.setdefault(item, 0)
				itemsPopularity[item] += 1

		validNun = int(len(itemsPopularity) * 0.3)
		print "total, valid num", len(itemsPopularity), validNun
		assert validNun
		sortedItems = heapq.nlargest(validNun, itemsPopularity.iteritems(), key=lambda e:e[1])
		self._sortedItems = [item for item, _ in sortedItems]

	def buildUserAction(self, negativeRatio):
		self._sortPopularityItem()

		sortedItems = self._sortedItems
		maxSortedItemIndex = len(sortedItems)-1
		usersAction = {}

		for user, items in self._train.iteritems():
			action = {}
			# positive
			for item in items:
				action[item] = 1

			negative_num = 0
			# negative
			itemsLen = len(items)
			targetNegativeNum = itemsLen * negativeRatio
			for i in xrange(0, itemsLen * 2):
				item = sortedItems[random.randint(0, maxSortedItemIndex)]
				if item in action:
					continue
				action[item] = 0
				negative_num += 1
				if negative_num >= targetNegativeNum :
					break

			usersAction[user] = action

		self._usersAction = usersAction

	def trainModel(self, itemClassNum, iterNum, learnRate, overfitParam):
		self._itemClassNum = itemClassNum
		self._initModel()
		userToClass = self._userToClass
		itemToClass = self._itemToClass

		for step in xrange(iterNum):
			for user, items in self._train.iteritems():
				userAction = self._usersAction[user]
				for item, interest in userAction.iteritems():
					interestDiff = interest - self._predict(user, item)

					for classIndex in xrange(itemClassNum):
						userWeight = userToClass[user][classIndex]
						itemWeight = itemToClass[item][classIndex]
						userToClass[user][classIndex] += learnRate * (interestDiff * itemWeight - overfitParam * userWeight)
						itemToClass[item][classIndex] += learnRate * (interestDiff * userWeight - overfitParam * itemWeight)

			learnRate *= 0.9

	def recommend(self, user, N):
		has_items = self._train.get(user, None)
		if not has_items:
			return []

		candidates = []
		for item in self._itemToClass.iterkeys():
			if item in has_items:
				continue
			interest = self._predict(user, item)
			candidates.append((item, interest))

		return heapq.nlargest(N, candidates, key=lambda e:e[1])

	def _predict(self, user, item):
		interest = 0
		userToClass = self._userToClass
		itemToClass = self._itemToClass

		for index in xrange(self._itemClassNum):
			interest += userToClass[user][index] * itemToClass[item][index]
		return interest

	def _initModel(self):
		userToClass = {}
		itemToClass = {}
		epsilon = 0.1
		itemClassNum = self._itemClassNum
		for user, items in self._train.iteritems():
			userToClass[user] = [random.uniform(0, epsilon) for i in xrange(itemClassNum)]

			for item in items:
				if item not in itemToClass:
					itemToClass[item] = [random.uniform(0, epsilon) for i in xrange(itemClassNum)]

		self._userToClass = userToClass
		self._itemToClass = itemToClass


MAX_DATA_NUM = 100000


if __name__ == '__main__':
	import os
	filePath = os.path.join(os.path.dirname(__file__), '../ml-1m/ratings.dat')
	lfm = LatentFactorModel(filePath)
	lfm.buildUserAction(1)
	lfm.trainModel(5, 100, 0.02, 0.01)
	# print lfm.recommend('1', 10)
	# print lfm.precisionAndRecall(10)
	print lfm.coverage(10), lfm.popularity(10)