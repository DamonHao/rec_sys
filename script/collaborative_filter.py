# -*- coding: utf-8 -*-

import math
import random


class CF(object):

	def _splitData(self, totalSplitNum, kthAsTest, seed):
		data = open(self._file_path)
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

	def _analyzeItems(self):
		popularity = {}
		all_items = set()
		train = self._train

		for user, items in train.iteritems():
			for item in items.iterkeys():
				popularity.setdefault(item, 0)
				popularity[item] += 1
				all_items.add(item)

		cold_item_num = 0
		for item, user_num in popularity.iteritems():
			if user_num < 2:
				cold_item_num += 1
		# N = min(len(all_items), 100)
		# rank = sorted(popularity.items(), key=lambda x: x[1], reverse=True)[0:N]
		print "data trend", len(all_items), len(train), cold_item_num

	def precisionAndRecall(self, K, N):
		hit = 0
		precision_all = 0
		recall_all = 0

		train = self._train
		test = self._test

		for user in train.iterkeys():
			test_set_user_items = test.get(user, None)
			if test_set_user_items is None:
				continue

			rank = self.recommend(user, K, N)
			for item, pui, in rank.iteritems():
				if item in test_set_user_items:
					hit += 1

			precision_all += N
			recall_all += len(test_set_user_items)
		return hit / float(precision_all), hit / float(recall_all)

	def coverage(self, K, N):
		recommend_items = set()
		all_items = set()
		train = self._train
		num = 0

		for user in train.iterkeys():
			for item in train[user].iterkeys():
				all_items.add(item)

			rank = self.recommend(user, K, N)
			num += len(rank)
			for item in rank.iterkeys():
				recommend_items.add(item)
		print "coverage", num, len(recommend_items), len(all_items)
		return len(recommend_items) / float(len(all_items))

	def popularity(self, K, N):
		item_popularity = {}
		popularity = 0
		num = 0

		for user, items in self._train.iteritems():
			for item in items.iterkeys():
				item_popularity.setdefault(item, 0)
				item_popularity[item] += 1

		for user in self._train.iterkeys():
			rank = self.recommend(user, K, N)
			for item in rank.iterkeys():
				popularity += math.log(1 + item_popularity[item])
				num += 1

		popularity /= float(num)
		return popularity


class UserBasedCF(CF):

	def __init__(self, file_path):
		self._file_path = file_path
		self._splitData(5, 1, 0)
		self._userSimilarity()

	def _userSimilarity(self):
		# build inverse table for item_ users
		self.item_users = {}
		for user, items in self._train.items():
			for i in items.iterkeys():
				if i not in self.item_users:
					self.item_users[i] = set()
				self.item_users[i].add(user)

		# calculate co-rated items between users
		C = {}
		N = {}
		for i, users in self.item_users.items():
			for user1 in users:
				N.setdefault(user1, 0)
				N[user1] += 1
				C.setdefault(user1, {})
				for user2 in users:
					if user1 == user2:
						continue
					C[user1].setdefault(user2, 0)
					C[user1][user2] += 1 / math.log(1+len(users))

		# calculate finial similarity matrix W
		W = {}
		for user1, related_users in C.items():
			W.setdefault(user1, {})
			for user2, common_interest in related_users.items():
				W[user1][user2] = common_interest / math.sqrt(N[user1] * N[user2])

		self.W = W
		return self.W

	def recommend(self, user, K=3, N=10):
		rank = {}
		action_item = self._train[user].iterkeys()
		for related_user, similarity in sorted(self.W[user].items(), key=lambda x: x[1], reverse=True)[0:K]:
			for item, interest_intensity in self._train[related_user].items():
				if item in action_item:
					continue
				rank.setdefault(item, 0)
				rank[item] += similarity * interest_intensity
		return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:N])


class ItemBasedCF(CF):

	def __init__(self, file_path):
		self._file_path = file_path
		self._splitData(5, 1, 0)
		self._userSimilarity()

	def _userSimilarity(self):
		# calculate co- rated users between items
		C = {}
		N = {}

		for user, items in self._train.iteritems():
			for item1 in items.iterkeys():
				N.setdefault(item1, 0)
				N[item1] += 1

				C.setdefault(item1, {})
				for item2 in items.iterkeys():
					if item1 == item2:
						continue
					C[item1].setdefault(item2, 0)
					C[item1][item2] += 1
					# C[item1][item2] += 1 / math.log(1 + float(len(items)))

		# calculate finial similarity matrix W
		W = {}
		for item1, related_items in C.iteritems():
			W.setdefault(item1, {})
			max_similarity = 0

			for item2, common_user in related_items.iteritems():
				cur_similarity = common_user / math.sqrt(N[item1] * N[item2])
				W[item1][item2] = cur_similarity

				if cur_similarity > max_similarity:
					max_similarity = cur_similarity

			if max_similarity:
				for item2, similarity in related_items.iteritems():
					if W[item1][item2]:
						W[item1][item2] /= max_similarity

		self.W = W

	def recommend(self, user, K=3, N=10):
		rank = {}
		action_items = self._train[user]
		W = self.W

		for item1, interest in action_items.iteritems():
			for item2, similarity in sorted(W[item1].iteritems(), key=lambda x: x[1], reverse=True)[0: K]:
				if item2 in action_items:
					continue
				rank.setdefault(item2, 0)
				rank[item2] += interest * similarity

		return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:N])


MAX_DATA_NUM = 500000

if __name__ == '__main__':


	import os
	file_path = os.path.join(os.path.dirname(__file__), '../ml-1m/ratings.dat')
	# cf = UserBasedCF(file_path)
	# cf = ItemBasedCF(file_path)
	# print cf.recommend('1')
	# print cf.precisionAndRecall(5, 10)
	# print cf.coverage(5, 10)
	# print cf.popularity(5, 10)
