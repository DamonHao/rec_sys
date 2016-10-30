# -*- coding: utf-8 -*-


def personalRank(graph, alpha, root, maxStep):
	rank = {x: 0 for x in graph.iterkeys()}
	rank[root] = 1
	# 开始迭代
	for k in range(maxStep):
		tmp = {x: 0 for x in graph.iterkeys()}
		# 取节点i和它的出边尾节点集合ri
		for i, relatedNodes in graph.iteritems():
			# 取节点i的出边的尾节点j以及边E(i,j)的权重edgeWeight, 边的权重都为1，在这不起实际作用
			for j, edgeWeight in relatedNodes.iteritems():
				# i是j的其中一条入边的首节点，因此需要遍历图找到j的入边的首节点，
				# 这个遍历过程就是此处的2层for循环，一次遍历就是一次游走
				tmp[j] += alpha * rank[i] / (1.0 * len(relatedNodes))

		# 我们每次游走都是从root节点出发，因此root节点的权重需要加上(1 - alpha)
		tmp[root] += (1 - alpha)
		rank = tmp

		# 输出每次迭代后各个节点的权重
		print 'iter: ' + str(k) + "\t",
		for key, value in rank.iteritems():
			print "%s:%.3f, \t" % (key, value),
		print

	return rank


if __name__ == '__main__':
	graph = {'A': {'a': 1, 'c': 1},
		'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
		'C': {'c': 1, 'd': 1},
		'a': {'A': 1, 'B': 1},
		'b': {'B': 1},
		'c': {'A': 1, 'B': 1, 'C': 1},
		'd': {'B': 1, 'C': 1}}

	personalRank(graph, 0.85, 'A', 100)