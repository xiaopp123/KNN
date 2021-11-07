# -*- coding:utf-8 -*-


import random
import numpy as np
from operator import itemgetter
from collections import defaultdict


class LSHIndex(object):
    def __init__(self, hash_family, k, L):
        # hash函数
        self.hash_family = hash_family
        # L组哈希下，每个包含k个
        self.k = k
        # L个哈希函数组
        self.L = 0
        # hash表集合，集合大小为self.L，其中元素为元组，元组中第一个为k个哈希函数，
        # 第二个为k个哈希函数的哈希结果字典，key为当前向量k个哈希结果，val为当前向量索引
        self.hash_tables = []
        self.resize(L)

    def resize(self, L):
        if L < self.L:
            self.hash_tables = self.hash_tables[:L]
        else:
            # 如果L > self.L,则再新建（L-self.L) * self.k个哈希函数
            hash_funcs = [[self.hash_family.create_hash_func() for _ in range(self.k)] for _ in range(self.L, L)]
            self.hash_tables.extend([(g, defaultdict(lambda:[])) for g in hash_funcs])

    def hash(self, g, p):
        # k个哈希函数分别计算，然后combine
        return self.hash_family.combine([h.hash(p) for h in g])

    def index(self, points):
        self.points = points
        for g, table in self.hash_tables:
            for idx, p in enumerate(self.points):
                table[self.hash(g, p)].append(idx)
        self.tot_touched = 0
        self.num_queries = 0

    def query(self, q, metric, max_results):
        """ find the max_results closest indexed points to q according to the supplied metric """
        candidates = set()
        for g, table in self.hash_tables:
            matches = table.get(self.hash(g, q), [])
            candidates.update(matches)
        self.tot_touched += len(candidates)
        self.num_queries += 1

        # rerank candidates
        candidates = [(ix, metric(q, self.points[ix])) for ix in candidates]
        candidates.sort(key=itemgetter(1))
        return candidates[:max_results]

    def get_avg_touched(self):
        """ mean number of candidates inspected per query """
        return self.tot_touched/self.num_queries


def dot(u, v):
    return sum(ux*vx for ux, vx in zip(u, v))


class L2HashFamily:
    def __init__(self, w, d):
        # w是桶的个数，d是向量维度
        self.w = w
        self.d = d

    def create_hash_func(self):
        # each L2Hash is initialised with a different random projection vector and offset
        return L2Hash(self.rand_vec(), self.rand_offset(), self.w)

    def rand_vec(self):
        return [random.gauss(0, 1) for i in range(self.d)]

    def rand_offset(self):
        return random.uniform(0, self.w)

    def combine(self, hashes):
        """
        combine hash values naively with str()
        - in a real implementation we can mix the values so they map to integer keys
        into a conventional map table
        """
        return str(hashes)


class L2Hash(object):
    def __init__(self, r, b, w):
        self.r = r
        self.b = b
        self.w = w

    def hash(self, vec):
        return int((dot(vec, self.r) + self.b) / self.w)


def L2_norm(u, v):
    return sum((ux - vx) ** 2 for ux, vx in zip(u, v)) ** 0.5


class LSHTester:
    """
    grid search over LSH parameters, evaluating by finding the specified
    number of nearest neighbours for the supplied queries from the supplied points
    """

    def __init__(self, points, queries, num_neighbours):
        self.points = points
        self.queries = queries
        self.num_neighbours = num_neighbours

    def run(self, name, metric, hash_family, k_vals, L_vals):
        """
        name: name of test
        metric: distance metric for nearest neighbour computation
        hash_family: hash family for LSH
        k_vals: numbers of hashes to concatenate in each hash function to try in grid search
        L_vals: numbers of hash functions/tables to try in grid search
        """
        # 暴力搜索
        exact_hits = [[ix for ix, dist in self.linear(q, metric, self.num_neighbours+1)]
                      for q in self.queries]
        print(name)
        for k in k_vals:
            lsh = LSHIndex(hash_family, k, 0)
            for L in L_vals:
                lsh.resize(L)
                lsh.index(self.points)
                correct = 0
                for q, hits in zip(self.queries, exact_hits):
                    lsh_hits = [idx for idx, dist in lsh.query(q, metric, self.num_neighbours+1)]
                    if lsh_hits == hits:
                        correct += 1
                print("{0}\t{1}\t{2}\t{3}".format(
                    L, k, float(correct)/100, float(lsh.get_avg_touched())/len(self.points)))

    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        candidates = [(ix, metric(q, p)) for ix, p in enumerate(self.points)]
        return sorted(candidates, key=itemgetter(1))[:max_results]


def main():

    # create a test dataset of vectors of non-negative integers
    d = 5
    xmax = 20
    num_points = 1000
    # 随机构建1000个d维向量
    points = [[random.randint(0, xmax) for i in range(d)] for j in range(num_points)]
    print(len(points))

    num_neighbours = 2
    radius = 0.1
    for point in points[:num_points]:
        # 为每个点构建两个临近向量
        for i in range(num_neighbours):
            points.append([x+random.uniform(-radius, radius) for x in point])
    print(len(points))
    print(points[:num_points//10])

    tester = LSHTester(points, points[:num_points//10], num_neighbours)

    args = {'name': 'L2',
            'metric': L2_norm,
            'hash_family': L2HashFamily(10 * radius, d),
            'k_vals': [2, 4, 8],
            'L_vals': [2, 4, 8, 16]}
    tester.run(**args)


if __name__ == '__main__':
    main()

# https://engineering.purdue.edu/kak/distLSH/LocalitySensitiveHashing-1.0.1.html