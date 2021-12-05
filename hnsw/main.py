# encoding=utf-8

import random
import numpy as np
from hnsw import HNSW
from operator import itemgetter
from space_l2 import get_distance


def linear(query, points, k):
    candidates = [(ix, get_distance(query, p))for ix, p in enumerate(points)]
    return sorted(candidates, key=itemgetter(1))[:k]


def main():
    d = 5
    xmax = 20
    num_points = 1000
    # 随机构建1000个d维向量
    points = [[random.randint(0, xmax) for i in range(d)]
              for j in range(num_points)]

    num_neighbours = 2
    radius = 0.1
    for point in points[:num_points]:
        # 为每个点构建两个临近向量
        for i in range(num_neighbours):
            points.append([x+random.uniform(-radius, radius) for x in point])

    all_points = len(points)
    # 构建hnsw
    hnsw = HNSW(max_elements=all_points, M=20, ef_construction=100, random_seed=20)
    for point in points:
        hnsw.add_point(point)

    queries = points[:num_points//10]

    # 暴力搜索
    exact_hits = [[ix for ix, dist in linear(q, points, k=num_neighbours+1)]
                  for q in queries]

    correct = 0
    for q, hits in zip(queries, exact_hits):
        # 使用hnsw搜索
        hnsw_hits = hnsw.search_knn(q, k=num_neighbours+1)
        hnsw_hit_id = [t[1] for t in hnsw_hits]
        if hnsw_hit_id == hits:
            correct += 1
    print(correct * 1.0 / 100)


if __name__ == '__main__':
    main()

