# encoding=utf-8


import math
import random
from queue import PriorityQueue
from space_l2 import get_distance
from collections import defaultdict


class HNSW(object):

    def __init__(self, max_elements, M=16, ef_construction=200, random_seed=100):
        # 每个节点的邻接点数量
        self.M_ = M
        # 非0层节点邻接点数量
        self.maxM = self.M_
        # 第0层节点邻接点数量
        self.maxM0 = 2 * self.M_

        # 候选集合大小
        self.ef_construction = max(ef_construction, self.M_)

        # todo
        self.ef_ = 10
        # 节点索引从0递增
        self.cur_element_count = 0

        # 初始化访问访问数据
        self.visited = [0] * max_elements

        self.random_seed = random_seed
        random.seed(self.random_seed)

        # 初始化进入点和最大层
        self.enterpoint_node_ = -1
        self.maxlevel_ = -1

        # 第0层
        self.data_level0 = [(i, set()) for i in range(max_elements)]
        # 节点每层邻接点
        self.data_list = [defaultdict(set) for i in range(max_elements)]
        # 每个元素的层数
        self.element_levels = [-1] * max_elements

        self.mult = 1 / math.log(1.0 * self.M_)

        #
        self.label_lookup_table = dict()
        self.data_dict = dict()

    def get_random_level(self):
        level = -math.log(random.uniform(0, 1)) * self.mult
        return int(level)

    def get_linklist(self, cur_obj, level):
        return self.data_list[cur_obj][level]

    def search_base_layer(self, ep_id, data_point, layer):
        """
        以ep_id为入点在layer层找到离data_point最近的ef_construction个节点
        :param ep_id:
        :param data_point:
        :param layer:
        :return:
        """

        # python默认小顶堆, 堆中元素为(dist, id)
        # top_candidates中为距离的相反数，故队首为距离最大的点(取反后）
        top_candidates = PriorityQueue()
        # candidate_set中按距离从小到大存储
        candidate_set = PriorityQueue()
        # 访问节点集合
        visited = set()
        # 计算进入点与查询点的距离
        lower_bound = get_distance(data_point, self.data_dict[ep_id])
        top_candidates.put((-lower_bound, ep_id))
        candidate_set.put((lower_bound, ep_id))
        visited.add(ep_id)

        while not candidate_set.empty():
            # 选择候选集合中距离查入点最近的
            curr_el_pair = candidate_set.get()
            # 候选集合中最近的点大于最初距离值,直接跳出
            if curr_el_pair[0] > lower_bound:
                break
            cur_node_num = curr_el_pair[1]
            # 当前元素的邻接点
            data = self.get_linklist(cur_node_num, layer)
            # print(cur_node_num, layer, data)
            # 遍历每一个邻接点
            for candidate_id in data:
                # 如果已经访问过，则跳过
                if candidate_id in visited:
                    continue
                curr_obj = self.data_dict[candidate_id]
                dist = get_distance(data_point, curr_obj)
                visited.add(candidate_id)
                # 如果当前选择队列不满或者当前点的距离小于 最大边界距离，则将当前点加入到候选集合中
                if top_candidates.qsize() < self.ef_construction or lower_bound > dist:
                    candidate_set.put((dist, candidate_id))
                    top_candidates.put((-dist, candidate_id))

                # 选择队列已经满了，则弹出队首
                if top_candidates.qsize() > self.ef_construction:
                    top_candidates.get()

                # 取选择队列中的最大距离作为边界距离
                if not top_candidates.empty():
                    top_val = top_candidates.get()
                    lower_bound = -top_val[0]
                    top_candidates.put(top_val)
        return top_candidates

    def get_neighbors_by_heuristic(self, top_candidates, M):
        """
        在候选队列中选择M个节点
        :param top_candidates:
        :param M:
        :return:
        """
        # 候选队列中元素个数小于M，直接返回
        if top_candidates.qsize() < M:
            return

        return_list = []
        # queue_closest按照距离从小到大存储
        queue_closest = PriorityQueue()
        while top_candidates.qsize() > 0:
            val = top_candidates.get()
            queue_closest.put((-val[0], val[1]))

        while queue_closest.qsize() > 0:
            # 已经选择了M个节点，则跳出循环
            if len(return_list) >= M:
                break
            curent_data = queue_closest.get()
            # 当前节点到查询点的距离
            dist_to_query = curent_data[0]
            good = True
            # 遍历已选节点, 如果当前节点与查询点距离小于所有到已选点的距离，则选择当前节点
            for data in return_list:
                cur_dist = get_distance(curent_data[1], data[1])
                if cur_dist < dist_to_query:
                    good = False
                    break
            if good:
                return_list.append(curent_data)
        # 将已经选择的节点放到top_candidates中
        for data in return_list:
            top_candidates.put((data[0], data[1]))

    def mutually_connect_new_element(self, data_point, cur_c, top_candidates, level):
        """
        查询节点与已选择的节点进行连接
        :param data_point:
        :param cur_c:
        :param top_candidates:
        :param level:
        :return:
        """
        # 第0层和非0层能邻接点数量不同
        cur_max_M = self.maxM if level > 0 else self.maxM0
        # 使用启发算法从近邻集合中选择最近的M个
        self.get_neighbors_by_heuristic(top_candidates, self.M_)

        selected_neighbors = []
        while top_candidates.qsize() > 0:
            temp_data = top_candidates.get()
            selected_neighbors.append(temp_data[1])

        # 最近点作为下一层的进入点
        next_closet_entry_point = selected_neighbors[0]

        # 当前节点（cur_c）与已选择节点进行连接
        for idx in selected_neighbors:
            # 将idx加入到当前节点（cur_c）的邻接点集合中
            if level not in self.data_list[cur_c]:
                self.data_list[cur_c][level] = set()
            self.data_list[cur_c][level].add(idx)

            # 如果cur_c在idx的邻接点集合中
            is_cur_c_present = False
            for j in self.data_list[idx][level]:
                if j == cur_c:
                    is_cur_c_present = True
                    break

            if not is_cur_c_present:
                # 如果idx的邻接点数量小于最大邻接点数量，则将cur_c加入idx的邻接点集合中
                if len(self.data_list[idx][level]) < cur_max_M:
                    self.data_list[idx][level].add(cur_c)
                else:
                    # 否则，用启发算法找到idx最近的cur_max_M个节点进行连接
                    d_max = get_distance(self.data_dict[idx], self.data_dict[cur_c])
                    candisates = PriorityQueue()
                    candisates.put((d_max, cur_c))
                    for j in self.data_list[idx][level]:
                        candisates.put((get_distance(self.data_dict[j], self.data_dict[idx]), j))
                    self.get_neighbors_by_heuristic(candisates, cur_max_M)

                    self.data_list[idx][level] = set()
                    while candisates.qsize() > 0:
                        temp_data = candisates.get()
                        self.data_list[idx][level].add(temp_data[1])
        return next_closet_entry_point

    def add_point(self, data_point, label=-1):
        """
        :param data_point: 数据
        :param label:
        :return:
        """
        cur_c = self.cur_element_count
        self.cur_element_count += 1
        self.data_dict[cur_c] = data_point

        # 插入层次
        cur_level = self.get_random_level()

        # 进入点
        curr_obj = self.enterpoint_node_

        # debug
        if cur_c % 100 == 0:
            print('cur node: ', cur_c)

        if curr_obj != -1:
            # 进入点非空
            # 从最高处到当前层访问
            if cur_level < self.maxlevel_:
                cur_dist = get_distance(data_point, self.data_dict[curr_obj])
                for level in range(self.maxlevel_, cur_level, -1):
                    change = True
                    while change:
                        change = False
                        # 找当前元素的邻接点
                        data = self.get_linklist(curr_obj, level)
                        # 遍历邻接点
                        for t in data:
                            d = get_distance(data_point, self.data_dict[t])
                            # 如果邻接点到q的距离小于当前距离，以距离最小的邻接点作为下一个目标进行搜索
                            if d < cur_dist:
                                cur_dist = d
                                curr_obj = t
                                change = True

            for level in range(min(cur_level, self.maxlevel_), -1, -1):
                # 找到距离当前节点最近的, 此时top_candidates是小顶堆，堆中第一个元素是最远距离（已取负）
                top_candidates = self.search_base_layer(curr_obj, data_point, level)
                # 从查找的(ef_construction个)近邻点集合中选择最近的M个节点连接
                curr_obj = self.mutually_connect_new_element(data_point, cur_c, top_candidates, level)
        else:
            self.enterpoint_node_ = 0
            self.maxlevel_ = cur_level

        if cur_level > self.maxlevel_:
            self.maxlevel_ = cur_level
            self.enterpoint_node_ = cur_c
        # print(self.maxlevel_)
        return cur_c

    def search_knn(self, query_data, k):
        curr_obj = self.enterpoint_node_
        curdist = get_distance(query_data, self.data_dict[curr_obj])
        for level in range(self.maxlevel_, 0, -1):
            changed = True
            while changed:
                changed = False
                for i in self.data_list[curr_obj][level]:
                    dist = get_distance(self.data_dict[i], query_data)
                    if dist < curdist:
                        curdist = dist
                        curr_obj = i
                        changed = True

        top_candidates = self.search_base_layer(curr_obj, query_data, 0)
        result = []
        while top_candidates.qsize() > 0:
            temp_data = top_candidates.get()
            result.append((-temp_data[0], temp_data[1]))
        return result[::-1][:k]
