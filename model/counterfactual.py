import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils


def get_shortest_path_tmp(adj, subj_pos, obj_pos, len_):
    subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
    obj_pos = [i for i in range(len_) if obj_pos[i] == 0]
    G = nx.convert_matrix.from_numpy_array(adj)
    # print('adj', adj)
    # print('G', G.nodes)
    # print('subj', subj_pos)
    # print('obj', obj_pos)
    for i in subj_pos[1:]:
        G = nx.contracted_nodes(G, subj_pos[0], i, self_loops=False)
    if subj_pos[0] != obj_pos[0]: # subj obj differ
        for i in obj_pos[1:]:
            G = nx.contracted_nodes(G, obj_pos[0], i, self_loops=False)
    subj_pos_start = subj_pos[0]
    obj_pos_start = obj_pos[0]
    path = nx.shortest_path(G, subj_pos_start, obj_pos_start)
    return path


def path_to_mask(path, maxlen):
    pass

def get_shortest_path(adj, subj_start, subj_end, obj_start, obj_end):
    G = nx.convert_matrix.from_numpy_array(adj)
    # print('adj', adj)
    # print('G', G.nodes)
    # print('subj', subj_pos)
    # print('obj', obj_pos)
    for i in range(subj_start+1, subj_end+1):
        G = nx.contracted_nodes(G, subj_start, i, self_loops=False)
    for i in range(obj_start+1, obj_end+1):
        G = nx.contracted_nodes(G, obj_start, i, self_loops=False)
    path = nx.shortest_path(G, subj_start, obj_start)
    return path

def get_neighbor(head, len_, subj_pos, obj_pos):
    subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
    obj_pos = [i for i in range(len_) if obj_pos[i] == 0]
    # print(subj_pos, obj_pos)
    ent_pos = set(subj_pos+obj_pos)
    # subj_children = []
    # obj_children = []
    neighbors = set()
    for i in range(len_):
        h = head[i]
        if h == 0:
            continue
        h = h - 1
        if h in ent_pos:
            neighbors.add(i)
    for i in ent_pos:
        h = head[i]
        if h == 0:
            continue
        neighbors.add(h-1)
    return list(neighbors)
    

def head_list_to_tree(head, tokens, len_, prune, subj_pos, obj_pos):
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1 # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]

        cas = None

        subj_ancestors = set(subj_pos)
        cas_tmp = None
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h-1]
                subj_ancestors.add(h-1)
                h = head[h-1]

            if cas is None:
                cas = set(tmp)
                #cas_tmp = cas # added for NYT
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            if o >= len(head):
                print("error")
            h = head[o]

            tmp = [o]
            while h > 0:
                tmp += [h-1]
                obj_ancestors.add(h-1)
                h = head[h-1]
            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k:0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4) # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h-1] is not None
                nodes[h-1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root
