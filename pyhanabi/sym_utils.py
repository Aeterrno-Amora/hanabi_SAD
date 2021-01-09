import torch
import math, itertools

def num_perms(n, m):
    return int(math.factorial(n) / math.factorial(n - m))

def sizes_to_size(n, sizes):
    return sum(num_perms(n, m) * sizes[m] for m in range(len(sizes)))

def mask_perm(perm, mask):
    return [mask.index(x) if x in mask else -1 for x in perm]

def unq_permutations(n, m, k):
    """unique m lenth permutations of n elements [0,1,..,k-1,-1,..,-1]
    also all the possible masked permutations"""
    return set(itertools.permutations(list(range(k)) + [-1] * (n-k), m))

def build_weight(parent_module, name, n, in_sizes, out_sizes):
    weight = {}
    for m1 in range(len(out_sizes)):
        if out_sizes[m1]:
            for m0 in range(len(in_sizes)):
                if in_sizes[m0]:
                    for perm in unq_permutations(n, m0, m1):
                        para = torch.nn.Parameter(torch.empty(out_sizes[m1], in_sizes[m0]))
                        parent_module.register_parameter(name + "_%d_%d_" % (m1,m0) + str(perm), para)
                        weight[m1, perm] = para
    return weight

def build_bias(parent_module, name, n, out_sizes):
    paras = []
    for m in range(len(out_sizes)):
        para = torch.nn.Parameter(torch.empty(out_sizes[m]))
        if out_sizes[m]:
            parent_module.register_parameter(name + "_%d" % m, para)
        paras.append(para)
    return paras

def linear(n : int, in_sizes, out_sizes, out_size, x, weight, bias):
    size = list(x.size())
    size[-1] = 1
    y = bias.repeat(torch.Size(size))
    #assert(y.size()[:-1] == x.size()[:-1])
    st_y = 0
    for m1 in range(len(out_sizes)):
        if out_sizes[m1]:
            for perm1 in itertools.permutations(list(range(n)), m1):
                yi = y.narrow(y.dim()-1, st_y, out_sizes[m1])
                st_x = 0
                for m0 in range(len(in_sizes)):
                    if (in_sizes[m0]):
                        for perm0 in itertools.permutations(mask_perm(range(n), perm1), m0):
                            xj = x.narrow(x.dim()-1, st_x, in_sizes[m0])
                            yi += xj.matmul(weight[m1, perm0])
                            st_x += in_sizes[m0]
                st_y += out_sizes[m1]
    return y
