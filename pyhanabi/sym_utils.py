import torch
import math, itertools

def num_perms(n, m):
    return math.factorial(n) / math.factorial(n - m)

def sizes_to_size(n, sizes):
    return sum(num_perms(n, m) * sizes[m] for m in range(n))

def mask_perm(perm, mask):
    return [mask.index(x) if x in mask else -1 for x in perm]

def unq_permutations(n, m, k):
    """unique m lenth permutations of n elements [0,1,..,k-1,-1,..,-1]
    also all the possible masked permutations"""
    if k == 0:
        yield [-1 for i in range(m)]
    else:
        for p in unq_permutations(n, m, k-1):
            for i in range(m):
                if p[i] == -1:
                    p[i] = k-1
                    yield p
                    p[i] = -1

def build_weight(n, in_sizes, out_sizes):
    weight = {}
    for m1 in range(n):
        if out_sizes[m1]:
            for m0 in range(n):
                if in_sizes[m0]:
                    for perm in common_utils.unq_permutations(n, m0, m1):
                        weight[m1, perm] = torch.nn.Parameter(torch.Tensor(out_sizes[m1], in_sizes[m0]))
    return weight

def build_bias(n, sizes, constructor):
    return [torch.nn.Parameter(torch.Tensor(sizes[m])) for m in range(n)]

def linear(n, in_sizes, out_sizes, out_size, x, weight, bias):
    y = torch.Tensor(*x.size()[:-1], out_size)
    assert(y.size()[:-1] == x.size()[:-1])
    st_y = 0
    for m1 in range(n):
        if out_sizes[m1]:
            for perm1 in itertools.permutations(range(n), m1):
                yi = y.narrow(y.dim()-1, st_y, out_sizes[m1])
                st_x = 0
                for m0 in range(n):
                    if (in_sizes[m0]):
                        for perm0 in itertools.permutations(mask_perm(range(n), perm1), m0):
                            xj = x.narrow(x.dim()-1, st_x, in_sizes[m0])
                            if st_x: yi.add_(torch.nn.functional.linear(xj, weight[m1, perm0]))
                            else: yi.copy_(torch.nn.functional.linear(xj, weight[m1, perm0], bias[m1]))
                            st_x += in_sizes[m0]
                st_y += out_sizes[m1]
    return y
