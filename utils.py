import torch
import numpy as np
from actor import state_to_tensor


def normalize_intr_r(data, center_rnd_model):
    intr_r_lst = []
    for mini_batch in data:
        _, _, _, _, _, s_prime, _, _, _ = mini_batch
        with torch.no_grad():
            intr_r = center_rnd_model(s_prime)
        intr_r_lst.append(intr_r.detach().cpu().numpy())
    intr_r_lst = np.array(intr_r_lst)
    # print(intr_r_lst)
    intr_r_lst = intr_r_lst.view()
    # print(intr_r_lst)
    mean = intr_r_lst.mean()
    std = intr_r_lst.std()

    return mean, std

# TODO: actor网络的s和learner的s‘格式不一样，也就不可能，rnd loss通过反向传播更新RND网络，使predictor network所代表的的从s‘到feature的
#  映射接近target network，而s到feature的映射是没有被直接更新到的。但也不是不行，只要intrinsic reward是一个合理的值，能够随着访问次数增多而
#  减小，否则actor计算得到错误的intrinsic reward，进一步导致intrinsic value的更新目标偏离真实值。
# TODO：用将make_batch中s的部分抠出来，在actor计算intrinsic reward前处理一下
def recover_s_prime(s_prime):
    pass