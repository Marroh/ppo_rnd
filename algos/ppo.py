import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
from actor import state_to_tensor


class Algo():
    def __init__(self, arg_dict, device=None):
        self.alpha = arg_dict["alpha"]
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = arg_dict["eps_clip"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]
        self.ratio = arg_dict["ratio"]

    def train(self, model, center_rnd_model, data, mean, std):
        tot_loss_lst = []
        pi_loss_lst = []
        entropy_lst = []
        move_entropy_lst = []
        v_loss_lst = []
        v_i_loss_lst = []
        rnd_loss_lst = []

        # to calculate fixed advantages before update
        data_with_adv = []
        for mini_batch in data:
            s, a, m, r, intr_r, s_prime, done_mask, prob, need_move = mini_batch  # s_prime is s_{t+1} contains obs & h
            # print("内部奖励和外部奖励的比例", r, intr_r)
            with torch.no_grad():
                # move 2 state for Q-learning
                pi, pi_move, v, v_i, _ = model(s)
                pi_prime, pi_m_prime, v_prime, v_i_prime, _ = model(s_prime)

            # external reward difference
            td_target = r + self.gamma * v_prime * done_mask
            delta = td_target - v  # [horizon * batch_size * 1]
            delta = delta.detach().cpu().numpy()

            # external reward advantage error
            advantage_lst = []
            advantage_e = np.array([0])
            for delta_t in delta[::-1]:
                advantage_e = self.gamma * self.lmbda * advantage_e + delta_t
                advantage_lst.append(advantage_e)
            advantage_lst.reverse()
            advantage_e = torch.tensor(advantage_lst, dtype=torch.float, device=model.device)

            # intrinsic reward difference
            td_intr_r = intr_r + self.alpha * v_i_prime * done_mask
            delta_intr = td_intr_r - v_i
            delta_intr = delta_intr.detach().cpu().numpy()

            # intrinsic reward advantage error
            advantage_i_lst = []
            advantage_i = np.array([0])
            for delta_i_t in delta_intr[::-1]:
                advantage_i = self.gamma * self.lmbda * advantage_i + delta_i_t
                advantage_i_lst.append(advantage_i)
            advantage_i_lst.reverse()
            advantage_i = torch.tensor(advantage_i_lst, dtype=torch.float, device=model.device)

            advantage = advantage_e + advantage_i * self.ratio
            data_with_adv.append((s, a, m, r, s_prime, done_mask, prob, need_move, td_target, td_intr_r, advantage))

        for i in range(self.K_epoch):
            # three times update for one sample
            for mini_batch in data_with_adv:
                s, a, m, r, s_prime, done_mask, prob, need_move, td_target, td_intr_r, advantage = mini_batch
                pi, pi_move, v, v_i, _ = model(s)
                pi_prime, pi_m_prime, v_prime, _, _ = model(s_prime)
                rnd_loss = center_rnd_model(s_prime)
                rnd_loss = rnd_loss / std * self.ratio

                pi_a = pi.gather(2, a)
                pi_m = pi_move.gather(2, m)
                pi_am = pi_a * (1 - need_move + need_move * pi_m)
                ratio = torch.exp(torch.log(pi_am) - torch.log(prob))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                entropy = -torch.log(pi_am)
                move_entropy = -need_move * torch.log(pi_m)

                # actor-critic style PPO
                surr_loss = -torch.min(surr1, surr2)
                v_loss = F.smooth_l1_loss(v, td_target.detach())
                v_i_loss = F.smooth_l1_loss(v_i, td_intr_r)
                entropy_loss = -1 * self.entropy_coef * entropy
                critic_loss = v_loss + v_i_loss

                if i == 0:
                    print(
                        "surr_loss:{}, ve_loss:{}, vi_loss:{}, entropy:{}, rnd_loss:{}".format(surr_loss.mean(), v_loss,
                                                                                               v_i_loss, entropy_loss.mean(),
                                                                                               rnd_loss))
                loss = surr_loss + critic_loss + entropy_loss.mean()
                loss = loss.mean()

                # Update policy network
                model.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                nn.utils.clip_grad_norm_(center_rnd_model.parameters(), self.grad_clip)
                model.optimizer.step()

                # Update RND network
                center_rnd_model.optimizer.zero_grad()
                rnd_loss.backward()
                center_rnd_model.optimizer.step()

                tot_loss_lst.append(loss.item())
                pi_loss_lst.append(surr_loss.mean().item())
                v_loss_lst.append(v_loss.item())
                v_i_loss_lst.append(v_i_loss.item())
                rnd_loss_lst.append(rnd_loss.item())
                entropy_lst.append(entropy.mean().item())
                n_need_move = torch.sum(need_move).item()
                if n_need_move == 0:
                    move_entropy_lst.append(0)
                else:
                    move_entropy_lst.append((torch.sum(move_entropy) / n_need_move).item())
        return np.mean(tot_loss_lst), np.mean(pi_loss_lst), np.mean(v_loss_lst), np.mean(v_i_loss_lst), np.mean(
            rnd_loss_lst), np.mean(entropy_lst), np.mean(move_entropy_lst)
