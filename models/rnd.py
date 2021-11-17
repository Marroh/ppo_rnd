import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import numpy as np


class FeatureEncoder(nn.Module):
    def __init__(self, arg_dict, device=None):
        super(FeatureEncoder, self).__init__()
        self.device = None
        if device:
            self.device = device

        self.arg_dict = arg_dict
        out_feature_size = arg_dict["lstm_size"]

        self.fc_player = nn.Linear(arg_dict["feature_dims"]["player"], 64)
        self.fc_ball = nn.Linear(arg_dict["feature_dims"]["ball"], 64)
        self.fc_left = nn.Linear(arg_dict["feature_dims"]["left_team"], 48)
        self.fc_right = nn.Linear(arg_dict["feature_dims"]["right_team"], 48)
        self.fc_left_closest = nn.Linear(arg_dict["feature_dims"]["left_team_closest"], 48)
        self.fc_right_closest = nn.Linear(arg_dict["feature_dims"]["right_team_closest"], 48)

        self.conv1d_left = nn.Conv1d(48, 36, 1, stride=1)
        self.conv1d_right = nn.Conv1d(48, 36, 1, stride=1)
        self.fc_left2 = nn.Linear(36 * 10, 96)
        self.fc_right2 = nn.Linear(36 * 11, 96)
        self.fc_cat = nn.Linear(96 + 96 + 64 + 64 + 48 + 48, out_feature_size)

        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_left = nn.LayerNorm(48)
        self.norm_left2 = nn.LayerNorm(96)
        self.norm_left_closest = nn.LayerNorm(48)
        self.norm_right = nn.LayerNorm(48)
        self.norm_right2 = nn.LayerNorm(96)
        self.norm_right_closest = nn.LayerNorm(48)
        self.norm_cat = nn.LayerNorm(out_feature_size)

    def forward(self, state_dict):
        player_state = state_dict["player"]
        ball_state = state_dict["ball"]
        left_team_state = state_dict["left_team"]
        left_closest_state = state_dict["left_closest"]
        right_team_state = state_dict["right_team"]
        right_closest_state = state_dict["right_closest"]

        player_embed = self.norm_player(self.fc_player(player_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
        left_team_embed = self.norm_left(self.fc_left(left_team_state))  # horizon, batch, n, dim
        left_closest_embed = self.norm_left_closest(self.fc_left_closest(left_closest_state))
        right_team_embed = self.norm_right(self.fc_right(right_team_state))
        right_closest_embed = self.norm_right_closest(self.fc_right_closest(right_closest_state))

        # print("\nplayer_embed:{}\n ball_embed:{}\n left_team_embed:{}\n left_closest_embed:{}".format(
        #     player_embed.shape, ball_embed.shape, left_team_embed.shape, left_closest_embed.shape))

        [horizon, batch_size, n_player, dim] = left_team_embed.size()
        left_team_embed = left_team_embed.view(horizon * batch_size, n_player, dim).permute(0, 2,
                                                                                            1)  # horizon * batch, dim1, n
        # print("left team embed viewed:{}".format(left_team_embed.shape))
        left_team_embed = F.relu(self.conv1d_left(left_team_embed)).permute(0, 2, 1)  # horizon * batch, n, dim2
        # print("left team embed after conv1d:{}".format(left_team_embed.shape))
        left_team_embed = left_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size,
                                                                                 -1)  # horizon, batch, n * dim2
        left_team_embed = F.relu(self.norm_left2(self.fc_left2(left_team_embed)))
        # print("left team embed final:{}".format(left_team_embed.shape))

        right_team_embed = right_team_embed.view(horizon * batch_size, n_player + 1, dim).permute(0, 2,
                                                                                                  1)  # horizon * batch, dim1, n
        right_team_embed = F.relu(self.conv1d_right(right_team_embed)).permute(0, 2, 1)  # horizon * batch, n * dim2
        right_team_embed = right_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size, -1)
        right_team_embed = F.relu(self.norm_right2(self.fc_right2(right_team_embed)))

        cat = torch.cat(
            [player_embed, ball_embed, left_team_embed, right_team_embed, left_closest_embed, right_closest_embed], 2)
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        # print("cat:{}".format(cat.shape))
        return cat


class RNDModel(nn.Module):
    """
    In RND there are 2 networks:
    - Target Network: generates a constant output for a given state
    - Prediction network: tries to predict the target network's output
    """

    def __init__(self, arg_dict, output_size):
        # input size: 1*1*256
        super(RNDModel, self).__init__()

        self.input_size = 256
        self.output_size = output_size

        # Prediction network
        self.predictor = FeatureEncoder(arg_dict=arg_dict)

        # Target network
        self.target = FeatureEncoder(arg_dict=arg_dict)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-4)

    def forward(self, next_obs):
        with torch.no_grad():
            target_next_feature = self.target(next_obs)
        predict_next_feature = self.predictor(next_obs)

        intrinsic_reward = (target_next_feature - predict_next_feature).norm(p=2)
        # norm_intr_reward = (torch.sigmoid(intrinsic_reward/20)-0.5)*10  # map to [-5,5]
        return intrinsic_reward
