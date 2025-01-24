from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import HParam


class DropoutNet(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 128,
            last_layer_dim_vf: int = 128,
            hidden_size: int = 128,
            hidden_layers: int = 3
    ):
        super().__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_in = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.policy_hidden = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Dropout(0.2),) for _ in range(hidden_layers)]
        )
        self.policy_out = nn.Sequential(
            nn.Linear(hidden_size, self.latent_dim_pi), nn.ReLU(),
        )

        # Value network
        self.value_in = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.value_hidden = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size),
                           nn.ReLU(),
                           nn.Dropout(0.2),) for _ in range(hidden_layers)]
        )
        self.value_out = nn.Sequential(
            nn.Linear(hidden_size, self.latent_dim_vf), nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        x = self.policy_in(features)
        for layer in self.policy_hidden:
            x = layer(x)
        return self.policy_out(x)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        x = self.value_in(features)
        for layer in self.value_hidden:
            x = layer(x)
        return self.value_out(x)


class DropoutPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            hidden_size: int = 128,
            hidden_layers: int = 3,
            *args,
            **kwargs,
    ):
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        # Disable orthogonal initialization
        # kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DropoutNet(self.features_dim,
                                        hidden_size=self.hidden_size, hidden_layers=self.hidden_layers)


class ParamCallbackPPO(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "ent_coef": self.model.ent_coef,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "noise": self.model.env.get_attr("noise")[0],
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # TensorBoard will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
