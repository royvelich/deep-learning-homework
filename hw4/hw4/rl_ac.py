import gym
import torch
import torch.nn as nn
import torch.nn.functional

from .rl_pg import PolicyAgent, TrainBatch, VanillaPolicyGradientLoss


class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()
        # TODO:
        #  Implement a dual-head neural net to approximate both the
        #  policy and value. You can have a common base part, or not.
        # ====== YOUR CODE: ======
        self.in_features = in_features
        self.out_actions = out_actions

        # TODO: Implement a simple neural net to approximate the policy.

        self.fc_actor = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_actions)
        )

        self.fc_critic = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(in_features, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU()
        #
        # )
        # self.fc_actor = nn.Linear(32, out_actions)
        # self.fc_critic = nn.Linear(32, 1)
        # ========================


    def forward(self, x):
        """
        :param x: Batch of states, shape (N,O) where N is batch size and O
        is the observation dimension (features).
        :return: A tuple of action values (N,A) and state values (N,1) where
        A is is the number of possible actions.
        """
        # TODO:
        #  Implement the forward pass.
        #  calculate both the action scores (policy) and the value of the
        #  given state.
        # ====== YOUR CODE: ======

        #x = self.fc.forward(x)

        action_scores = self.fc_actor.forward(x)
        state_values = self.fc_critic.forward(x)
        # ========================

        return action_scores, state_values

    @staticmethod
    def build_for_env(env: gym.Env, device='cpu', **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        # TODO: Implement according to docstring.
        # ====== YOUR CODE: ======
        net = AACPolicyNet(8, 4)
        # ========================
        return net.to(device)


class AACPolicyAgent(PolicyAgent):

    def current_action_distribution(self) -> torch.Tensor:
        # TODO: Generate the distribution as described above.
        # ====== YOUR CODE: ======
        m = nn.Softmax(dim=-1)
        scores, _ = self.p_net(self.curr_state)
        actions_proba = m(scores)
        # m = nn.Softmax(dim=0)
        # scores, _ = self.p_net(self.curr_state)
        # actions_proba = m(scores)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        action_scores, state_values = model_output

        # TODO: Calculate the policy loss loss_p, state-value loss loss_v and
        #  advantage vector per state.
        #  Use the helper functions in this class and its base.
        # ====== YOUR CODE: ======
        #print(state_values.shape)
        state_values_flat = state_values.flatten()
        #print(state_values_flat.shape)
        policy_weight = self._policy_weight(batch, state_values_flat)

        loss_v = self._value_loss(batch, state_values_flat)
        advantage = policy_weight.squeeze(-1)

        #loss_p = self._policy_loss(batch, action_scores, policy_weight)
        loss_p = self._policy_loss(batch, action_scores, advantage)
        # ========================
        loss_v *= self.delta
        loss_t = loss_p + loss_v
        return loss_t, dict(loss_p=loss_p.item(), loss_v=loss_v.item(),
                            adv_m=advantage.mean().item())

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO:
        #  Calculate the weight term of the AAC policy gradient (advantage).
        #  Notice that we don't want to backprop errors from the policy
        #  loss into the state-value network.
        # ====== YOUR CODE: ======
        #policy_weight = batch.q_vals
        #print(batch.q_vals[0])
        #print(state_values[0])
        advantage = (batch.q_vals - state_values.squeeze(-1))
        #print(advantage[0])
        # ========================
        return advantage

    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO: Calculate the state-value loss.
        # ====== YOUR CODE: ======
        #advantage_se = torch.pow(state_values.squeeze(-1) - batch.q_vals, 2)
        #loss_v = advantage_se.mean()
        loss = nn.MSELoss()
        loss_v = loss(state_values.squeeze(-1), batch.q_vals.float())
        # ========================
        return loss_v

