"""This modules creates a sac model in PyTorch."""
import copy

from dowel import logger, tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.torch.utils import np_to_torch, torch_to_np


class SAC(OffPolicyRLAlgorithm):
    """ A SAC Model in Torch.

    Soft Actor Critic (SAC) is an algorithm which optimizes a stochastic
    policy in an off-policy way, forming a bridge between stochastic policy
    optimization and DDPG-style approaches.
    A central feature of SAC is entropy regularization. The policy is trained
    to maximize a trade-off between expected return and entropy, a measure of
    randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more
    exploration, which can accelerate learning later on. It can also prevent
    the policy from prematurely converging to a bad local optimum.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf1,
                 qf2,
                 alpha,
                 replay_buffer,
                 target_entropy=None,
                 use_automatic_entropy_tuning=False,
                 discount=0.99,
                 max_path_length=None,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 rollout_batch_size=1,
                 exploration_strategy=None,
                 target_update_tau=1e-2,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 policy_weight_decay=0,
                 qf_weight_decay=0,
                 optimizer=torch.optim.Adam,
                 clip_pos_returns=False,
                 clip_return=np.inf,
                 max_action=None,
                 reward_scale=1.,
                 smooth_return=True,
                 input_include_goal=False):

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.alpha = alpha
        self.replay_buffer = replay_buffer
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.policy_weight_decay = policy_weight_decay
        self.qf_weight_decay = qf_weight_decay
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.evaluate = False
        self.input_include_goal = input_include_goal

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf=qf1,
                         n_train_steps=1,
                         n_epoch_cycles=1,
                         max_path_length=max_path_length,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         rollout_batch_size=rollout_batch_size,
                         exploration_strategy=exploration_strategy,
                         replay_buffer=replay_buffer,
                         use_target=True,
                         discount=discount,
                         reward_scale=reward_scale,
                         smooth_return=smooth_return)

        self.target_policy = copy.deepcopy(self.policy)
        # use 2 target q networks
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.policy_optimizer = optimizer(self.policy.parameters(),
                                          lr=self.policy_lr)
        self.qf1_optimizer = optimizer(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer(self.qf2.parameters(), lr=self.qf_lr)

        # automatic entropy coefficient tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()

        self.episode_rewards = []
        self.success_history = []

    # 0) update policy using updated min q function
    # 1) compute targets from Q functions
    # 2) update Q functions using optimizer
    # 3) query Q functons, take min of those functions
    def train_once(self, itr, paths):
        """
        """
        paths = self.process_samples(itr, paths)
        epoch = itr / self.n_epoch_cycles
        self.episode_rewards.extend([
            path for path, complete in zip(paths['undiscounted_returns'],
                                           paths['complete']) if complete
        ])
        self.success_history.extend([
            path for path, complete in zip(paths['success_history'],
                                           paths['complete']) if complete
        ])
        last_average_return = np.mean(self.episode_rewards)
        # add paths to replay buffer
        for train_itr in range(self.n_train_steps):
            # if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
            samples = self.replay_buffer.sample(self.buffer_batch_size)
            import ipdb; ipdb.set_trace()
            self.update_q_functions(itr, samples)
            import ipdb; ipdb.set_trace()
            self.optimize_policy(itr, samples)
            self.adjust_temperature(itr)
            self.update_targets()

        return last_average_return

    def update_q_functions(self, itr, samples):
        """ Update the q functions using the target q_functions.

        Args:
            itr (int) - current training iteration
            samples() - samples recovered from the replay buffer
        """
        obs = samples["observation"]
        actions = samples["action"]
        rewards = samples["reward"]
        next_obs = samples["next_observation"]

        with torch.no_grad():
            next_actions, _ = self.policy.get_actions(torch.Tensor(next_obs))
            next_ll = self.policy.log_likelihood(torch.Tensor(next_obs),
                                               torch.Tensor(next_actions))

        qfs = [self.qf1, self.qf2]
        target_qfs = [self.target_qf1, self.target_qf2]
        qf_optimizers = [self.qf1_optimizer, self.qf2_optimizer] 
        for target_qf, qf, qf_optimizer in zip(target_qfs, qfs, qf_optimizers):
            curr_q_val = qf(torch.Tensor(obs), torch.Tensor(actions)).flatten()
            with torch.no_grad():
                targ_out = target_qf(torch.Tensor(next_obs), torch.Tensor(next_actions)).flatten()
            bootstrapped_value = targ_out - (self.alpha * next_ll)
            bellman = torch.Tensor(rewards) + self.discount*(bootstrapped_value)
            q_objective = 0.5 * F.mse_loss(curr_q_val, bellman)
            qf_optimizer.zero_grad()
            q_objective.backward()
            qf_optimizer.step()

    def optimize_policy(self, itr, samples):
        """ Optimize the policy based on the policy objective from the sac paper.

        Args:
            itr (int) - current training iteration
            samples() - samples recovered from the replay buffer
        Returns:
            None
        """

        obs = samples["observation"]
        # use the forward function instead of the get action function
        # in order to make sure that policy is differentiated. 
        action_dists = self.policy(obs)
        actions = action_dists.rsample()
        with torch.no_grad():
            log_pi = self.policy.log_likelihood(obs,actions)
        with torch.no_grad():
            min_q = torch.min(self.qf1(obs, actions), self.qf2(obs, actions))

        policy_objective = ((self.alpha * log_pi) - min_q).mean()
        self.policy_optimizer.zero_grad()
        policy_objective.backwards()
        self.policy_optimizer.step()

    def adjust_temperature(self, itr):        
        pass

    def update_targets(self):
        """Update parameters in the target q-functions."""
        # update for target_qf1
        target_qfs = [self.target_qf1, self.target_qf2]
        qfs = [self.qf1, self.qf2]
        for target_qf, qf in zip(target_qfs, qfs):
                for t_param, param in zip(target_qf.parameters(),
                                            qf.parameters()):
                        t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                        param.data * self.tau)
