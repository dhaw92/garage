"""
Torch argmax policy
"""
import numpy as np

from garage.torch.algos.core import PyTorchModule
import garage.torch.algos.pytorch_util as ptu
from garage.torch.policies.base import SerializablePolicy


class ArgmaxDiscretePolicy(PyTorchModule, SerializablePolicy):
    def __init__(self, qf):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), {}
