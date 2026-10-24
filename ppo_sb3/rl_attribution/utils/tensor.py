import numpy as np
import torch

def flatten_ob_act(ob: torch.Tensor, act: torch.Tensor):
    """Flatten and concatenate observation and action tensors."""
    return tuple(np.concatenate((ob.cpu().numpy().flatten(),
                                 act.cpu().numpy().flatten())))

def flatten_ob_act_prob_adv(ob: torch.Tensor, act: torch.Tensor,
                            prob: torch.Tensor, adv: torch.Tensor):
    """Flatten and concatenate observation, action, probability, and advantage tensors."""
    return tuple(np.concatenate((ob.cpu().numpy().flatten(),
                                 act.cpu().numpy().flatten(),
                                 prob.cpu().numpy().flatten(),
                                 adv.cpu().numpy().flatten())))
