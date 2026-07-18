"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math
import numpy as np
import torch

class CategoricalDiffusion(object):

  def __init__(self, T):
    # Diffusion steps
    self.T = T

    # Noise schedule
    b0 = 1e-4
    bT = 2e-2
    self.beta = np.linspace(b0, bT, T)
  
    self.alpha = 1.0 - self.beta
    self.alpha_bar = np.concatenate(([1.0], np.cumprod(self.alpha)))

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.array([[1, 0], [1, 0]], dtype=float).reshape((1, 2, 2))

    # One-way corruption:
    # 0 stays 0, while 1 can only be erased into 0.
    self.Qs = (1 - beta) * eye + beta * ones

    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    self.Q_bar = np.stack(Q_bar, axis=0)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0_onehot, t,x0_batch):
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device).reshape(t.shape[0],2,2)
    Q_bar = Q_bar[x0_batch]
    xt = torch.matmul(x0_onehot, Q_bar)
    return torch.bernoulli(xt[..., 1].clamp(0, 1))

  def keep_ratio(self, t):
    return np.clip(1.0 - (np.asarray(t, dtype=float) / float(self.T)), 0.0, 1.0)


class InferenceSchedule(object):
  def __init__(self, T=1000, inference_T=1000):
   
    self.T = T
    self.inference_T = inference_T

  def __call__(self, i):
    assert 0 <= i < self.inference_T
    t1 = self.T - int((float(i) / self.inference_T) * self.T)
    t1 = np.clip(t1, 1, self.T)

    t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
    t2 = np.clip(t2, 0, self.T - 1)
    return t1, t2
    
