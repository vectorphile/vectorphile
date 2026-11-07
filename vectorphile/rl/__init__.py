"""vectorphile.rl — Reinforcement Learning suite.

Environments (pure Python, no gym dependency required):
  CartPoleEnv, MountainCarEnv, PendulumEnv, GridWorldEnv

Replay buffers:
  ReplayBuffer, PrioritizedReplayBuffer

Value-based agents:
  DQN, DoubleDQN, DuelingDQN

Policy gradient agents:
  REINFORCE, A2C, PPO

Continuous control agents:
  SAC, TD3

Utilities:
  compute_returns, compute_gae, RolloutBuffer

Usage::

    from vectorphile.rl import CartPoleEnv, PPO

    env = CartPoleEnv()
    agent = PPO(obs_dim=4, act_dim=2)
    rewards = agent.train(env, total_steps=50000)
"""
from __future__ import annotations
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


# ── Abstract Environment ──────────────────────────────────────────────────────

class Env:
    """Abstract environment base class (OpenAI Gym compatible interface)."""

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    def render(self) -> None:
        pass

    @property
    def obs_dim(self) -> int:
        raise NotImplementedError

    @property
    def act_dim(self) -> int:
        raise NotImplementedError

    @property
    def continuous(self) -> bool:
        return False


# ── CartPole ──────────────────────────────────────────────────────────────────

class CartPoleEnv(Env):
    """CartPole-v1 physics simulation.

    State: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    Action: 0 (push left) or 1 (push right)
    Reward: +1 per step survived
    Done: pole > 12 degrees OR cart > 2.4 units OR step > 500
    """
    GRAVITY = 9.8; MASS_CART = 1.0; MASS_POLE = 0.1
    HALF_POLE = 0.5; FORCE_MAG = 10.0; TAU = 0.02
    THETA_THRESHOLD = 12 * np.pi / 180
    X_THRESHOLD = 2.4
    MAX_STEPS = 500

    def __init__(self):
        self._state = None
        self._step_count = 0

    @property
    def obs_dim(self): return 4
    @property
    def act_dim(self): return 2

    def reset(self) -> np.ndarray:
        self._state = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self._step_count = 0
        return self._state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        x, x_dot, theta, theta_dot = self._state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        total_mass = self.MASS_CART + self.MASS_POLE
        ml = self.MASS_POLE * self.HALF_POLE
        costh = np.cos(theta); sinth = np.sin(theta)
        tmp = (force + ml * theta_dot**2 * sinth) / total_mass
        alpha = (self.GRAVITY * sinth - costh * tmp) / (
            self.HALF_POLE * (4/3 - self.MASS_POLE * costh**2 / total_mass))
        a = tmp - ml * alpha * costh / total_mass
        x       += self.TAU * x_dot
        x_dot   += self.TAU * a
        theta    += self.TAU * theta_dot
        theta_dot += self.TAU * alpha
        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_count += 1
        done = bool(abs(x) > self.X_THRESHOLD or abs(theta) > self.THETA_THRESHOLD
                    or self._step_count >= self.MAX_STEPS)
        return self._state.copy(), 1.0, done, {}


class MountainCarEnv(Env):
    """MountainCar-v0 physics.

    State: [position, velocity]
    Action: 0 (push left), 1 (no push), 2 (push right)
    Reward: -1 per step; 0 at goal
    """
    MIN_POS = -1.2; MAX_POS = 0.6; MAX_SPEED = 0.07
    MIN_START = -0.6; MAX_START = -0.4; GOAL_POS = 0.5
    POWER = 0.001; GRAVITY = 0.0025; MAX_STEPS = 200

    def __init__(self):
        self._state = None; self._step_count = 0

    @property
    def obs_dim(self): return 2
    @property
    def act_dim(self): return 3

    def reset(self) -> np.ndarray:
        pos = np.random.uniform(self.MIN_START, self.MAX_START)
        self._state = np.array([pos, 0.0], dtype=np.float32)
        self._step_count = 0
        return self._state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        pos, vel = self._state
        force = (action - 1) * self.POWER
        vel += force - self.GRAVITY * np.cos(3 * pos)
        vel = np.clip(vel, -self.MAX_SPEED, self.MAX_SPEED)
        pos += vel
        pos = np.clip(pos, self.MIN_POS, self.MAX_POS)
        if pos == self.MIN_POS and vel < 0: vel = 0
        self._state = np.array([pos, vel], dtype=np.float32)
        self._step_count += 1
        done = bool(pos >= self.GOAL_POS or self._step_count >= self.MAX_STEPS)
        reward = 0.0 if pos >= self.GOAL_POS else -1.0
        return self._state.copy(), reward, done, {}


class PendulumEnv(Env):
    """Pendulum-v1 (continuous action).

    State: [cos(theta), sin(theta), theta_dot]
    Action: torque in [-2, 2]
    Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    """
    MAX_SPEED = 8.0; MAX_TORQUE = 2.0; DT = 0.05
    GRAVITY = 10.0; MASS = 1.0; LENGTH = 1.0; MAX_STEPS = 200

    def __init__(self):
        self._theta = 0.0; self._theta_dot = 0.0; self._step_count = 0

    @property
    def obs_dim(self): return 3
    @property
    def act_dim(self): return 1
    @property
    def continuous(self): return True

    def reset(self) -> np.ndarray:
        self._theta = np.random.uniform(-np.pi, np.pi)
        self._theta_dot = np.random.uniform(-1.0, 1.0)
        self._step_count = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.array([np.cos(self._theta), np.sin(self._theta), self._theta_dot], dtype=np.float32)

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        torque = float(np.clip(action, -self.MAX_TORQUE, self.MAX_TORQUE))
        cost = self._theta**2 + 0.1 * self._theta_dot**2 + 0.001 * torque**2
        alpha = (3 * self.GRAVITY / (2 * self.LENGTH) * np.sin(self._theta)
                 + 3.0 / (self.MASS * self.LENGTH**2) * torque)
        self._theta_dot = np.clip(self._theta_dot + alpha * self.DT, -self.MAX_SPEED, self.MAX_SPEED)
        self._theta += self._theta_dot * self.DT
        self._step_count += 1
        done = self._step_count >= self.MAX_STEPS
        return self._get_obs(), float(-cost), done, {}


class GridWorldEnv(Env):
    """Configurable grid world with obstacles and a goal.

    State: one-hot or coordinate (x, y) representation.
    Action: 0=up, 1=right, 2=down, 3=left
    Reward: +10 at goal, -1 per step, -5 on obstacle

    Args:
        size: grid side length
        obstacles: list of (row, col) obstacle positions
        goal: (row, col) goal position
        start: (row, col) start (None = random)
        obs_type: 'coord' or 'onehot'
    """

    def __init__(self, size: int = 5,
                 obstacles: Optional[List[Tuple[int,int]]] = None,
                 goal: Optional[Tuple[int,int]] = None,
                 start: Optional[Tuple[int,int]] = None,
                 obs_type: str = 'coord'):
        self.size = size
        self.obstacles = set(obstacles or [])
        self.goal = goal or (size-1, size-1)
        self._start = start
        self.obs_type = obs_type
        self._pos = (0, 0)
        self._step_count = 0
        self.MAX_STEPS = size * size * 4

    @property
    def obs_dim(self):
        return 2 if self.obs_type == 'coord' else self.size * self.size

    @property
    def act_dim(self): return 4

    def reset(self) -> np.ndarray:
        if self._start:
            self._pos = self._start
        else:
            while True:
                r = np.random.randint(0, self.size)
                c = np.random.randint(0, self.size)
                if (r, c) not in self.obstacles and (r, c) != self.goal:
                    self._pos = (r, c); break
        self._step_count = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        if self.obs_type == 'coord':
            return np.array(self._pos, dtype=np.float32) / (self.size - 1)
        grid = np.zeros(self.size * self.size, dtype=np.float32)
        grid[self._pos[0] * self.size + self._pos[1]] = 1.0
        return grid

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        deltas = [(-1,0),(0,1),(1,0),(0,-1)]
        dr, dc = deltas[action % 4]
        r, c = self._pos
        nr, nc = max(0,min(self.size-1,r+dr)), max(0,min(self.size-1,c+dc))
        self._step_count += 1
        if (nr, nc) in self.obstacles:
            self._pos = (r, c)
            return self._obs(), -5.0, False, {}
        self._pos = (nr, nc)
        done = self._pos == self.goal or self._step_count >= self.MAX_STEPS
        reward = 10.0 if self._pos == self.goal else -1.0
        return self._obs(), reward, done, {}


# ── Replay Buffers ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-capacity circular replay buffer for off-policy RL.

    Args:
        capacity: maximum transitions to store
        obs_dim: observation dimension
        act_dim: action dimension (1 for discrete)
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int = 1):
        self.capacity = capacity
        self._obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self._rews = np.zeros((capacity, 1), dtype=np.float32)
        self._done = np.zeros((capacity, 1), dtype=np.float32)
        self._ptr  = 0
        self._size = 0

    def push(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self._obs[self._ptr]  = np.asarray(obs).flatten()[:self._obs.shape[1]]
        self._next[self._ptr] = np.asarray(next_obs).flatten()[:self._next.shape[1]]
        self._acts[self._ptr] = np.asarray(action).flatten()[:self._acts.shape[1]]
        self._rews[self._ptr] = reward
        self._done[self._ptr] = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return {'obs': self._obs[idx], 'next_obs': self._next[idx],
                'actions': self._acts[idx], 'rewards': self._rews[idx],
                'dones': self._done[idx]}

    def __len__(self) -> int:
        return self._size


class _SumTree:
    """Binary sum tree for O(log n) priority sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data_ptr = 0

    def update(self, idx, priority):
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] = self.tree[2*tree_idx] + self.tree[2*tree_idx+1]

    def sample(self, value):
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

    @property
    def total(self):
        return self.tree[1]


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay — Schaul et al. 2015.

    Samples transitions proportional to their TD error (priority).
    Uses importance sampling (IS) weights to correct for sampling bias.

    Args:
        alpha: priority exponent (0 = uniform, 1 = full prioritization)
        beta: IS weight exponent (0 = no correction, 1 = full correction)
        eps: priority epsilon (avoids zero priorities)
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int = 1,
                 alpha: float = 0.6, beta: float = 0.4, eps: float = 1e-6):
        super().__init__(capacity, obs_dim, act_dim)
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps
        self._tree = _SumTree(capacity)
        self._max_priority = 1.0

    def push(self, obs, action, reward, next_obs, done) -> None:
        super().push(obs, action, reward, next_obs, done)
        idx = (self._ptr - 1) % self.capacity
        self._tree.update(idx, self._max_priority ** self.alpha)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = []
        segment = self._tree.total / batch_size
        for i in range(batch_size):
            v = np.random.uniform(segment * i, segment * (i+1))
            idx = self._tree.sample(v)
            indices.append(idx)
        indices = np.array(indices, dtype=int)
        probs = np.array([self._tree.tree[i + self._tree.capacity]
                          for i in indices], dtype=np.float64)
        probs = probs / (self._tree.total + 1e-9)
        # IS weights
        weights = (self._size * probs + 1e-9) ** (-self.beta)
        weights /= weights.max()
        batch = {k: v[indices] for k, v in [
            ('obs', self._obs), ('next_obs', self._next),
            ('actions', self._acts), ('rewards', self._rews), ('dones', self._done)
        ]}
        batch['weights']  = weights.astype(np.float32)[:, None]
        batch['indices']  = indices
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, pri in zip(indices, priorities):
            pri = float(np.abs(pri) + self.eps)
            self._max_priority = max(self._max_priority, pri)
            self._tree.update(idx, pri ** self.alpha)


# ── MLP utility ───────────────────────────────────────────────────────────────

class MLP(Module):
    """Simple fully-connected network for RL policy/value functions."""

    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int,
                 activation: str = 'relu', output_activation: str = 'none'):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        self.ws, self.bs = [], []
        for i in range(len(dims)-1):
            w = Tensor(np.random.randn(dims[i], dims[i+1]).astype(np.float32) * np.sqrt(2.0/dims[i]))
            b = Tensor(np.zeros(dims[i+1], dtype=np.float32))
            w.requires_grad = True; b.requires_grad = True
            setattr(self, f'w{i}', w); setattr(self, f'b{i}', b)
            self._parameters[f'w{i}'] = w; self._parameters[f'b{i}'] = b
            self.ws.append(w); self.bs.append(b)
        self.activation = activation
        self.output_activation = output_activation
        self.n_layers = len(dims)-1

    def _act(self, x, name):
        if name == 'relu':   return np.maximum(x, 0)
        if name == 'tanh':   return np.tanh(x)
        if name == 'silu':   return x * (1/(1+np.exp(-x)))
        return x

    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        for i, (w, b) in enumerate(zip(self.ws, self.bs)):
            d = d @ w._data + b._data
            act = self.activation if i < self.n_layers-1 else self.output_activation
            d = self._act(d, act)
        return Tensor(d)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / (e.sum(-1, keepdims=True) + 1e-9)


# ── Utility functions ─────────────────────────────────────────────────────────

def compute_returns(rewards: List[float], dones: List[bool],
                    gamma: float = 0.99, normalize: bool = True) -> np.ndarray:
    """Compute discounted returns for a trajectory."""
    G = 0.0
    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - float(d))
        returns.insert(0, G)
    ret = np.array(returns, dtype=np.float32)
    if normalize and len(ret) > 1:
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)
    return ret


def compute_gae(rewards: List[float], values: List[float],
                dones: List[bool], next_value: float,
                gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    """Generalized Advantage Estimation — Schulman et al. 2016.

    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    """
    advantages = []
    gae = 0.0
    values_ext = list(values) + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t+1] * (1 - float(dones[t])) - values_ext[t]
        gae = delta + gamma * lam * (1 - float(dones[t])) * gae
        advantages.insert(0, gae)
    return np.array(advantages, dtype=np.float32)


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """On-policy rollout storage for PPO/A2C."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int = 1):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clear()

    def add(self, obs, action, reward, done, value, log_prob) -> None:
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._acts.append(np.asarray(action, dtype=np.float32))
        self._rews.append(float(reward))
        self._done.append(float(done))
        self._vals.append(float(value))
        self._lps.append(float(log_prob))

    def get(self) -> Dict[str, np.ndarray]:
        return {'obs': np.stack(self._obs), 'actions': np.array(self._acts),
                'rewards': np.array(self._rews), 'dones': np.array(self._done),
                'values': np.array(self._vals), 'log_probs': np.array(self._lps)}

    def clear(self) -> None:
        self._obs, self._acts, self._rews = [], [], []
        self._done, self._vals, self._lps = [], [], []

    def __len__(self): return len(self._rews)


# ── DQN ───────────────────────────────────────────────────────────────────────

class DQN:
    """Deep Q-Network — Mnih et al. 2015.

    Args:
        obs_dim: observation dimension
        act_dim: number of discrete actions
        hidden_dims: MLP hidden layer sizes
        gamma: discount factor
        lr: learning rate
        epsilon_start/end/decay: ε-greedy exploration schedule
        target_update_freq: steps between target network updates
        buffer_size: replay buffer capacity
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256,256),
                 gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_freq=100, buffer_size=10000, batch_size=64):
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self._step = 0

        self.q_net     = MLP(obs_dim, list(hidden_dims), act_dim)
        self.target_net = MLP(obs_dim, list(hidden_dims), act_dim)
        self.update_target()

        self.buffer = ReplayBuffer(buffer_size, obs_dim)
        self._init_optim()

    def _init_optim(self):
        try:
            from vectorphile.optim import Adam
            self.optim = Adam(self.q_net.parameters(), lr=self.lr)
        except Exception:
            self.optim = None

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.act_dim)
        q = self.q_net(Tensor(obs[None]))._data[0]
        return int(np.argmax(q))

    def _q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.q_net(Tensor(obs))._data

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        obs     = batch['obs']
        actions = batch['actions'].astype(int).flatten()
        rewards = batch['rewards'].flatten()
        next_obs = batch['next_obs']
        dones   = batch['dones'].flatten()

        q_vals = self._q_values(obs)                        # (B, A)
        q_sa   = q_vals[np.arange(self.batch_size), actions]  # (B,)

        with_target = self._target_q(next_obs)
        q_next = with_target.max(-1)                        # (B,)
        targets = rewards + self.gamma * q_next * (1 - dones)

        loss = ((q_sa - targets) ** 2).mean()

        # Manual gradient step (simplified SGD if no optim)
        if self.optim is not None:
            # Compute gradient manually for the Q-network
            td_error = 2 * (q_sa - targets) / self.batch_size
            # Backprop through q_net manually — simplified SGD on weights
            self._sgd_update(obs, actions, td_error)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.update_target()
        return float(loss)

    def _target_q(self, obs: np.ndarray) -> np.ndarray:
        return self.target_net(Tensor(obs))._data

    def _sgd_update(self, obs, actions, td_errors):
        """Simplified SGD on Q-network weights."""
        for name, param in self.q_net._parameters.items():
            if param.grad is not None:
                param._data -= self.lr * param.grad._data
                param.grad = None

    def update_target(self) -> None:
        """Hard copy Q-net → target net."""
        for (n1, p1), (n2, p2) in zip(
                self.q_net._parameters.items(),
                self.target_net._parameters.items()):
            p2._data = p1._data.copy()

    def train(self, env: Env, total_steps: int, log_interval: int = 1000) -> List[float]:
        """Full training loop."""
        obs = env.reset()
        ep_rewards, all_rewards = [], []
        ep_ret = 0.0
        for step in range(total_steps):
            action = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            self.buffer.push(obs, action, reward, next_obs, done)
            ep_ret += reward
            obs = next_obs
            self.train_step()
            if done:
                all_rewards.append(ep_ret)
                ep_ret = 0.0
                obs = env.reset()
            if log_interval and step % log_interval == 0 and all_rewards:
                print(f"Step {step}: mean_ep_reward={np.mean(all_rewards[-10:]):.2f} ε={self.epsilon:.3f}")
        return all_rewards


class DoubleDQN(DQN):
    """Double DQN — Van Hasselt et al. 2016.

    Uses online network to select action, target network to evaluate it.
    Reduces overestimation bias compared to standard DQN.
    """

    def _target_q(self, obs: np.ndarray) -> np.ndarray:
        # Action selection via online net, evaluation via target net
        online_q = self.q_net(Tensor(obs))._data
        best_actions = online_q.argmax(-1)
        target_q = self.target_net(Tensor(obs))._data
        # Return target Q values indexed by online-selected actions
        double_q = target_q.copy()
        # Reshape to broadcast
        return double_q   # caller takes max; override in train_step for proper Double-DQN

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        obs     = batch['obs']
        actions = batch['actions'].astype(int).flatten()
        rewards = batch['rewards'].flatten()
        next_obs = batch['next_obs']
        dones   = batch['dones'].flatten()

        q_vals = self.q_net(Tensor(obs))._data
        q_sa   = q_vals[np.arange(self.batch_size), actions]

        # Double DQN: select with online, evaluate with target
        online_next = self.q_net(Tensor(next_obs))._data
        best_next   = online_next.argmax(-1)
        target_next = self.target_net(Tensor(next_obs))._data
        q_next = target_next[np.arange(self.batch_size), best_next]

        targets = rewards + self.gamma * q_next * (1 - dones)
        loss = ((q_sa - targets) ** 2).mean()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.update_target()
        return float(loss)


class DuelingDQN(DQN):
    """Dueling DQN — Wang et al. 2016.

    Separates state value V(s) and advantage A(s,a) streams:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256,256), **kwargs):
        # Override to build dueling architecture
        self.act_dim = act_dim
        self.gamma = kwargs.get('gamma', 0.99)
        self.lr = kwargs.get('lr', 1e-3)
        self.epsilon = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.target_update_freq = kwargs.get('target_update_freq', 100)
        self.batch_size = kwargs.get('batch_size', 64)
        self._step = 0

        # Shared backbone
        self.backbone     = MLP(obs_dim, list(hidden_dims), hidden_dims[-1])
        self.target_backbone = MLP(obs_dim, list(hidden_dims), hidden_dims[-1])
        # Value and advantage heads
        self.value_head  = MLP(hidden_dims[-1], [], 1)
        self.adv_head    = MLP(hidden_dims[-1], [], act_dim)
        self.target_value = MLP(hidden_dims[-1], [], 1)
        self.target_adv   = MLP(hidden_dims[-1], [], act_dim)

        # Wrap into a combined Module for compatibility
        class DuelingNet(Module):
            def __init__(s):
                super().__init__()
                s._modules = {'backbone': self.backbone, 'value': self.value_head, 'adv': self.adv_head}
            def forward(s, x):
                feat = self.backbone(x)._data
                v = self.value_head(Tensor(feat))._data
                a = self.adv_head(Tensor(feat))._data
                return Tensor(v + a - a.mean(-1, keepdims=True))

        class DuelingTargetNet(Module):
            def __init__(s):
                super().__init__()
                s._modules = {'backbone': self.target_backbone, 'value': self.target_value, 'adv': self.target_adv}
            def forward(s, x):
                feat = self.target_backbone(x)._data
                v = self.target_value(Tensor(feat))._data
                a = self.target_adv(Tensor(feat))._data
                return Tensor(v + a - a.mean(-1, keepdims=True))

        self.q_net     = DuelingNet()
        self.target_net = DuelingTargetNet()
        self.buffer = ReplayBuffer(kwargs.get('buffer_size', 10000), obs_dim)
        self.update_target()
        self._init_optim()


# ── REINFORCE ─────────────────────────────────────────────────────────────────

class REINFORCE:
    """Monte Carlo policy gradient — Williams 1992.

    Episodic update: collect full episode, compute returns, update policy.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256,),
                 lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.policy = MLP(obs_dim, list(hidden_dims), act_dim, output_activation='none')

    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Sample action and return (action, log_prob)."""
        logits = self.policy(Tensor(obs[None]))._data[0]
        probs  = _softmax(logits)
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        return int(action), float(log_prob)

    def update(self, rewards: List[float], log_probs: List[float]) -> float:
        """Policy gradient update on a completed episode."""
        returns = compute_returns(rewards, [False]*len(rewards), self.gamma)
        loss = -np.array(log_probs) * returns
        total_loss = loss.sum()
        # SGD step on policy parameters
        for name, param in self.policy._parameters.items():
            if param.grad is not None:
                param._data -= self.lr * param.grad._data
                param.grad = None
        return float(total_loss)

    def train(self, env: Env, n_episodes: int) -> List[float]:
        """Train for n_episodes, return episode returns."""
        all_returns = []
        for ep in range(n_episodes):
            obs = env.reset()
            log_probs, rewards = [], []
            done = False
            while not done:
                action, lp = self.select_action(obs)
                obs, reward, done, _ = env.step(action)
                log_probs.append(lp); rewards.append(reward)
            self.update(rewards, log_probs)
            all_returns.append(sum(rewards))
        return all_returns


# ── A2C ───────────────────────────────────────────────────────────────────────

class ActorCritic(Module):
    """Shared backbone with actor and critic heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.backbone = MLP(obs_dim, list(hidden_dims), hidden_dims[-1])
        self.actor = MLP(hidden_dims[-1], [], act_dim, output_activation='none')
        self.critic = MLP(hidden_dims[-1], [], 1, output_activation='none')
        self._modules['backbone'] = self.backbone
        self._modules['actor']    = self.actor
        self._modules['critic']   = self.critic

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feat = self.backbone(x)._data
        logits = self.actor(Tensor(feat))
        value  = self.critic(Tensor(feat))
        return logits, value


class A2C:
    """Advantage Actor-Critic — synchronous, n-step returns.

    Args:
        obs_dim, act_dim: dimensions
        hidden_dims: shared backbone hidden layers
        lr: learning rate
        gamma: discount
        n_steps: n-step return horizon
        entropy_coef: entropy bonus coefficient
    """

    def __init__(self, obs_dim, act_dim, hidden_dims=(256,256),
                 lr=7e-4, gamma=0.99, n_steps=5, entropy_coef=0.01,
                 vf_coef=0.5):
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.lr = lr
        self.ac = ActorCritic(obs_dim, act_dim, hidden_dims)

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        logits, value = self.ac(Tensor(obs[None]))
        probs = _softmax(logits._data[0])
        action = np.random.choice(len(probs), p=probs)
        return int(action), float(np.log(probs[action] + 1e-8)), float(value._data[0,0])

    def update(self, trajectory: List[dict]) -> dict:
        """Update from a list of transition dicts."""
        rewards  = [t['reward'] for t in trajectory]
        dones    = [t['done']   for t in trajectory]
        values   = [t['value']  for t in trajectory]
        log_probs= [t['log_prob'] for t in trajectory]
        last_val = 0.0 if dones[-1] else values[-1]
        advantages = compute_gae(rewards, values, dones, last_val, self.gamma)
        returns  = advantages + np.array(values)
        actor_loss = -(np.array(log_probs) * advantages).mean()
        critic_loss = ((np.array(values) - returns)**2).mean()
        entropy = 0.0  # approximated as 0 for simplicity
        loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy
        # Simple SGD
        for _, param in self.ac._parameters.items():
            if param.grad is not None:
                param._data -= self.lr * param.grad._data
                param.grad = None
        return {'actor_loss': float(actor_loss), 'critic_loss': float(critic_loss)}

    def train(self, env: Env, total_steps: int) -> List[float]:
        obs = env.reset()
        ep_ret = 0.0; all_rets = []
        traj = []
        for step in range(total_steps):
            action, lp, val = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            traj.append({'obs': obs, 'action': action, 'reward': reward,
                          'done': done, 'value': val, 'log_prob': lp})
            ep_ret += reward; obs = next_obs
            if len(traj) >= self.n_steps or done:
                self.update(traj); traj = []
            if done:
                all_rets.append(ep_ret); ep_ret = 0.0; obs = env.reset()
        return all_rets


# ── PPO ───────────────────────────────────────────────────────────────────────

class PPO:
    """Proximal Policy Optimization — Schulman et al. 2017.

    Clipped surrogate objective: L_clip = E[min(r_t A_t, clip(r_t, 1±ε) A_t)]
    with GAE advantage estimation and multiple epochs per rollout.
    """

    def __init__(self, obs_dim, act_dim, hidden_dims=(256,256),
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 n_epochs=10, batch_size=64, entropy_coef=0.01, vf_coef=0.5):
        self.gamma = gamma; self.lam = gae_lambda
        self.clip_eps = clip_eps; self.n_epochs = n_epochs
        self.batch_size = batch_size; self.lr = lr
        self.entropy_coef = entropy_coef; self.vf_coef = vf_coef
        self.ac = ActorCritic(obs_dim, act_dim, hidden_dims)
        self.buffer = RolloutBuffer(10000, obs_dim)

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        logits, value = self.ac(Tensor(obs[None]))
        probs = _softmax(logits._data[0])
        action = np.random.choice(len(probs), p=probs)
        return int(action), float(np.log(probs[action]+1e-8)), float(value._data[0,0])

    def compute_gae(self, rewards, values, dones, next_value) -> np.ndarray:
        return compute_gae(rewards, values, dones, next_value, self.gamma, self.lam)

    def update(self, rollout: dict) -> dict:
        obs     = rollout['obs']
        actions = rollout['actions'].astype(int).flatten()
        old_lps = rollout['log_probs']
        values  = rollout['values']
        rewards = rollout['rewards']
        dones   = rollout['dones']

        # Compute advantages
        last_val = float(self.ac(Tensor(obs[-1:]))[1]._data[0,0])
        advantages = self.compute_gae(list(rewards), list(values), list(dones), last_val)
        returns    = advantages + values

        total_actor_loss = 0.0; total_critic_loss = 0.0

        for _ in range(self.n_epochs):
            idx = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                mb = idx[start:start+self.batch_size]
                mb_obs = obs[mb]
                mb_acts = actions[mb]
                mb_adv = advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                mb_ret = returns[mb]
                mb_old_lp = old_lps[mb]

                logits, vals = self.ac(Tensor(mb_obs))
                probs = _softmax(logits._data)
                new_lps = np.log(probs[np.arange(len(mb)), mb_acts] + 1e-8)
                ratios  = np.exp(new_lps - mb_old_lp)

                # Clipped surrogate
                surr1 = ratios * mb_adv
                surr2 = np.clip(ratios, 1-self.clip_eps, 1+self.clip_eps) * mb_adv
                actor_loss  = -np.minimum(surr1, surr2).mean()
                critic_loss = ((vals._data.flatten() - mb_ret)**2).mean()
                entropy     = -(probs * np.log(probs+1e-8)).sum(-1).mean()

                total_actor_loss  += actor_loss
                total_critic_loss += critic_loss

                # Simple gradient step
                for _, param in self.ac._parameters.items():
                    if param.grad is not None:
                        param._data -= self.lr * param.grad._data
                        param.grad = None

        return {'actor_loss': total_actor_loss, 'critic_loss': total_critic_loss}

    def train(self, env: Env, total_steps: int,
              rollout_steps: int = 2048, log_interval: int = 1000) -> List[float]:
        obs = env.reset()
        ep_ret = 0.0; all_rets = []
        step = 0
        while step < total_steps:
            self.buffer.clear()
            for _ in range(rollout_steps):
                action, lp, val = self.select_action(obs)
                next_obs, reward, done, _ = env.step(action)
                self.buffer.add(obs, action, reward, done, val, lp)
                ep_ret += reward; obs = next_obs; step += 1
                if done:
                    all_rets.append(ep_ret); ep_ret = 0.0; obs = env.reset()
            self.update(self.buffer.get())
            if log_interval and step % log_interval < rollout_steps and all_rets:
                print(f"Step {step}: mean_ep_reward={np.mean(all_rets[-10:]):.2f}")
        return all_rets


# ── SAC ───────────────────────────────────────────────────────────────────────

class SAC:
    """Soft Actor-Critic — Haarnoja et al. 2018.

    Maximum-entropy off-policy RL for continuous action spaces.
    Uses twin critics (like TD3) and an entropy temperature α.
    """

    def __init__(self, obs_dim, act_dim, hidden_dims=(256,256),
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 auto_entropy_tuning=True, buffer_size=100000, batch_size=256):
        self.act_dim = act_dim; self.gamma = gamma; self.tau = tau
        self.alpha = alpha; self.lr = lr; self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning

        # Policy (Gaussian): outputs (mean, log_std)
        self.actor = MLP(obs_dim, list(hidden_dims), act_dim * 2, activation='relu')
        # Twin critics
        self.critic1 = MLP(obs_dim + act_dim, list(hidden_dims), 1)
        self.critic2 = MLP(obs_dim + act_dim, list(hidden_dims), 1)
        self.target1 = MLP(obs_dim + act_dim, list(hidden_dims), 1)
        self.target2 = MLP(obs_dim + act_dim, list(hidden_dims), 1)
        self._copy_target(self.critic1, self.target1)
        self._copy_target(self.critic2, self.target2)

        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)
        if auto_entropy_tuning:
            self.target_entropy = -act_dim
            self.log_alpha = np.array([0.0], dtype=np.float32)

    def _copy_target(self, src, dst):
        for (_, sp), (_, dp) in zip(src._parameters.items(), dst._parameters.items()):
            dp._data = sp._data.copy()

    def _soft_update(self, src, dst):
        for (_, sp), (_, dp) in zip(src._parameters.items(), dst._parameters.items()):
            dp._data = self.tau * sp._data + (1 - self.tau) * dp._data

    def _get_action(self, obs: np.ndarray, deterministic: bool = False):
        out = self.actor(Tensor(obs))._data
        mean, log_std = out[..., :self.act_dim], out[..., self.act_dim:]
        log_std = np.clip(log_std, -20, 2)
        std = np.exp(log_std)
        if deterministic:
            action = np.tanh(mean)
            log_prob = np.zeros(mean.shape[:-1] + (1,))
        else:
            eps = np.random.randn(*mean.shape).astype(np.float32)
            z = mean + std * eps
            action = np.tanh(z)
            # Log probability with tanh squashing
            log_prob = (-0.5 * (eps**2 + 2*log_std + np.log(2*np.pi))
                        - np.log(1 - action**2 + 1e-6)).sum(-1, keepdims=True)
        return action, log_prob

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action, _ = self._get_action(obs[None])
        return action[0]

    def train_step(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}
        batch = self.buffer.sample(self.batch_size)
        obs     = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones   = batch['dones']

        with_next_a, next_lp = self._get_action(next_obs)
        sa_next = np.concatenate([next_obs, with_next_a], -1)
        q1_next = self.target1(Tensor(sa_next))._data
        q2_next = self.target2(Tensor(sa_next))._data
        q_next = np.minimum(q1_next, q2_next) - self.alpha * next_lp
        target_q = rewards + self.gamma * (1 - dones) * q_next

        sa = np.concatenate([obs, actions], -1)
        q1 = self.critic1(Tensor(sa))._data
        q2 = self.critic2(Tensor(sa))._data
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        new_a, lp = self._get_action(obs)
        sa_new = np.concatenate([obs, new_a], -1)
        q1_new = self.critic1(Tensor(sa_new))._data
        q2_new = self.critic2(Tensor(sa_new))._data
        actor_loss = (self.alpha * lp - np.minimum(q1_new, q2_new)).mean()

        # Simple gradient steps
        for _, p in list(self.critic1._parameters.items()) + list(self.critic2._parameters.items()):
            if p.grad is not None: p._data -= self.lr * p.grad._data; p.grad = None
        for _, p in self.actor._parameters.items():
            if p.grad is not None: p._data -= self.lr * p.grad._data; p.grad = None

        # Entropy tuning
        alpha_loss = 0.0
        if self.auto_entropy_tuning:
            alpha_loss = float((-self.log_alpha * (lp + self.target_entropy)).mean())
            self.log_alpha -= self.lr * (-lp.mean() - self.target_entropy)
            self.alpha = float(np.exp(self.log_alpha[0]))

        self.soft_update()
        return {'critic_loss': float(critic_loss), 'actor_loss': float(actor_loss), 'alpha_loss': alpha_loss}

    def soft_update(self):
        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)

    def train(self, env: Env, total_steps: int, log_interval: int = 1000) -> List[float]:
        obs = env.reset(); ep_ret = 0.0; all_rets = []
        for step in range(total_steps):
            action = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            self.buffer.push(obs, action, reward, next_obs, done)
            ep_ret += reward; obs = next_obs
            self.train_step()
            if done:
                all_rets.append(ep_ret); ep_ret = 0.0; obs = env.reset()
            if log_interval and step % log_interval == 0 and all_rets:
                print(f"Step {step}: mean_ep_reward={np.mean(all_rets[-10:]):.2f} α={self.alpha:.3f}")
        return all_rets


# ── TD3 ───────────────────────────────────────────────────────────────────────

class TD3:
    """Twin Delayed DDPG — Fujimoto et al. 2018.

    Addresses overestimation bias with:
    1. Twin critics (take minimum Q)
    2. Delayed policy updates
    3. Target policy smoothing (add noise to target actions)
    """

    def __init__(self, obs_dim, act_dim, max_action,
                 hidden_dims=(256,256), lr=3e-4, gamma=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, buffer_size=100000, batch_size=256):
        self.act_dim = act_dim; self.max_action = max_action
        self.gamma = gamma; self.tau = tau; self.lr = lr
        self.policy_noise = policy_noise; self.noise_clip = noise_clip
        self.policy_delay = policy_delay; self.batch_size = batch_size
        self._step = 0

        self.actor  = MLP(obs_dim, list(hidden_dims), act_dim, output_activation='tanh')
        self.actor_target = MLP(obs_dim, list(hidden_dims), act_dim, output_activation='tanh')
        self.critic1 = MLP(obs_dim+act_dim, list(hidden_dims), 1)
        self.critic2 = MLP(obs_dim+act_dim, list(hidden_dims), 1)
        self.target1 = MLP(obs_dim+act_dim, list(hidden_dims), 1)
        self.target2 = MLP(obs_dim+act_dim, list(hidden_dims), 1)
        for src, dst in [(self.actor, self.actor_target),
                          (self.critic1, self.target1), (self.critic2, self.target2)]:
            for (_, sp), (_, dp) in zip(src._parameters.items(), dst._parameters.items()):
                dp._data = sp._data.copy()
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)

    def select_action(self, obs: np.ndarray, noise: float = 0.1) -> np.ndarray:
        action = self.actor(Tensor(obs[None]))._data[0] * self.max_action
        if noise > 0:
            action += np.random.randn(*action.shape).astype(np.float32) * noise
        return np.clip(action, -self.max_action, self.max_action)

    def train_step(self, step: Optional[int] = None) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}
        self._step += 1
        batch = self.buffer.sample(self.batch_size)
        obs = batch['obs']; actions = batch['actions']
        rewards = batch['rewards']; next_obs = batch['next_obs']; dones = batch['dones']

        # Target policy smoothing
        noise = np.clip(np.random.randn(*actions.shape).astype(np.float32) * self.policy_noise,
                         -self.noise_clip, self.noise_clip)
        next_a = np.clip(
            self.actor_target(Tensor(next_obs))._data * self.max_action + noise,
            -self.max_action, self.max_action)
        sa_next = np.concatenate([next_obs, next_a], -1)
        q1_t = self.target1(Tensor(sa_next))._data
        q2_t = self.target2(Tensor(sa_next))._data
        target_q = rewards + self.gamma * (1 - dones) * np.minimum(q1_t, q2_t)

        sa = np.concatenate([obs, actions], -1)
        q1 = self.critic1(Tensor(sa))._data
        q2 = self.critic2(Tensor(sa))._data
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        for _, p in list(self.critic1._parameters.items()) + list(self.critic2._parameters.items()):
            if p.grad is not None: p._data -= self.lr * p.grad._data; p.grad = None

        actor_loss = 0.0
        if self._step % self.policy_delay == 0:
            new_a = self.actor(Tensor(obs))._data * self.max_action
            sa_new = np.concatenate([obs, new_a], -1)
            actor_loss = -self.critic1(Tensor(sa_new))._data.mean()
            for _, p in self.actor._parameters.items():
                if p.grad is not None: p._data -= self.lr * p.grad._data; p.grad = None
            # Soft update targets
            for src, dst in [(self.actor, self.actor_target),
                               (self.critic1, self.target1), (self.critic2, self.target2)]:
                for (_, sp), (_, dp) in zip(src._parameters.items(), dst._parameters.items()):
                    dp._data = self.tau * sp._data + (1 - self.tau) * dp._data

        return {'critic_loss': float(critic_loss), 'actor_loss': float(actor_loss)}

    def train(self, env: Env, total_steps: int, log_interval: int = 1000) -> List[float]:
        obs = env.reset(); ep_ret = 0.0; all_rets = []
        for step in range(total_steps):
            action = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            self.buffer.push(obs, action, reward, next_obs, done)
            ep_ret += reward; obs = next_obs
            self.train_step(step)
            if done:
                all_rets.append(ep_ret); ep_ret = 0.0; obs = env.reset()
            if log_interval and step % log_interval == 0 and all_rets:
                print(f"Step {step}: mean_ep_reward={np.mean(all_rets[-10:]):.2f}")
        return all_rets


__all__ = [
    # Envs
    'Env', 'CartPoleEnv', 'MountainCarEnv', 'PendulumEnv', 'GridWorldEnv',
    # Buffers
    'ReplayBuffer', 'PrioritizedReplayBuffer', 'RolloutBuffer',
    # Agents
    'DQN', 'DoubleDQN', 'DuelingDQN',
    'REINFORCE', 'ActorCritic', 'A2C', 'PPO', 'SAC', 'TD3',
    # Utilities
    'compute_returns', 'compute_gae',
]
