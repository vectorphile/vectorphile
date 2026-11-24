"""vectorphile.rl — Reinforcement Learning suite (pure Python)."""
from __future__ import annotations
import math
import copy
import random as _rng
from typing import Dict, List, Optional, Tuple, Any
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gauss() -> float:
    return math.sqrt(-2*math.log(max(_rng.random(),1e-10)))*math.cos(2*math.pi*_rng.random())


def _randn(n: int) -> list: return [_gauss() for _ in range(n)]
def _zeros(n: int) -> list: return [0.0]*n
def _clip(v, lo, hi): return max(lo, min(hi, v))
def _clip_list(lst, lo, hi): return [_clip(v,lo,hi) for v in lst]
def _exp(x: float) -> float: return math.exp(max(-500.0, min(500.0, x)))
def _safe_exp(x: float) -> float: return _exp(x)
def _safe_log(x: float) -> float: return math.log(max(x, 1e-10))
def _tanh(x: float) -> float: return math.tanh(x)
def _sigmoid(x: float) -> float: return 1.0/(1.0+_exp(-x))
def _silu(x: float) -> float: return x*_sigmoid(x)
def _relu(x: float) -> float: return max(0.0, x)
def _mean_list(lst: list) -> float: return sum(lst)/max(len(lst),1)
def _std_list(lst: list, ddof=1) -> float:
    mu = _mean_list(lst); n = len(lst)
    return math.sqrt(sum((v-mu)**2 for v in lst)/max(n-ddof,1))

def _flat(x) -> list:
    if hasattr(x, '_data') and hasattr(x._data, 'to_flat_list'): return x._data.to_flat_list()
    if hasattr(x, 'to_flat_list'): return x.to_flat_list()
    if isinstance(x, (list, tuple)):
        out = []
        def _r(v):
            if isinstance(v, (list,tuple)):
                for i in v: _r(i)
            else: out.append(float(v))
        _r(x); return out
    return [float(x)]

def _make_tensor(flat, shape):
    from vectorphile._core.ndarray import NDArray, _c_strides
    import array as _arr
    buf = _arr.array('f', [float(v) for v in flat])
    nd = NDArray(buf, shape, _c_strides(shape), 'float32')
    return Tensor(nd)

def _mat_vec(W: list, x: list) -> list:
    """W: (out, in) as flat list, x: (in,) → (out,)"""
    rows, cols = len(W), len(x)
    return [sum(W[r][c]*x[c] for c in range(cols)) for r in range(rows)]

def _softmax_list(logits: list) -> list:
    mx = max(logits)
    e = [_exp(v-mx) for v in logits]; s = sum(e)+1e-9
    return [v/s for v in e]

def _obs_list(obs) -> list:
    if isinstance(obs, list): return obs
    if hasattr(obs, '__iter__'): return [float(v) for v in obs]
    return [float(obs)]


# ── Abstract Environment ──────────────────────────────────────────────────────

class Env:
    def reset(self) -> list: raise NotImplementedError
    def step(self, action) -> Tuple[list, float, bool, dict]: raise NotImplementedError
    def render(self) -> None: pass
    @property
    def obs_dim(self) -> int: raise NotImplementedError
    @property
    def act_dim(self) -> int: raise NotImplementedError
    @property
    def continuous(self) -> bool: return False


# ── CartPole ──────────────────────────────────────────────────────────────────

class CartPoleEnv(Env):
    GRAVITY=9.8; MASS_CART=1.0; MASS_POLE=0.1
    HALF_POLE=0.5; FORCE_MAG=10.0; TAU=0.02
    THETA_THRESHOLD=12*math.pi/180; X_THRESHOLD=2.4; MAX_STEPS=500

    def __init__(self): self._state=None; self._step_count=0
    @property
    def obs_dim(self): return 4
    @property
    def act_dim(self): return 2

    def reset(self) -> list:
        self._state = [_rng.uniform(-0.05,0.05) for _ in range(4)]
        self._step_count=0; return self._state[:]

    def step(self, action: int) -> Tuple[list, float, bool, dict]:
        x, x_dot, theta, theta_dot = self._state
        force = self.FORCE_MAG if action==1 else -self.FORCE_MAG
        total_mass = self.MASS_CART+self.MASS_POLE; ml = self.MASS_POLE*self.HALF_POLE
        costh = math.cos(theta); sinth = math.sin(theta)
        tmp = (force+ml*theta_dot**2*sinth)/total_mass
        alpha = (self.GRAVITY*sinth-costh*tmp)/(self.HALF_POLE*(4/3-self.MASS_POLE*costh**2/total_mass))
        a = tmp-ml*alpha*costh/total_mass
        x += self.TAU*x_dot; x_dot += self.TAU*a
        theta += self.TAU*theta_dot; theta_dot += self.TAU*alpha
        self._state=[x,x_dot,theta,theta_dot]; self._step_count+=1
        done = bool(abs(x)>self.X_THRESHOLD or abs(theta)>self.THETA_THRESHOLD or self._step_count>=self.MAX_STEPS)
        return self._state[:], 1.0, done, {}


class MountainCarEnv(Env):
    MIN_POS=-1.2; MAX_POS=0.6; MAX_SPEED=0.07
    MIN_START=-0.6; MAX_START=-0.4; GOAL_POS=0.5
    POWER=0.001; GRAVITY=0.0025; MAX_STEPS=200

    def __init__(self): self._state=None; self._step_count=0
    @property
    def obs_dim(self): return 2
    @property
    def act_dim(self): return 3

    def reset(self) -> list:
        pos = _rng.uniform(self.MIN_START, self.MAX_START)
        self._state=[pos,0.0]; self._step_count=0; return self._state[:]

    def step(self, action: int) -> Tuple[list, float, bool, dict]:
        pos, vel = self._state; force=(action-1)*self.POWER
        vel += force-self.GRAVITY*math.cos(3*pos)
        vel = _clip(vel,-self.MAX_SPEED,self.MAX_SPEED)
        pos += vel; pos = _clip(pos,self.MIN_POS,self.MAX_POS)
        if pos==self.MIN_POS and vel<0: vel=0
        self._state=[pos,vel]; self._step_count+=1
        done = bool(pos>=self.GOAL_POS or self._step_count>=self.MAX_STEPS)
        return self._state[:], 0.0 if pos>=self.GOAL_POS else -1.0, done, {}


class PendulumEnv(Env):
    MAX_SPEED=8.0; MAX_TORQUE=2.0; DT=0.05
    GRAVITY=10.0; MASS=1.0; LENGTH=1.0; MAX_STEPS=200

    def __init__(self): self._theta=0.0; self._theta_dot=0.0; self._step_count=0
    @property
    def obs_dim(self): return 3
    @property
    def act_dim(self): return 1
    @property
    def continuous(self): return True

    def reset(self) -> list:
        self._theta=_rng.uniform(-math.pi,math.pi)
        self._theta_dot=_rng.uniform(-1.0,1.0); self._step_count=0
        return self._get_obs()

    def _get_obs(self) -> list:
        return [math.cos(self._theta), math.sin(self._theta), self._theta_dot]

    def step(self, action) -> Tuple[list, float, bool, dict]:
        torque = _clip(float(action) if not isinstance(action,list) else action[0], -self.MAX_TORQUE, self.MAX_TORQUE)
        cost = self._theta**2+0.1*self._theta_dot**2+0.001*torque**2
        alpha = (3*self.GRAVITY/(2*self.LENGTH)*math.sin(self._theta)+3.0/(self.MASS*self.LENGTH**2)*torque)
        self._theta_dot = _clip(self._theta_dot+alpha*self.DT,-self.MAX_SPEED,self.MAX_SPEED)
        self._theta += self._theta_dot*self.DT; self._step_count+=1
        return self._get_obs(), float(-cost), self._step_count>=self.MAX_STEPS, {}


class GridWorldEnv(Env):
    def __init__(self, size=5, obstacles=None, goal=None, start=None, obs_type='coord'):
        self.size=size; self.obstacles=set(obstacles or [])
        self.goal=goal or (size-1,size-1); self._start=start
        self.obs_type=obs_type; self._pos=(0,0); self._step_count=0
        self.MAX_STEPS=size*size*4

    @property
    def obs_dim(self): return 2 if self.obs_type=='coord' else self.size*self.size
    @property
    def act_dim(self): return 4

    def reset(self) -> list:
        if self._start: self._pos=self._start
        else:
            while True:
                r=_rng.randint(0,self.size-1); c=_rng.randint(0,self.size-1)
                if (r,c) not in self.obstacles and (r,c)!=self.goal: self._pos=(r,c); break
        self._step_count=0; return self._obs()

    def _obs(self) -> list:
        if self.obs_type=='coord':
            return [self._pos[0]/(self.size-1), self._pos[1]/(self.size-1)]
        grid=[0.0]*self.size*self.size; grid[self._pos[0]*self.size+self._pos[1]]=1.0; return grid

    def step(self, action: int) -> Tuple[list, float, bool, dict]:
        deltas=[(-1,0),(0,1),(1,0),(0,-1)]; dr,dc=deltas[action%4]
        r,c=self._pos; nr,nc=max(0,min(self.size-1,r+dr)),max(0,min(self.size-1,c+dc))
        self._step_count+=1
        if (nr,nc) in self.obstacles: self._pos=(r,c); return self._obs(),-5.0,False,{}
        self._pos=(nr,nc); done=self._pos==self.goal or self._step_count>=self.MAX_STEPS
        return self._obs(), 10.0 if self._pos==self.goal else -1.0, done, {}


# ── Replay Buffers ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int = 1):
        self.capacity=capacity; self.obs_dim=obs_dim; self.act_dim=act_dim
        self._obs=[None]*capacity; self._next=[None]*capacity
        self._acts=[None]*capacity; self._rews=[0.0]*capacity; self._done=[0.0]*capacity
        self._ptr=0; self._size=0

    def push(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self._obs[self._ptr]=_obs_list(obs)[:self.obs_dim]
        self._next[self._ptr]=_obs_list(next_obs)[:self.obs_dim]
        self._acts[self._ptr]=_obs_list(action)[:self.act_dim]
        self._rews[self._ptr]=float(reward); self._done[self._ptr]=float(done)
        self._ptr=(self._ptr+1)%self.capacity; self._size=min(self._size+1,self.capacity)

    def sample(self, batch_size: int) -> Dict[str, list]:
        idx=[_rng.randint(0,self._size-1) for _ in range(batch_size)]
        return {'obs':[self._obs[i] for i in idx],
                'next_obs':[self._next[i] for i in idx],
                'actions':[self._acts[i] for i in idx],
                'rewards':[[self._rews[i]] for i in idx],
                'dones':[[self._done[i]] for i in idx]}

    def __len__(self) -> int: return self._size


class _SumTree:
    def __init__(self, capacity):
        self.capacity=capacity; self.tree=[0.0]*(2*capacity); self.data_ptr=0

    def update(self, idx, priority):
        tree_idx=idx+self.capacity; self.tree[tree_idx]=priority
        while tree_idx>1:
            tree_idx//=2; self.tree[tree_idx]=self.tree[2*tree_idx]+self.tree[2*tree_idx+1]

    def sample(self, value):
        idx=1
        while idx<self.capacity:
            left=2*idx
            if value<=self.tree[left]: idx=left
            else: value-=self.tree[left]; idx=left+1
        return idx-self.capacity

    @property
    def total(self): return self.tree[1]


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, obs_dim, act_dim=1, alpha=0.6, beta=0.4, eps=1e-6):
        super().__init__(capacity, obs_dim, act_dim)
        self.alpha=alpha; self.beta=beta; self.eps=eps
        self._tree=_SumTree(capacity); self._max_priority=1.0

    def push(self, obs, action, reward, next_obs, done) -> None:
        super().push(obs, action, reward, next_obs, done)
        idx=(self._ptr-1)%self.capacity; self._tree.update(idx, self._max_priority**self.alpha)

    def sample(self, batch_size: int) -> Dict[str, list]:
        indices=[]; segment=self._tree.total/batch_size
        for i in range(batch_size):
            v=_rng.uniform(segment*i, segment*(i+1)); indices.append(self._tree.sample(v))
        probs=[self._tree.tree[i+self._tree.capacity] for i in indices]
        total=self._tree.total+1e-9; probs=[p/total for p in probs]
        weights=[(self._size*p+1e-9)**(-self.beta) for p in probs]
        mx=max(weights)+1e-9; weights=[w/mx for w in weights]
        batch={'obs':[self._obs[i] for i in indices],
               'next_obs':[self._next[i] for i in indices],
               'actions':[self._acts[i] for i in indices],
               'rewards':[[self._rews[i]] for i in indices],
               'dones':[[self._done[i]] for i in indices],
               'weights':[[w] for w in weights],
               'indices':indices}
        return batch

    def update_priorities(self, indices: list, priorities: list) -> None:
        for idx, pri in zip(indices, priorities):
            pri=float(abs(pri)+self.eps); self._max_priority=max(self._max_priority,pri)
            self._tree.update(idx, pri**self.alpha)


# ── MLP utility ───────────────────────────────────────────────────────────────

class MLP(Module):
    """Pure Python MLP for RL policy/value functions."""

    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int,
                 activation: str = 'relu', output_activation: str = 'none'):
        super().__init__()
        dims=[in_dim]+hidden_dims+[out_dim]
        self.ws=[]; self.bs=[]
        scale=[math.sqrt(2.0/dims[i]) for i in range(len(dims)-1)]
        for i in range(len(dims)-1):
            w=[[_gauss()*scale[i] for _ in range(dims[i+1])] for _ in range(dims[i])]
            b=[0.0]*dims[i+1]
            w_flat=[v for row in w for v in row]
            w_t=_make_tensor(w_flat,(dims[i],dims[i+1])); b_t=_make_tensor(b,(dims[i+1],))
            w_t.requires_grad=True; b_t.requires_grad=True
            setattr(self,f'w{i}',w_t); setattr(self,f'b{i}',b_t)
            self._parameters[f'w{i}']=w_t; self._parameters[f'b{i}']=b_t
            self.ws.append(w); self.bs.append(b)
        self.dims=dims; self.activation=activation; self.output_activation=output_activation

    def _act_fn(self, x: list, name: str) -> list:
        if name=='relu': return [_relu(v) for v in x]
        if name=='tanh': return [_tanh(v) for v in x]
        if name=='silu': return [_silu(v) for v in x]
        return x[:]

    def forward(self, x: Tensor) -> Tensor:
        flat=_flat(x)
        sh_x=x._data.shape if hasattr(x,'_data') and hasattr(x._data,'shape') else (len(flat)//self.dims[0], self.dims[0])
        B=sh_x[0] if len(sh_x)>=2 else 1
        in_d=self.dims[0]
        # Process batch
        result=[]
        for b in range(B):
            row=flat[b*in_d:(b+1)*in_d]
            for i,(w,b_bias) in enumerate(zip(self.ws,self.bs)):
                out_d=len(b_bias)
                new_row=[sum(row[j]*w[j][k] for j in range(len(row)))+b_bias[k] for k in range(out_d)]
                act=self.activation if i<len(self.ws)-1 else self.output_activation
                row=self._act_fn(new_row, act)
            result.extend(row)
        out_d=self.dims[-1]; sh_out=(B,out_d) if B>1 else (out_d,)
        return _make_tensor(result, sh_out)


def _softmax_2d(logits_flat: list, B: int, V: int) -> list:
    result=[]
    for b in range(B):
        row=logits_flat[b*V:(b+1)*V]
        result.extend(_softmax_list(row))
    return result


# ── Utility functions ─────────────────────────────────────────────────────────

def compute_returns(rewards: List[float], dones: List[bool],
                    gamma: float = 0.99, normalize: bool = True) -> list:
    G=0.0; returns=[]
    for r,d in zip(reversed(rewards), reversed(dones)):
        G=r+gamma*G*(1-float(d)); returns.insert(0, G)
    if normalize and len(returns)>1:
        mu=_mean_list(returns); std=_std_list(returns)+1e-8
        returns=[(v-mu)/std for v in returns]
    return returns


def compute_gae(rewards: List[float], values: List[float],
                dones: List[bool], next_value: float,
                gamma: float = 0.99, lam: float = 0.95) -> list:
    advantages=[]; gae=0.0; values_ext=list(values)+[next_value]
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*values_ext[t+1]*(1-float(dones[t]))-values_ext[t]
        gae=delta+gamma*lam*(1-float(dones[t]))*gae; advantages.insert(0, gae)
    return advantages


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int = 1):
        self.capacity=capacity; self.obs_dim=obs_dim; self.act_dim=act_dim; self.clear()

    def add(self, obs, action, reward, done, value, log_prob) -> None:
        self._obs.append(_obs_list(obs)[:self.obs_dim])
        self._acts.append(_obs_list(action)[:self.act_dim])
        self._rews.append(float(reward)); self._done.append(float(done))
        self._vals.append(float(value)); self._lps.append(float(log_prob))

    def get(self) -> Dict[str, list]:
        return {'obs':self._obs, 'actions':self._acts, 'rewards':self._rews,
                'dones':self._done, 'values':self._vals, 'log_probs':self._lps}

    def clear(self) -> None:
        self._obs=[]; self._acts=[]; self._rews=[]; self._done=[]; self._vals=[]; self._lps=[]

    def __len__(self): return len(self._rews)


# ── DQN ───────────────────────────────────────────────────────────────────────

class DQN:
    def __init__(self, obs_dim, act_dim, hidden_dims=(256,256), gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_freq=100, buffer_size=10000, batch_size=64):
        self.act_dim=act_dim; self.gamma=gamma; self.lr=lr
        self.epsilon=epsilon_start; self.epsilon_end=epsilon_end; self.epsilon_decay=epsilon_decay
        self.target_update_freq=target_update_freq; self.batch_size=batch_size; self._step=0
        self.q_net=MLP(obs_dim,list(hidden_dims),act_dim)
        self.target_net=MLP(obs_dim,list(hidden_dims),act_dim)
        self.update_target()
        self.buffer=ReplayBuffer(buffer_size, obs_dim)

    def select_action(self, obs, greedy: bool = False) -> int:
        if not greedy and _rng.random()<self.epsilon: return _rng.randint(0,self.act_dim-1)
        q=_flat(self.q_net(_make_tensor(_obs_list(obs),(1,len(_obs_list(obs))))))
        return q.index(max(q))

    def train_step(self) -> Optional[float]:
        if len(self.buffer)<self.batch_size: return None
        batch=self.buffer.sample(self.batch_size)
        obs=batch['obs']; actions=[int(a[0]) for a in batch['actions']]
        rewards=[r[0] for r in batch['rewards']]; next_obs=batch['next_obs']
        dones=[d[0] for d in batch['dones']]
        B=len(obs); in_d=len(obs[0])
        obs_flat=[v for row in obs for v in row]
        q_flat=_flat(self.q_net(_make_tensor(obs_flat,(B,in_d))))
        q_sa=[q_flat[b*self.act_dim+actions[b]] for b in range(B)]
        next_flat=[v for row in next_obs for v in row]
        tgt_flat=_flat(self.target_net(_make_tensor(next_flat,(B,in_d))))
        q_next=[max(tgt_flat[b*self.act_dim:(b+1)*self.act_dim]) for b in range(B)]
        targets=[rewards[b]+self.gamma*q_next[b]*(1-dones[b]) for b in range(B)]
        loss=sum((q_sa[b]-targets[b])**2 for b in range(B))/B
        self.epsilon=max(self.epsilon_end, self.epsilon*self.epsilon_decay)
        self._step+=1
        if self._step%self.target_update_freq==0: self.update_target()
        return float(loss)

    def update_target(self) -> None:
        for (n1,p1),(n2,p2) in zip(self.q_net._parameters.items(),self.target_net._parameters.items()):
            w1=_flat(p1); sh=p1._data.shape if hasattr(p1._data,'shape') else (len(w1),)
            p2._data=_make_tensor(w1,sh)._data

    def train(self, env: Env, total_steps: int, log_interval: int = 1000) -> List[float]:
        obs=env.reset(); ep_ret=0.0; all_rewards=[]
        for step in range(total_steps):
            action=self.select_action(obs); next_obs,reward,done,_=env.step(action)
            self.buffer.push(obs,action,reward,next_obs,done)
            ep_ret+=reward; obs=next_obs; self.train_step()
            if done: all_rewards.append(ep_ret); ep_ret=0.0; obs=env.reset()
            if log_interval and step%log_interval==0 and all_rewards:
                print(f"Step {step}: mean_ep_reward={_mean_list(all_rewards[-10:]):.2f} ε={self.epsilon:.3f}")
        return all_rewards


class DoubleDQN(DQN):
    def train_step(self) -> Optional[float]:
        if len(self.buffer)<self.batch_size: return None
        batch=self.buffer.sample(self.batch_size)
        obs=batch['obs']; actions=[int(a[0]) for a in batch['actions']]
        rewards=[r[0] for r in batch['rewards']]; next_obs=batch['next_obs']
        dones=[d[0] for d in batch['dones']]; B=len(obs); in_d=len(obs[0])
        obs_flat=[v for row in obs for v in row]
        q_flat=_flat(self.q_net(_make_tensor(obs_flat,(B,in_d))))
        q_sa=[q_flat[b*self.act_dim+actions[b]] for b in range(B)]
        next_flat=[v for row in next_obs for v in row]
        online_q=_flat(self.q_net(_make_tensor(next_flat,(B,in_d))))
        best_next=[online_q[b*self.act_dim:(b+1)*self.act_dim].index(max(online_q[b*self.act_dim:(b+1)*self.act_dim]))
                   for b in range(B)]
        tgt_q=_flat(self.target_net(_make_tensor(next_flat,(B,in_d))))
        q_next=[tgt_q[b*self.act_dim+best_next[b]] for b in range(B)]
        targets=[rewards[b]+self.gamma*q_next[b]*(1-dones[b]) for b in range(B)]
        loss=sum((q_sa[b]-targets[b])**2 for b in range(B))/B
        self.epsilon=max(self.epsilon_end, self.epsilon*self.epsilon_decay)
        self._step+=1
        if self._step%self.target_update_freq==0: self.update_target()
        return float(loss)


class DuelingDQN(DQN):
    pass  # Inherits DQN; full dueling arch requires custom forward — using base DQN logic


# ── REINFORCE ─────────────────────────────────────────────────────────────────

class REINFORCE:
    def __init__(self, obs_dim, act_dim, hidden_dims=(256,), lr=1e-3, gamma=0.99):
        self.gamma=gamma; self.lr=lr
        self.policy=MLP(obs_dim,list(hidden_dims),act_dim,output_activation='none')

    def select_action(self, obs) -> Tuple[int, float]:
        logits=_flat(self.policy(_make_tensor(_obs_list(obs),(1,len(_obs_list(obs))))))
        probs=_softmax_list(logits)
        r=_rng.random(); cumsum=0.0; action=len(probs)-1
        for i,p in enumerate(probs):
            cumsum+=p
            if cumsum>=r: action=i; break
        return int(action), math.log(probs[action]+1e-8)

    def update(self, rewards: List[float], log_probs: List[float]) -> float:
        returns=compute_returns(rewards,[False]*len(rewards),self.gamma)
        loss=-sum(lp*g for lp,g in zip(log_probs,returns))/max(len(log_probs),1)
        for _,param in self.policy._parameters.items():
            if param.grad is not None:
                g=_flat(param.grad); w=_flat(param)
                new_w=[w[i]-self.lr*g[i] for i in range(len(w))]
                sh=param._data.shape if hasattr(param._data,'shape') else (len(w),)
                param._data=_make_tensor(new_w,sh)._data; param.grad=None
        return float(loss)

    def train(self, env: Env, n_episodes: int) -> List[float]:
        all_returns=[]
        for _ in range(n_episodes):
            obs=env.reset(); log_probs=[]; rewards=[]; done=False
            while not done:
                action,lp=self.select_action(obs); obs,reward,done,_=env.step(action)
                log_probs.append(lp); rewards.append(reward)
            self.update(rewards,log_probs); all_returns.append(sum(rewards))
        return all_returns


# ── A2C ───────────────────────────────────────────────────────────────────────

class ActorCritic(Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256,256)):
        super().__init__()
        self.backbone=MLP(obs_dim,list(hidden_dims),hidden_dims[-1])
        self.actor=MLP(hidden_dims[-1],[],act_dim,output_activation='none')
        self.critic=MLP(hidden_dims[-1],[],1,output_activation='none')
        self._modules['backbone']=self.backbone
        self._modules['actor']=self.actor; self._modules['critic']=self.critic
        self.act_dim=act_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feat=self.backbone(x)
        logits=self.actor(feat); value=self.critic(feat)
        return logits, value


class A2C:
    def __init__(self, obs_dim, act_dim, hidden_dims=(256,256),
                 lr=7e-4, gamma=0.99, n_steps=5, entropy_coef=0.01, vf_coef=0.5):
        self.gamma=gamma; self.n_steps=n_steps; self.entropy_coef=entropy_coef
        self.vf_coef=vf_coef; self.lr=lr
        self.ac=ActorCritic(obs_dim,act_dim,hidden_dims)

    def select_action(self, obs) -> Tuple[int, float, float]:
        in_d=len(_obs_list(obs)); obs_t=_make_tensor(_obs_list(obs),(1,in_d))
        logits,value=self.ac(obs_t)
        probs=_softmax_list(_flat(logits)); val=_flat(value)[0]
        r=_rng.random(); cumsum=0.0; action=len(probs)-1
        for i,p in enumerate(probs):
            cumsum+=p
            if cumsum>=r: action=i; break
        return int(action), math.log(probs[action]+1e-8), val

    def _sgd(self, lr):
        for _,param in self.ac._parameters.items():
            if param.grad is not None:
                g=_flat(param.grad); w=_flat(param)
                sh=param._data.shape if hasattr(param._data,'shape') else (len(w),)
                param._data=_make_tensor([w[i]-lr*g[i] for i in range(len(w))],sh)._data; param.grad=None

    def update(self, trajectory: List[dict]) -> dict:
        rewards=[t['reward'] for t in trajectory]; dones=[t['done'] for t in trajectory]
        values=[t['value'] for t in trajectory]; log_probs=[t['log_prob'] for t in trajectory]
        last_val=0.0 if dones[-1] else values[-1]
        advantages=compute_gae(rewards,values,dones,last_val,self.gamma)
        returns=[advantages[i]+values[i] for i in range(len(values))]
        actor_loss=-_mean_list([lp*adv for lp,adv in zip(log_probs,advantages)])
        critic_loss=_mean_list([(v-r)**2 for v,r in zip(values,returns)])
        self._sgd(self.lr)
        return {'actor_loss': float(actor_loss), 'critic_loss': float(critic_loss)}

    def train(self, env: Env, total_steps: int) -> List[float]:
        obs=env.reset(); ep_ret=0.0; all_rets=[]; traj=[]
        for step in range(total_steps):
            action,lp,val=self.select_action(obs); next_obs,reward,done,_=env.step(action)
            traj.append({'obs':obs,'action':action,'reward':reward,'done':done,'value':val,'log_prob':lp})
            ep_ret+=reward; obs=next_obs
            if len(traj)>=self.n_steps or done: self.update(traj); traj=[]
            if done: all_rets.append(ep_ret); ep_ret=0.0; obs=env.reset()
        return all_rets


# ── PPO ───────────────────────────────────────────────────────────────────────

class PPO:
    def __init__(self, obs_dim, act_dim, hidden_dims=(256,256),
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 n_epochs=10, batch_size=64, entropy_coef=0.01, vf_coef=0.5):
        self.gamma=gamma; self.lam=gae_lambda; self.clip_eps=clip_eps
        self.n_epochs=n_epochs; self.batch_size=batch_size; self.lr=lr
        self.entropy_coef=entropy_coef; self.vf_coef=vf_coef
        self.ac=ActorCritic(obs_dim,act_dim,hidden_dims)
        self.buffer=RolloutBuffer(10000,obs_dim)

    def select_action(self, obs) -> Tuple[int, float, float]:
        in_d=len(_obs_list(obs)); obs_t=_make_tensor(_obs_list(obs),(1,in_d))
        logits,value=self.ac(obs_t); probs=_softmax_list(_flat(logits)); val=_flat(value)[0]
        r=_rng.random(); cumsum=0.0; action=len(probs)-1
        for i,p in enumerate(probs):
            cumsum+=p
            if cumsum>=r: action=i; break
        return int(action), math.log(probs[action]+1e-8), val

    def _sgd(self):
        for _,param in self.ac._parameters.items():
            if param.grad is not None:
                g=_flat(param.grad); w=_flat(param)
                sh=param._data.shape if hasattr(param._data,'shape') else (len(w),)
                param._data=_make_tensor([w[i]-self.lr*g[i] for i in range(len(w))],sh)._data; param.grad=None

    def update(self, rollout: dict) -> dict:
        obs=rollout['obs']; actions=[int(a[0]) if isinstance(a,list) else int(a) for a in rollout['actions']]
        old_lps=rollout['log_probs']; values=rollout['values']
        rewards=rollout['rewards']; dones=rollout['dones']; N=len(obs)
        in_d=len(obs[0])
        last_val=0.0
        if obs:
            _,last_val_t=self.ac(_make_tensor(obs[-1],(1,in_d))); last_val=_flat(last_val_t)[0]
        advantages=compute_gae(rewards,values,dones,last_val,self.gamma,self.lam)
        returns=[advantages[i]+values[i] for i in range(N)]
        total_actor=0.0; total_critic=0.0
        for _ in range(self.n_epochs):
            idx=list(range(N)); _rng.shuffle(idx)
            for start in range(0,N,self.batch_size):
                mb=idx[start:start+self.batch_size]
                mb_obs=[obs[i] for i in mb]; mb_acts=[actions[i] for i in mb]
                mb_adv=[advantages[i] for i in mb]; mb_ret=[returns[i] for i in mb]
                mb_old_lp=[old_lps[i] for i in mb]; bsz=len(mb)
                if bsz==0: continue
                mu=_mean_list(mb_adv); std=_std_list(mb_adv)+1e-8
                mb_adv=[(v-mu)/std for v in mb_adv]
                obs_flat=[v for row in mb_obs for v in row]
                logits,vals=self.ac(_make_tensor(obs_flat,(bsz,in_d)))
                all_probs=_flat(logits); V=self.ac.act_dim
                new_lps=[math.log(max(_softmax_list(all_probs[b*V:(b+1)*V])[mb_acts[b]],1e-8)) for b in range(bsz)]
                ratios=[_exp(new_lps[b]-mb_old_lp[b]) for b in range(bsz)]
                surr1=[ratios[b]*mb_adv[b] for b in range(bsz)]
                surr2=[_clip(ratios[b],1-self.clip_eps,1+self.clip_eps)*mb_adv[b] for b in range(bsz)]
                actor_loss=-_mean_list([min(s1,s2) for s1,s2 in zip(surr1,surr2)])
                val_flat=_flat(vals)
                critic_loss=_mean_list([(val_flat[b]-mb_ret[b])**2 for b in range(bsz)])
                total_actor+=actor_loss; total_critic+=critic_loss
                self._sgd()
        return {'actor_loss':total_actor,'critic_loss':total_critic}

    def train(self, env: Env, total_steps: int, rollout_steps: int = 2048, log_interval: int = 1000) -> List[float]:
        obs=env.reset(); ep_ret=0.0; all_rets=[]; step=0
        while step<total_steps:
            self.buffer.clear()
            for _ in range(rollout_steps):
                action,lp,val=self.select_action(obs); next_obs,reward,done,_=env.step(action)
                self.buffer.add(obs,action,reward,done,val,lp)
                ep_ret+=reward; obs=next_obs; step+=1
                if done: all_rets.append(ep_ret); ep_ret=0.0; obs=env.reset()
            self.update(self.buffer.get())
            if log_interval and step%log_interval<rollout_steps and all_rets:
                print(f"Step {step}: mean_ep_reward={_mean_list(all_rets[-10:]):.2f}")
        return all_rets


# ── SAC ───────────────────────────────────────────────────────────────────────

class SAC:
    def __init__(self, obs_dim, act_dim, hidden_dims=(256,256), lr=3e-4, gamma=0.99,
                 tau=0.005, alpha=0.2, auto_entropy_tuning=True,
                 buffer_size=100000, batch_size=256):
        self.act_dim=act_dim; self.gamma=gamma; self.tau=tau; self.alpha=alpha
        self.lr=lr; self.batch_size=batch_size; self.auto_entropy_tuning=auto_entropy_tuning
        self.actor=MLP(obs_dim,list(hidden_dims),act_dim*2,activation='relu')
        self.critic1=MLP(obs_dim+act_dim,list(hidden_dims),1)
        self.critic2=MLP(obs_dim+act_dim,list(hidden_dims),1)
        self.target1=MLP(obs_dim+act_dim,list(hidden_dims),1)
        self.target2=MLP(obs_dim+act_dim,list(hidden_dims),1)
        self._copy(self.critic1,self.target1); self._copy(self.critic2,self.target2)
        self.buffer=ReplayBuffer(buffer_size,obs_dim,act_dim)
        self.target_entropy=-float(act_dim); self.log_alpha=[0.0]

    def _copy(self,src,dst):
        for (n1,p1),(n2,p2) in zip(src._parameters.items(),dst._parameters.items()):
            p2._data=_make_tensor(_flat(p1),p1._data.shape if hasattr(p1._data,'shape') else (len(_flat(p1)),))._data

    def _soft(self,src,dst):
        for (n1,p1),(n2,p2) in zip(src._parameters.items(),dst._parameters.items()):
            w1=_flat(p1); w2=_flat(p2); sh=p2._data.shape if hasattr(p2._data,'shape') else (len(w2),)
            p2._data=_make_tensor([self.tau*w1[i]+(1-self.tau)*w2[i] for i in range(len(w1))],sh)._data

    def _gauss_policy(self, obs_flat: list, B: int, in_d: int):
        out=_flat(self.actor(_make_tensor(obs_flat,(B,in_d))))
        d=self.act_dim; means=[]; log_stds=[]; actions=[]; log_probs=[]
        for b in range(B):
            mean=out[b*2*d:b*2*d+d]; ls=[_clip(v,-20,2) for v in out[b*2*d+d:b*2*d+2*d]]
            std=[_exp(v) for v in ls]
            eps=[_gauss() for _ in range(d)]
            z=[mean[k]+std[k]*eps[k] for k in range(d)]
            a=[_tanh(v) for v in z]
            lp=sum(-0.5*(eps[k]**2+2*ls[k]+math.log(2*math.pi))-math.log(1-a[k]**2+1e-6) for k in range(d))
            means.append(mean); log_stds.append(ls); actions.append(a); log_probs.append(lp)
        return actions, log_probs

    def select_action(self, obs, deterministic=False):
        obs_l=_obs_list(obs); in_d=len(obs_l)
        actions,_=self._gauss_policy(obs_l,1,in_d)
        return actions[0]

    def _sgd_params(self, net, lr):
        for _,p in net._parameters.items():
            if p.grad is not None:
                g=_flat(p.grad); w=_flat(p); sh=p._data.shape if hasattr(p._data,'shape') else (len(w),)
                p._data=_make_tensor([w[i]-lr*g[i] for i in range(len(w))],sh)._data; p.grad=None

    def train_step(self) -> dict:
        if len(self.buffer)<self.batch_size: return {}
        batch=self.buffer.sample(self.batch_size)
        obs=batch['obs']; actions=batch['actions']; rewards=[r[0] for r in batch['rewards']]
        next_obs=batch['next_obs']; dones=[d[0] for d in batch['dones']]
        B=len(obs); in_d=len(obs[0])
        next_flat=[v for row in next_obs for v in row]
        next_as,next_lps=self._gauss_policy(next_flat,B,in_d)
        sa_next_flat=[v for b in range(B) for v in (next_obs[b]+next_as[b])]
        in_d2=in_d+self.act_dim
        q1n=_flat(self.target1(_make_tensor(sa_next_flat,(B,in_d2))))
        q2n=_flat(self.target2(_make_tensor(sa_next_flat,(B,in_d2))))
        tgt_q=[rewards[b]+self.gamma*(1-dones[b])*(min(q1n[b],q2n[b])-self.alpha*next_lps[b]) for b in range(B)]
        sa_flat=[v for b in range(B) for v in (obs[b]+actions[b])]
        q1=_flat(self.critic1(_make_tensor(sa_flat,(B,in_d2))))
        q2=_flat(self.critic2(_make_tensor(sa_flat,(B,in_d2))))
        critic_loss=_mean_list([(q1[b]-tgt_q[b])**2+(q2[b]-tgt_q[b])**2 for b in range(B)])
        obs_flat=[v for row in obs for v in row]
        new_as,lps=self._gauss_policy(obs_flat,B,in_d)
        sa_new=[v for b in range(B) for v in (obs[b]+new_as[b])]
        q1n2=_flat(self.critic1(_make_tensor(sa_new,(B,in_d2))))
        q2n2=_flat(self.critic2(_make_tensor(sa_new,(B,in_d2))))
        actor_loss=_mean_list([self.alpha*lps[b]-min(q1n2[b],q2n2[b]) for b in range(B)])
        self._sgd_params(self.critic1,self.lr); self._sgd_params(self.critic2,self.lr)
        self._sgd_params(self.actor,self.lr)
        alpha_loss=0.0
        if self.auto_entropy_tuning:
            alpha_loss=_mean_list([(-self.log_alpha[0])*(lp+self.target_entropy) for lp in lps])
            self.log_alpha[0]-=self.lr*(-_mean_list(lps)-self.target_entropy)
            self.alpha=_exp(self.log_alpha[0])
        self._soft(self.critic1,self.target1); self._soft(self.critic2,self.target2)
        return {'critic_loss':float(critic_loss),'actor_loss':float(actor_loss),'alpha_loss':alpha_loss}

    def train(self, env: Env, total_steps: int, log_interval: int = 1000) -> List[float]:
        obs=env.reset(); ep_ret=0.0; all_rets=[]
        for step in range(total_steps):
            action=self.select_action(obs); next_obs,reward,done,_=env.step(action)
            self.buffer.push(obs,action,reward,next_obs,done); ep_ret+=reward; obs=next_obs
            self.train_step()
            if done: all_rets.append(ep_ret); ep_ret=0.0; obs=env.reset()
            if log_interval and step%log_interval==0 and all_rets:
                print(f"Step {step}: mean={_mean_list(all_rets[-10:]):.2f} α={self.alpha:.3f}")
        return all_rets


# ── TD3 ───────────────────────────────────────────────────────────────────────

class TD3:
    def __init__(self, obs_dim, act_dim, max_action, hidden_dims=(256,256), lr=3e-4,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, buffer_size=100000, batch_size=256):
        self.act_dim=act_dim; self.max_action=max_action; self.gamma=gamma; self.tau=tau
        self.lr=lr; self.policy_noise=policy_noise; self.noise_clip=noise_clip
        self.policy_delay=policy_delay; self.batch_size=batch_size; self._step=0
        self.actor=MLP(obs_dim,list(hidden_dims),act_dim,output_activation='tanh')
        self.actor_target=MLP(obs_dim,list(hidden_dims),act_dim,output_activation='tanh')
        self.critic1=MLP(obs_dim+act_dim,list(hidden_dims),1)
        self.critic2=MLP(obs_dim+act_dim,list(hidden_dims),1)
        self.target1=MLP(obs_dim+act_dim,list(hidden_dims),1)
        self.target2=MLP(obs_dim+act_dim,list(hidden_dims),1)
        for src,dst in [(self.actor,self.actor_target),(self.critic1,self.target1),(self.critic2,self.target2)]:
            self._copy(src,dst)
        self.buffer=ReplayBuffer(buffer_size,obs_dim,act_dim)

    def _copy(self,src,dst):
        for (n1,p1),(n2,p2) in zip(src._parameters.items(),dst._parameters.items()):
            w=_flat(p1); sh=p1._data.shape if hasattr(p1._data,'shape') else (len(w),)
            p2._data=_make_tensor(w,sh)._data

    def _soft(self,src,dst):
        for (n1,p1),(n2,p2) in zip(src._parameters.items(),dst._parameters.items()):
            w1=_flat(p1); w2=_flat(p2); sh=p2._data.shape if hasattr(p2._data,'shape') else (len(w2),)
            p2._data=_make_tensor([self.tau*w1[i]+(1-self.tau)*w2[i] for i in range(len(w1))],sh)._data

    def _sgd(self,net):
        for _,p in net._parameters.items():
            if p.grad is not None:
                g=_flat(p.grad); w=_flat(p); sh=p._data.shape if hasattr(p._data,'shape') else (len(w),)
                p._data=_make_tensor([w[i]-self.lr*g[i] for i in range(len(w))],sh)._data; p.grad=None

    def select_action(self, obs, noise: float = 0.1):
        in_d=len(_obs_list(obs)); a_flat=_flat(self.actor(_make_tensor(_obs_list(obs),(1,in_d))))
        action=[v*self.max_action for v in a_flat]
        if noise>0:
            action=[v+_gauss()*noise for v in action]
        return [_clip(v,-self.max_action,self.max_action) for v in action]

    def train_step(self, step=None) -> dict:
        if len(self.buffer)<self.batch_size: return {}
        self._step+=1
        batch=self.buffer.sample(self.batch_size)
        obs=batch['obs']; actions=batch['actions']; rewards=[r[0] for r in batch['rewards']]
        next_obs=batch['next_obs']; dones=[d[0] for d in batch['dones']]
        B=len(obs); in_d=len(obs[0]); ad=self.act_dim
        next_flat=[v for row in next_obs for v in row]
        next_a_raw=_flat(self.actor_target(_make_tensor(next_flat,(B,in_d))))
        noise=[_clip(_gauss()*self.policy_noise,-self.noise_clip,self.noise_clip) for _ in range(B*ad)]
        next_a=[[_clip(next_a_raw[b*ad+k]*self.max_action+noise[b*ad+k],-self.max_action,self.max_action) for k in range(ad)] for b in range(B)]
        sa_next=[v for b in range(B) for v in (next_obs[b]+next_a[b])]
        in_d2=in_d+ad
        q1t=_flat(self.target1(_make_tensor(sa_next,(B,in_d2))))
        q2t=_flat(self.target2(_make_tensor(sa_next,(B,in_d2))))
        tgt_q=[rewards[b]+self.gamma*(1-dones[b])*min(q1t[b],q2t[b]) for b in range(B)]
        sa=[v for b in range(B) for v in (obs[b]+actions[b])]
        q1=_flat(self.critic1(_make_tensor(sa,(B,in_d2))))
        q2=_flat(self.critic2(_make_tensor(sa,(B,in_d2))))
        critic_loss=_mean_list([(q1[b]-tgt_q[b])**2+(q2[b]-tgt_q[b])**2 for b in range(B)])
        self._sgd(self.critic1); self._sgd(self.critic2)
        actor_loss=0.0
        if self._step%self.policy_delay==0:
            obs_flat=[v for row in obs for v in row]
            new_a_raw=_flat(self.actor(_make_tensor(obs_flat,(B,in_d))))
            new_a=[[new_a_raw[b*ad+k]*self.max_action for k in range(ad)] for b in range(B)]
            sa_new=[v for b in range(B) for v in (obs[b]+new_a[b])]
            q1n=_flat(self.critic1(_make_tensor(sa_new,(B,in_d2))))
            actor_loss=-_mean_list(q1n)
            self._sgd(self.actor)
            for src,dst in [(self.actor,self.actor_target),(self.critic1,self.target1),(self.critic2,self.target2)]:
                self._soft(src,dst)
        return {'critic_loss':float(critic_loss),'actor_loss':float(actor_loss)}

    def train(self, env: Env, total_steps: int, log_interval: int = 1000) -> List[float]:
        obs=env.reset(); ep_ret=0.0; all_rets=[]
        for step in range(total_steps):
            action=self.select_action(obs); next_obs,reward,done,_=env.step(action)
            self.buffer.push(obs,action,reward,next_obs,done); ep_ret+=reward; obs=next_obs
            self.train_step(step)
            if done: all_rets.append(ep_ret); ep_ret=0.0; obs=env.reset()
            if log_interval and step%log_interval==0 and all_rets:
                print(f"Step {step}: mean={_mean_list(all_rets[-10:]):.2f}")
        return all_rets



# ── DDPG ──────────────────────────────────────────────────────────────────────

class DDPG:
    """Deep Deterministic Policy Gradient (Lillicrap et al. 2016)."""
    def __init__(self, obs_dim, act_dim, max_action, hidden_dims=(256,256), lr=1e-3,
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=256):
        self.act_dim=act_dim; self.max_action=max_action; self.gamma=gamma
        self.tau=tau; self.lr=lr; self.batch_size=batch_size
        self.actor=MLP(obs_dim,list(hidden_dims),act_dim,output_activation='tanh')
        self.actor_target=MLP(obs_dim,list(hidden_dims),act_dim,output_activation='tanh')
        self.critic=MLP(obs_dim+act_dim,list(hidden_dims),1)
        self.critic_target=MLP(obs_dim+act_dim,list(hidden_dims),1)
        for src,dst in [(self.actor,self.actor_target),(self.critic,self.critic_target)]:
            self._copy(src,dst)
        self.buffer=ReplayBuffer(buffer_size,obs_dim,act_dim)

    def _copy(self,src,dst):
        for (n1,p1),(n2,p2) in zip(src._parameters.items(),dst._parameters.items()):
            w=_flat(p1); sh=p1._data.shape if hasattr(p1._data,'shape') else (len(w),)
            p2._data=_make_tensor(w,sh)._data

    def _soft(self,src,dst):
        for (n1,p1),(n2,p2) in zip(src._parameters.items(),dst._parameters.items()):
            w1=_flat(p1); w2=_flat(p2); sh=p2._data.shape if hasattr(p2._data,'shape') else (len(w2),)
            p2._data=_make_tensor([self.tau*w1[i]+(1-self.tau)*w2[i] for i in range(len(w1))],sh)._data

    def _sgd(self,net):
        for _,p in net._parameters.items():
            if p.grad is not None:
                g=_flat(p.grad); w=_flat(p); sh=p._data.shape if hasattr(p._data,'shape') else (len(w),)
                p._data=_make_tensor([w[i]-self.lr*g[i] for i in range(len(w))],sh)._data; p.grad=None

    def select_action(self, obs, noise: float = 0.1):
        in_d=len(_obs_list(obs)); a=_flat(self.actor(_make_tensor(_obs_list(obs),(1,in_d))))
        a=[v*self.max_action+(_gauss()*noise if noise>0 else 0.0) for v in a]
        return [_clip(v,-self.max_action,self.max_action) for v in a]

    def train_step(self) -> dict:
        if len(self.buffer)<self.batch_size: return {}
        batch=self.buffer.sample(self.batch_size)
        obs=batch['obs']; actions=batch['actions']; rewards=[r[0] for r in batch['rewards']]
        next_obs=batch['next_obs']; dones=[d[0] for d in batch['dones']]
        B=len(obs); in_d=len(obs[0]); ad=self.act_dim; in_d2=in_d+ad
        next_flat=[v for row in next_obs for v in row]
        next_a_raw=_flat(self.actor_target(_make_tensor(next_flat,(B,in_d))))
        next_a=[[next_a_raw[b*ad+k]*self.max_action for k in range(ad)] for b in range(B)]
        sa_next=[v for b in range(B) for v in (next_obs[b]+next_a[b])]
        qt=_flat(self.critic_target(_make_tensor(sa_next,(B,in_d2))))
        tgt=[rewards[b]+self.gamma*(1-dones[b])*qt[b] for b in range(B)]
        sa=[v for b in range(B) for v in (obs[b]+actions[b])]
        q=_flat(self.critic(_make_tensor(sa,(B,in_d2))))
        critic_loss=_mean_list([(q[b]-tgt[b])**2 for b in range(B)])
        self._sgd(self.critic)
        obs_flat=[v for row in obs for v in row]
        new_a_raw=_flat(self.actor(_make_tensor(obs_flat,(B,in_d))))
        new_a=[[new_a_raw[b*ad+k]*self.max_action for k in range(ad)] for b in range(B)]
        sa_new=[v for b in range(B) for v in (obs[b]+new_a[b])]
        q_new=_flat(self.critic(_make_tensor(sa_new,(B,in_d2))))
        actor_loss=-_mean_list(q_new)
        self._sgd(self.actor)
        for src,dst in [(self.actor,self.actor_target),(self.critic,self.critic_target)]:
            self._soft(src,dst)
        return {'critic_loss':float(critic_loss),'actor_loss':float(actor_loss)}

    def train(self, env: Env, total_steps: int, log_interval: int = 1000) -> List[float]:
        obs=env.reset(); ep_ret=0.0; all_rets=[]
        for step in range(total_steps):
            action=self.select_action(obs); next_obs,reward,done,_=env.step(action)
            self.buffer.push(obs,action,reward,next_obs,done); ep_ret+=reward; obs=next_obs
            self.train_step()
            if done: all_rets.append(ep_ret); ep_ret=0.0; obs=env.reset()
            if log_interval and step%log_interval==0 and all_rets:
                print(f"Step {step}: mean={_mean_list(all_rets[-10:]):.2f}")
        return all_rets


# ── C51 (Categorical/Distributional DQN) ──────────────────────────────────────

class C51:
    """Categorical DQN / C51 (Bellemare et al. 2017).
    Models Q as a distribution over n_atoms support atoms in [v_min, v_max].
    """
    def __init__(self, obs_dim, n_actions, n_atoms=51, v_min=-10.0, v_max=10.0,
                 hidden_dims=(128,128), lr=1e-3, gamma=0.99,
                 buffer_size=50000, batch_size=64, eps_start=1.0, eps_end=0.01,
                 eps_decay=0.995):
        self.n_actions=n_actions; self.n_atoms=n_atoms
        self.v_min=v_min; self.v_max=v_max; self.delta=(v_max-v_min)/(n_atoms-1)
        self.atoms=[v_min+i*self.delta for i in range(n_atoms)]
        self.gamma=gamma; self.batch_size=batch_size; self.eps=eps_start
        self.eps_end=eps_end; self.eps_decay=eps_decay
        # Network outputs logits: (obs_dim) -> (n_actions * n_atoms)
        self.net=MLP(obs_dim,list(hidden_dims),n_actions*n_atoms)
        self.target=MLP(obs_dim,list(hidden_dims),n_actions*n_atoms)
        self._copy(self.net,self.target)
        self.buffer=ReplayBuffer(buffer_size,obs_dim,1)
        self.lr=lr; self._step=0; self._update_freq=100

    def _copy(self,src,dst):
        for (n1,p1),(n2,p2) in zip(src._parameters.items(),dst._parameters.items()):
            w=_flat(p1); sh=p1._data.shape if hasattr(p1._data,'shape') else (len(w),)
            p2._data=_make_tensor(w,sh)._data

    def _sgd(self,net):
        for _,p in net._parameters.items():
            if p.grad is not None:
                g=_flat(p.grad); w=_flat(p); sh=p._data.shape if hasattr(p._data,'shape') else (len(w),)
                p._data=_make_tensor([w[i]-self.lr*g[i] for i in range(len(w))],sh)._data; p.grad=None

    def _softmax_rows(self, logits_flat, B, n):
        """Softmax over n columns for B rows; returns list of B*n probs."""
        probs=[]
        for b in range(B):
            row=logits_flat[b*n:(b+1)*n]
            mx=max(row); exps=[_safe_exp(v-mx) for v in row]; s=sum(exps)
            probs.extend([v/s for v in exps])
        return probs

    def _q_from_probs(self, probs_flat, B):
        """Compute E[Z] = sum(atoms * probs) for each action."""
        q=[]
        for b in range(B):
            q_b=[]
            for a in range(self.n_actions):
                q_b.append(sum(self.atoms[i]*probs_flat[b*self.n_actions*self.n_atoms+a*self.n_atoms+i]
                               for i in range(self.n_atoms)))
            q.append(q_b)
        return q

    def select_action(self, obs) -> int:
        import random as _rnd
        if _rnd.random()<self.eps: return _rnd.randint(0,self.n_actions-1)
        in_d=len(_obs_list(obs))
        logits=_flat(self.net(_make_tensor(_obs_list(obs),(1,in_d))))
        probs=self._softmax_rows(logits,1,self.n_atoms)
        q=[sum(self.atoms[i]*probs[a*self.n_atoms+i] for i in range(self.n_atoms)) for a in range(self.n_actions)]
        return max(range(self.n_actions),key=lambda a:q[a])

    def train_step(self) -> dict:
        if len(self.buffer)<self.batch_size: return {}
        self._step+=1; self.eps=max(self.eps_end,self.eps*self.eps_decay)
        batch=self.buffer.sample(self.batch_size)
        obs=batch['obs']; actions_raw=batch['actions']
        rewards=[r[0] for r in batch['rewards']]
        next_obs=batch['next_obs']; dones=[d[0] for d in batch['dones']]
        B=len(obs); in_d=len(obs[0])
        actions=[int(round(actions_raw[b][0])) for b in range(B)]
        # Target distribution via Bellman projection
        next_flat=[v for row in next_obs for v in row]
        tgt_logits=_flat(self.target(_make_tensor(next_flat,(B,in_d))))
        tgt_probs=self._softmax_rows(tgt_logits,B,self.n_atoms)
        # Choose greedy action from online net for double-C51
        online_logits=_flat(self.net(_make_tensor(next_flat,(B,in_d))))
        online_probs=self._softmax_rows(online_logits,B,self.n_atoms)
        # Distribute projected atoms into target distribution
        m_list=[]
        for b in range(B):
            m=[0.0]*self.n_atoms
            for i in range(self.n_atoms):
                # Find greedy action
                q_a=[sum(online_probs[(b*self.n_actions+a)*self.n_atoms+i2]*self.atoms[i2]
                         for i2 in range(self.n_atoms)) for a in range(self.n_actions)]
                a_star=max(range(self.n_actions),key=lambda a:q_a[a])
                p_i=tgt_probs[b*self.n_atoms+i] if self.n_actions==1 else \
                    tgt_probs[(b*self.n_actions+a_star)*self.n_atoms+i] if \
                    b*self.n_actions*self.n_atoms+a_star*self.n_atoms+i < len(tgt_probs) else \
                    tgt_probs[b*self.n_atoms+i] if b*self.n_atoms+i < len(tgt_probs) else 0.0
                tz=_clip(rewards[b]+self.gamma*(1-dones[b])*self.atoms[i],self.v_min,self.v_max)
                bj=(tz-self.v_min)/self.delta; lo=int(bj); hi=min(lo+1,self.n_atoms-1)
                m[lo]+=p_i*(hi-bj); m[hi]+=p_i*(bj-lo)
            m_list.append(m)
        # Cross-entropy loss: -sum(m * log(p_a))
        obs_flat=[v for row in obs for v in row]
        logits=_flat(self.net(_make_tensor(obs_flat,(B,in_d))))
        probs=self._softmax_rows(logits,B,self.n_atoms)
        loss=0.0
        for b in range(B):
            a=actions[b]
            for i in range(self.n_atoms):
                p=max(probs[b*self.n_atoms+i] if self.n_actions==1
                      else probs[(b*self.n_actions+a)*self.n_atoms+i]
                      if b*self.n_actions*self.n_atoms+a*self.n_atoms+i<len(probs)
                      else probs[b*self.n_atoms+i],1e-8)
                loss-=m_list[b][i]*_safe_log(p)
        loss/=B
        self._sgd(self.net)
        if self._step%self._update_freq==0: self._copy(self.net,self.target)
        return {'loss':float(loss)}

    def train(self, env: Env, total_steps: int, log_interval: int = 500) -> List[float]:
        obs=env.reset(); ep_ret=0.0; all_rets=[]
        for step in range(total_steps):
            action=self.select_action(obs); next_obs,reward,done,_=env.step([float(action)])
            self.buffer.push(obs,[float(action)],reward,next_obs,done); ep_ret+=reward; obs=next_obs
            self.train_step()
            if done: all_rets.append(ep_ret); ep_ret=0.0; obs=env.reset()
            if log_interval and step%log_interval==0 and all_rets:
                print(f"Step {step}: mean={_mean_list(all_rets[-10:]):.2f}")
        return all_rets


__all__ = [
    'Env', 'CartPoleEnv', 'MountainCarEnv', 'PendulumEnv', 'GridWorldEnv',
    'ReplayBuffer', 'PrioritizedReplayBuffer', 'RolloutBuffer',
    'DQN', 'DoubleDQN', 'DuelingDQN',
    'REINFORCE', 'ActorCritic', 'A2C', 'PPO', 'SAC', 'TD3',
    'DDPG', 'C51',
    'compute_returns', 'compute_gae',
]
