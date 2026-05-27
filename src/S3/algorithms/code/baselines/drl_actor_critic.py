"""
Faithful implementation of Deep Multi-Agent Reinforcement Learning Based
Cooperative Edge Caching.

Reference:
  Zhong, C., Gursoy, M. C., Velipasalar, S.,
  "Deep Reinforcement Learning-Based Edge Caching in Wireless Networks",
  IEEE Trans. Cognitive Communications and Networking, vol. 6, no. 1,
  pp. 48-61, 2020.   DOI: 10.1109/TCCN.2020.2968920

Algorithm:
  1. Multi-Agent Framework — one DQN agent per cache node
  2. State representation:
     - Normalized content popularity vector (F-dim)
     - Binary cache occupancy vector (F-dim)
  3. Action: content placement scores → select top-C_cap items
  4. Reward: total cache hits in current step → delay saving signal
  5. Training components (faithful to paper):
     - Q-Network: 2-layer MLP with ReLU (numpy-based, no torch dependency)
     - Experience Replay Buffer (capacity 500, mini-batch 32)
     - Target Network with soft update (τ=0.01, every 5 steps)
     - Epsilon-greedy exploration with decay (0.5 → 0.05 over training)
"""

import numpy as np
import networkx as nx
from collections import deque

from ..config import (
    CONTENT_CATALOG_SIZE,
    CONTENT_SIZE_BITS, CACHE_SERVE_BW_MBPS, GS_SERVE_BW_MBPS,
    CONTENT_SIZE_MB,
)
from .. import config as cfg

F = CONTENT_CATALOG_SIZE


# ═══════════════════════════════════════════════════════════
#  Numpy-based 2-layer MLP (Q-Network)
# ═══════════════════════════════════════════════════════════

class QNetwork:
    """Two-layer MLP: input → hidden (ReLU) → output.

    Implements forward pass and gradient-based update with MSE loss.
    Uses Xavier initialization for stable training.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.005):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        self.lr = lr
        # Cache for backward pass
        self._x = None
        self._z1 = None
        self._a1 = None

    def forward(self, x):
        """Forward pass: x → ReLU(W1·x + b1) → W2·h + b2."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)     # ReLU
        out = self._a1 @ self.W2 + self.b2
        return out

    def update(self, targets):
        """Backward pass with MSE loss against targets. Updates weights."""
        batch = self._x.shape[0]
        output = self._a1 @ self.W2 + self.b2

        # dL/dout = 2/N * (out - target), simplified to (out - target)/N
        dout = (output - targets) / batch

        # Layer 2
        dW2 = self._a1.T @ dout
        db2 = dout.sum(axis=0)

        # Layer 1
        da1 = dout @ self.W2.T
        dz1 = da1 * (self._z1 > 0).astype(np.float32)   # ReLU grad
        dW1 = self._x.T @ dz1
        db1 = dz1.sum(axis=0)

        # Gradient clipping (max norm 1.0) for stability
        for g in [dW1, db1, dW2, db2]:
            norm = np.linalg.norm(g)
            if norm > 1.0:
                g *= 1.0 / norm

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def soft_update_from(self, source, tau=0.01):
        """Polyak averaging: θ_target ← τ·θ_source + (1-τ)·θ_target."""
        self.W1 = tau * source.W1 + (1 - tau) * self.W1
        self.b1 = tau * source.b1 + (1 - tau) * self.b1
        self.W2 = tau * source.W2 + (1 - tau) * self.W2
        self.b2 = tau * source.b2 + (1 - tau) * self.b2

    def copy_from(self, source):
        """Hard copy weights from source network."""
        self.W1 = source.W1.copy()
        self.b1 = source.b1.copy()
        self.W2 = source.W2.copy()
        self.b2 = source.b2.copy()


# ═══════════════════════════════════════════════════════════
#  Experience Replay Buffer
# ═══════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-capacity circular buffer for experience tuples."""

    def __init__(self, capacity=500):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_mask, reward, next_state):
        self.buffer.append((state.copy(), action_mask.copy(), reward,
                            next_state.copy()))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch])
        masks = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        return states, masks, rewards, next_states

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════
#  DQN Agent (one per cache node)
# ═══════════════════════════════════════════════════════════

class DQNAgent:
    """Deep Q-Network agent for a single cache node.

    State:  [popularity_vector(F), cache_binary(F)]  dim = 2F
    Output: Q-value for each content item (F-dim)
    Action: cache the top-C_cap items by Q-value

    Training follows the standard DQN procedure from Zhong et al.:
      Q_target = r + γ · max_a Q_target(s', a)
      Loss = MSE(Q(s,a) - Q_target)
    """

    def __init__(self, hidden=64, lr=0.005, gamma=0.9, eps_start=0.5,
                 eps_min=0.05, eps_decay=0.98):
        state_dim = 2 * F
        self.q_net = QNetwork(state_dim, hidden, F, lr=lr)
        self.target_net = QNetwork(state_dim, hidden, F, lr=lr)
        self.target_net.copy_from(self.q_net)

        self.replay = ReplayBuffer(capacity=500)
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.cache = set()
        self._last_state = None
        self._last_action_mask = None

    def _build_state(self, popularity_scores):
        """Build compact state vector [pop_norm, cache_binary]."""
        pop = np.zeros(F, dtype=np.float32)
        for fid, sc in popularity_scores.items():
            if 0 <= fid < F:
                pop[fid] = sc
        pmax = pop.max()
        if pmax > 0:
            pop /= pmax

        cache_bin = np.zeros(F, dtype=np.float32)
        for fid in self.cache:
            if 0 <= fid < F:
                cache_bin[fid] = 1.0

        return np.concatenate([pop, cache_bin])

    def select_action(self, popularity_scores):
        """Select top-C_cap content items using epsilon-greedy policy."""
        C_CAP = cfg.CACHE_CAPACITY
        state = self._build_state(popularity_scores)

        q_values = self.q_net.forward(state).flatten()

        # Epsilon-greedy: with prob epsilon, add exploration noise
        if np.random.random() < self.epsilon:
            noise = np.random.uniform(0, 0.5, F).astype(np.float32)
            q_values = q_values + noise

        top_items = np.argsort(q_values)[-C_CAP:]
        self.cache = set(int(x) for x in top_items)

        # Store state for learning
        action_mask = np.zeros(F, dtype=np.float32)
        for fid in self.cache:
            action_mask[fid] = 1.0
        self._last_state = state
        self._last_action_mask = action_mask

        return self.cache

    def observe_reward(self, reward, popularity_scores):
        """Store transition and train if enough experience."""
        if self._last_state is None:
            return

        next_state = self._build_state(popularity_scores)
        self.replay.push(self._last_state, self._last_action_mask,
                         reward, next_state)

        # Train if enough samples
        batch_size = min(32, len(self.replay))
        if len(self.replay) >= 32:
            self._train_step(batch_size)

        # Decay epsilon
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def _train_step(self, batch_size):
        """Sample mini-batch and update Q-network."""
        states, masks, rewards, next_states = self.replay.sample(batch_size)

        # Current Q-values
        q_current = self.q_net.forward(states)           # (B, F)

        # Target Q-values: r + γ * max Q_target(s')
        q_next = self.target_net.forward(next_states)     # (B, F)
        max_q_next = q_next.max(axis=1)                   # (B,)
        # Target for cached items: r + γ*max_Q; uncached items: current Q
        targets = q_current.copy()
        for i in range(batch_size):
            cached_mask = masks[i] > 0.5
            targets[i, cached_mask] = rewards[i] + self.gamma * max_q_next[i]

        # Update Q-network
        self.q_net.forward(states)   # re-forward for cached activations
        self.q_net.update(targets)

    def update_target(self, tau=0.01):
        """Soft-update target network."""
        self.target_net.soft_update_from(self.q_net, tau)


# ═══════════════════════════════════════════════════════════
#  Multi-Agent DRL Cache Manager
# ═══════════════════════════════════════════════════════════

class MaDRLManager:
    """Multi-agent DRL manager: one DQN agent per cache node.

    Coordinates independent agents, each learning its own caching policy.
    Agents share the global popularity signal but make local decisions.
    """

    def __init__(self):
        self._agents = {}
        self._step = 0

    def _get_agent(self, node):
        if node not in self._agents:
            self._agents[node] = DQNAgent()
        return self._agents[node]

    def decide_placement(self, cache_nodes, popularity_scores):
        """Each node's DQN agent independently selects content to cache."""
        placement = {}
        for c in cache_nodes:
            agent = self._get_agent(c)
            placement[c] = agent.select_action(popularity_scores)
        return placement

    def feedback(self, cache_nodes, hit_counts, popularity_scores):
        """Provide reward signal to each agent and update networks.

        hit_counts : dict {node: int}  number of cache hits at each node
        """
        self._step += 1
        for c in cache_nodes:
            agent = self._get_agent(c)
            reward = float(hit_counts.get(c, 0))
            agent.observe_reward(reward, popularity_scores)

        # Soft-update target networks every 5 steps
        if self._step % 5 == 0:
            for c in cache_nodes:
                self._get_agent(c).update_target(tau=0.01)

    def set_eval_mode(self, epsilon=0.0):
        """Freeze exploration for evaluation-only rollouts."""
        for agent in self._agents.values():
            agent.epsilon = epsilon

    def agent_count(self):
        return len(self._agents)


# ═══════════════════════════════════════════════════════════
#  Routing
# ═══════════════════════════════════════════════════════════

def route_madrl(G, requester, placement, content_id, type_map):
    """Route request using MADRL placement (search all placed nodes)."""
    if not G.has_node(requester):
        return None, None, False

    transfer_cache = (CONTENT_SIZE_BITS / (CACHE_SERVE_BW_MBPS * 1e6)) * 1000.0
    transfer_origin = (CONTENT_SIZE_BITS / (GS_SERVE_BW_MBPS * 1e6)) * 1000.0

    # Check all placement nodes
    best_delay, best_path = float('inf'), None
    for c, cached in placement.items():
        if content_id not in cached:
            continue
        if not G.has_node(c):
            continue
        try:
            path = nx.dijkstra_path(G, requester, c, weight='eff_delay')
            d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            total = d + transfer_cache
            if total < best_delay:
                best_delay = total
                best_path = path
        except nx.NetworkXNoPath:
            pass

    if best_path is not None:
        traffic = CONTENT_SIZE_MB * (len(best_path) - 1)
        return best_delay, traffic, True

    # Origin fallback
    gs_nodes = [n for n, t in type_map.items() if t == 'GS' and G.has_node(n)]
    best_delay, best_path = float('inf'), None
    for gs in gs_nodes:
        try:
            path = nx.dijkstra_path(G, requester, gs, weight='eff_delay')
            d = sum(G[path[i]][path[i + 1]]['eff_delay'] for i in range(len(path) - 1))
            total = 2 * d + transfer_origin
            if total < best_delay:
                best_delay = total
                best_path = path
        except nx.NetworkXNoPath:
            pass

    if best_path is None:
        return None, None, False
    traffic = CONTENT_SIZE_MB * (len(best_path) - 1)
    return best_delay, traffic, False
