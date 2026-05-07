import torch
import torch.nn.functional as F
import torch.linalg as linalg
from core.shiva_policy import ContinuousSACPolicy

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1, dtype=torch.float32)
        self.data = [None] * capacity
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, p):
        change = p - self.tree[idx].item()
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child].item():
                parent_idx = left_child
            else:
                v -= self.tree[left_child].item()
                parent_idx = right_child
        return leaf_idx, self.tree[leaf_idx].item(), self.data[leaf_idx - self.capacity + 1]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def add(self, sample):
        self.tree.add(self.max_priority, sample)

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        segment = self.tree.tree[0].item() / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            v = torch.empty(1).uniform_(a, b).item()
            idx, p, data = self.tree.get_leaf(v)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        priorities_tensor = torch.tensor(priorities, dtype=torch.float32)
        sampling_probabilities = priorities_tensor / self.tree.tree[0].item()
        is_weights = torch.pow(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (torch.abs(torch.as_tensor(error)) + 1e-6) ** self.alpha
            p = p.item()
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

class ShivaTrainer:
    def __init__(self, d_model, action_dim, shiva_model=None, latent_aligner=None, emotional_core=None, capacity=1000000, device="cpu"):
        self.device = torch.device(device)
        self.policy = ContinuousSACPolicy(d_model, action_dim).to(self.device)
        self.target_policy = ContinuousSACPolicy(d_model, action_dim).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        self.buffer = PrioritizedReplayBuffer(capacity)
        self.gamma = 0.99
        self.tau = 0.005  # Soft update coefficient
        self.alpha_entropy = 0.2  # SAC temperature
        
        self.actor_optimizer = torch.optim.Adam(
            list(self.policy.actor1.parameters()) + 
            list(self.policy.actor2.parameters()) + 
            list(self.policy.gate.parameters()), lr=3e-4
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.policy.critic1.parameters()) + 
            list(self.policy.critic2.parameters()), lr=3e-4
        )
        
        self.model = shiva_model if shiva_model is not None else self.policy
        self.aligner = latent_aligner
        self.emotions = emotional_core

    def _process_batch(self, batch):
        states = torch.stack([s[0] for s in batch]).to(self.device)
        actions = torch.stack([s[1] for s in batch]).to(self.device)
        rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack([s[3] for s in batch]).to(self.device)
        dones = torch.tensor([s[4] for s in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    def update_step(self, batch_size):
        if self.buffer.tree.write < batch_size and self.buffer.tree.data[batch_size] is None:
            return None 
        batch, idxs, is_weights = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = self._process_batch(batch)
        is_weights = is_weights.to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.target_policy.get_action(next_states)
            q1_target, q2_target = self.target_policy.evaluate_q(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha_entropy * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * min_q_target

        current_q1, current_q2 = self.policy.evaluate_q(states, actions)
        
        
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2

        critic1_loss = (is_weights * F.mse_loss(current_q1, target_q, reduction='none')).mean()
        critic2_loss = (is_weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs, _ = self.policy.get_action(states)
        q1_new, q2_new = self.policy.evaluate_q(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (is_weights * (self.alpha_entropy * log_probs - min_q_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        new_priorities = (torch.abs(td_error1) + torch.abs(td_error2)).detach().cpu() / 2
        self.buffer.update_priorities(idxs, new_priorities)

        self._soft_update()
        
        return critic_loss.item(), actor_loss.item()

    def _soft_update(self):
        for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def fit_weight_dim(self, W_ext, target_shape):
        if W_ext.shape == target_shape:
            return W_ext
    
        U, S, Vh = linalg.svd(W_ext, full_matrices=False)

        U_target = U[:target_shape[0], :min(U.shape[1], target_shape[1])]
        S_target = torch.diag(S[:min(len(S), target_shape[0], target_shape[1])])
        Vh_target = Vh[:min(Vh.shape[0], target_shape[0], target_shape[1]), :target_shape[1]]

        result = torch.zeros(target_shape)
        fitted = U_target @ S_target @ Vh_target
        result[:fitted.shape[0], :fitted.shape[1]] = fitted
        return result

    def average_attention_heads(self, W_mha, src_heads, target_heads):
        d_model = W_mha.shape[0]
        d_head = d_model // src_heads
        
        reshaped = W_mha.view(src_heads, d_head, d_model)
        bucket_size = src_heads // target_heads
        
        averaged = torch.stack([
            reshaped[i*bucket_size : (i+1)*bucket_size].mean(dim=0)
            for i in range(target_heads)
        ])
        return averaged.view(-1, d_model)

    def rapid_frankenmerge(self, ext_state_dict, ext_config):
        if hasattr(self.model, "config"):
            target_dim = self.model.config.hidden_size #type: ignore
            target_heads = self.model.config.num_heads #type: ignore
        else:
            target_dim = self.model.backbone.d_model
            target_heads = self.model.backbone.num_heads
        new_state = {}

        for name, param in ext_state_dict.items():
            if "attn" in name or "attention" in name:
                # Handle Attention Head Compression
                new_state[name] = self.average_attention_heads(param, ext_config['num_heads'], target_heads)
            else:
                # Handle Dimensional Fitting
                target_shape = self.model.state_dict()[name].shape if name in self.model.state_dict() else param.shape
                new_state[name] = self.fit_weight_dim(param, target_shape)

        self.model.load_state_dict(new_state, strict=False)
        # Trigger 'Curiosity' update in emotional core after ingestion
        if self.emotions is not None:
            self.emotions.update_homeostasis(action_impact=0.1, environment_surprise=0.8)