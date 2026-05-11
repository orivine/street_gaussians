import math

import torch
import torch.nn as nn

from lib.utils.general_utils import quaternion_raw_multiply_theta


def _cfg_get(cfg_node, key, default=None):
    if isinstance(cfg_node, dict):
        return cfg_node.get(key, default)
    return getattr(cfg_node, key, default)


class EMDPoseLite(nn.Module):
    def __init__(self, obj_info, tracklet_timestamps, motion_cfg, device=None):
        super().__init__()
        self.cfg = motion_cfg
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.time_encoding = _cfg_get(motion_cfg, 'time_encoding', 'learnable')
        if self.time_encoding not in ['learnable', 'fourier']:
            raise NotImplementedError(f'Motion time encoding "{self.time_encoding}" is not implemented yet')

        self.temporal_embedding_dim = int(_cfg_get(motion_cfg, 'temporal_embedding_dim', 4))
        self.min_embeddings = int(_cfg_get(motion_cfg, 'min_embeddings', 30))
        self.max_embeddings = int(_cfg_get(motion_cfg, 'max_embeddings', 150))
        self.c2f_temporal_iter = int(_cfg_get(motion_cfg, 'c2f_temporal_iter', 25000))
        self.object_embedding_dim = int(_cfg_get(motion_cfg, 'object_embedding_dim', 16))
        self.use_coarse_fine = bool(_cfg_get(motion_cfg, 'use_coarse_fine', True))
        self.hidden_dim = int(_cfg_get(motion_cfg, 'hidden_dim', 64))
        self.num_layers = int(_cfg_get(motion_cfg, 'num_layers', 2))
        self.activation = _cfg_get(motion_cfg, 'activation', 'silu')
        self.zero_init = bool(_cfg_get(motion_cfg, 'zero_init', True))
        self.delta_t_scale = float(_cfg_get(motion_cfg, 'delta_t_scale', 0.05))
        self.delta_r_scale = float(_cfg_get(motion_cfg, 'delta_r_scale', 0.03))
        self.warmup_start = int(_cfg_get(motion_cfg, 'warmup_start', 2000))
        self.warmup_end = int(_cfg_get(motion_cfg, 'warmup_end', 8000))
        self.lr_mlp = float(_cfg_get(motion_cfg, 'lr_mlp', 0.0005))
        self.lr_embedding = float(_cfg_get(motion_cfg, 'lr_embedding', 0.001))
        self.lambda_motion_reg_t = float(_cfg_get(motion_cfg, 'lambda_motion_reg_t', 0.001))
        self.lambda_motion_reg_r = float(_cfg_get(motion_cfg, 'lambda_motion_reg_r', 0.001))
        self.apply_to_val = bool(_cfg_get(motion_cfg, 'apply_to_val', True))
        self.fail_on_missing_track = bool(_cfg_get(motion_cfg, 'fail_on_missing_track', False))

        if self.min_embeddings < 1 or self.max_embeddings < self.min_embeddings:
            raise ValueError('Motion temporal embedding counts must satisfy 1 <= min_embeddings <= max_embeddings')
        if self.temporal_embedding_dim < 1:
            raise ValueError('Motion temporal_embedding_dim must be positive')
        if self.object_embedding_dim < 1:
            raise ValueError('Motion object_embedding_dim must be positive')

        self.track_ids = sorted(int(track_id) for track_id in obj_info.keys())
        self.track_id_to_index = {track_id: idx for idx, track_id in enumerate(self.track_ids)}
        self.track_time_bounds = {}
        fallback_start = float(tracklet_timestamps[0]) if len(tracklet_timestamps) > 0 else 0.0
        fallback_end = float(tracklet_timestamps[-1]) if len(tracklet_timestamps) > 0 else fallback_start + 1.0
        for track_id in self.track_ids:
            obj_meta = obj_info[track_id]
            start = float(obj_meta.get('start_timestamp', fallback_start))
            end = float(obj_meta.get('end_timestamp', fallback_end))
            if end <= start:
                end = start + 1e-6
            self.track_time_bounds[track_id] = (start, end)

        self.object_embedding = nn.Embedding(len(self.track_ids), self.object_embedding_dim)
        if self.time_encoding == 'learnable':
            self.temporal_embedding = nn.Parameter(
                torch.zeros(self.max_embeddings, self.temporal_embedding_dim)
            )
            nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.01)
        else:
            self.register_parameter('temporal_embedding', None)

        feature_dim = self.temporal_embedding_dim + self.object_embedding_dim
        self.coarse_mlp = self._build_mlp(feature_dim, self.hidden_dim)
        self.coarse_head = nn.Linear(self.hidden_dim, 4)
        if self.use_coarse_fine:
            self.fine_mlp = self._build_mlp(feature_dim + self.hidden_dim, self.hidden_dim)
            self.fine_head = nn.Linear(self.hidden_dim, 4)
        else:
            self.fine_mlp = None
            self.fine_head = None

        if self.zero_init:
            self._zero_init_head(self.coarse_head)
            if self.fine_head is not None:
                self._zero_init_head(self.fine_head)

        self.optimizer = None
        self.loaded_iteration = None
        self.current_iteration = None
        self.clear_current_view()
        self.to(self.device)

    def _activation(self):
        if self.activation == 'silu':
            return nn.SiLU()
        if self.activation == 'relu':
            return nn.ReLU(inplace=True)
        if self.activation == 'gelu':
            return nn.GELU()
        raise NotImplementedError(f'Motion activation "{self.activation}" is not implemented yet')

    def _build_mlp(self, input_dim, hidden_dim):
        layers = []
        last_dim = input_dim
        for _ in range(max(1, self.num_layers)):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(self._activation())
            last_dim = hidden_dim
        return nn.Sequential(*layers)

    @staticmethod
    def _zero_init_head(head):
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def save_state_dict(self, is_final):
        state_dict = {'params': super().state_dict()}
        if not is_final and self.optimizer is not None:
            state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, load_optimizer=False):
        super().load_state_dict(state_dict['params'])
        if load_optimizer and self.optimizer is not None and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def training_setup(self):
        mlp_params = []
        mlp_params.extend(self.coarse_mlp.parameters())
        mlp_params.extend(self.coarse_head.parameters())
        if self.fine_mlp is not None:
            mlp_params.extend(self.fine_mlp.parameters())
        if self.fine_head is not None:
            mlp_params.extend(self.fine_head.parameters())

        embedding_params = list(self.object_embedding.parameters())
        if self.temporal_embedding is not None:
            embedding_params.append(self.temporal_embedding)

        params = [
            {'params': mlp_params, 'lr': self.lr_mlp, 'name': 'motion_mlp'},
            {'params': embedding_params, 'lr': self.lr_embedding, 'name': 'motion_embedding'},
        ]
        self.optimizer = torch.optim.Adam(params=params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        self.current_iteration = iteration
        if self.optimizer is None:
            return
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'motion_mlp':
                param_group['lr'] = self.lr_mlp
            elif param_group['name'] == 'motion_embedding':
                param_group['lr'] = self.lr_embedding

    def update_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def set_loaded_iteration(self, iteration):
        if iteration is not None:
            self.loaded_iteration = int(iteration)

    def _resolve_iteration(self, iteration):
        if iteration is not None:
            return int(iteration)
        if self.current_iteration is not None:
            return int(self.current_iteration)
        if self.loaded_iteration is not None:
            return int(self.loaded_iteration)
        return None

    def compute_warmup(self, iteration=None):
        iteration = self._resolve_iteration(iteration)
        if iteration is None:
            return 1.0
        if iteration < self.warmup_start:
            return 0.0
        if self.warmup_end <= self.warmup_start:
            return 1.0
        if iteration >= self.warmup_end:
            return 1.0
        return float(iteration - self.warmup_start) / float(self.warmup_end - self.warmup_start)

    def effective_temporal_embeddings(self, iteration=None):
        iteration = self._resolve_iteration(iteration)
        if iteration is None:
            return self.max_embeddings
        if self.c2f_temporal_iter <= 0:
            return self.max_embeddings
        progress = min(max(float(iteration), 0.0), float(self.c2f_temporal_iter)) / float(self.c2f_temporal_iter)
        n_embeddings = self.min_embeddings + (self.max_embeddings - self.min_embeddings) * progress
        return int(round(n_embeddings))

    def clear_current_view(self):
        self._current_delta_t = []
        self._current_delta_r = []
        self._current_warmup = []

    def _track_index_tensor(self, track_id):
        track_id = int(track_id)
        if track_id not in self.track_id_to_index:
            if self.fail_on_missing_track:
                raise KeyError(f'Track id {track_id} is missing from EMDPoseLite')
            return None
        return torch.tensor(self.track_id_to_index[track_id], dtype=torch.long, device=self.device)

    def _normalized_time(self, track_id, camera):
        track_id = int(track_id)
        timestamp = float(camera.meta['timestamp'])
        start, end = self.track_time_bounds[track_id]
        tau = (timestamp - start) / max(end - start, 1e-6)
        tau = min(max(tau, 0.0), 1.0)
        return torch.tensor([tau], dtype=torch.float32, device=self.device)

    def _temporal_features(self, tau, iteration=None):
        if self.time_encoding == 'fourier':
            return self._fourier_temporal_features(tau)

        n_embeddings = self.effective_temporal_embeddings(iteration)
        n_embeddings = max(1, min(n_embeddings, self.max_embeddings))
        table = self.temporal_embedding[:n_embeddings]
        if n_embeddings == 1:
            return table[0]

        pos = tau.squeeze(0) * float(n_embeddings - 1)
        idx0 = torch.floor(pos).long().clamp(0, n_embeddings - 1)
        idx1 = torch.clamp(idx0 + 1, max=n_embeddings - 1)
        frac = (pos - idx0.float()).unsqueeze(0)
        return table[idx0] * (1.0 - frac) + table[idx1] * frac

    def _fourier_temporal_features(self, tau):
        half_dim = self.temporal_embedding_dim // 2
        features = []
        if half_dim > 0:
            freqs = (2.0 ** torch.arange(half_dim, device=self.device, dtype=torch.float32)) * math.pi
            angles = tau * freqs
            features.extend([torch.sin(angles), torch.cos(angles)])
        if self.temporal_embedding_dim % 2 == 1:
            features.append(tau)
        if len(features) == 0:
            return torch.zeros(self.temporal_embedding_dim, device=self.device)
        encoded = torch.cat(features, dim=0)
        return encoded[:self.temporal_embedding_dim]

    def predict_residual(self, track_id, camera, iteration=None):
        track_index = self._track_index_tensor(track_id)
        if track_index is None:
            return None

        tau = self._normalized_time(track_id, camera)
        temporal_feature = self._temporal_features(tau, iteration=iteration)
        object_feature = self.object_embedding(track_index)
        feature = torch.cat([temporal_feature, object_feature], dim=0).unsqueeze(0)

        coarse_hidden = self.coarse_mlp(feature)
        coarse_raw = self.coarse_head(coarse_hidden).squeeze(0)
        if self.use_coarse_fine:
            fine_feature = torch.cat([feature, coarse_hidden], dim=-1)
            fine_hidden = self.fine_mlp(fine_feature)
            fine_raw = self.fine_head(fine_hidden).squeeze(0)
            raw = coarse_raw + fine_raw
        else:
            raw = coarse_raw

        delta_t = self.delta_t_scale * torch.tanh(raw[:3])
        delta_r = self.delta_r_scale * torch.tanh(raw[3:4]).squeeze(0)
        return delta_t, delta_r

    def refine_pose(self, track_id, camera, base_rot, base_trans, iteration=None):
        if camera.meta.get('is_val', False) and not self.apply_to_val:
            return base_rot, base_trans, False

        residual = self.predict_residual(track_id, camera, iteration=iteration)
        if residual is None:
            return base_rot, base_trans, False

        delta_t, delta_r = residual
        warmup = self.compute_warmup(iteration=iteration)
        warmup_tensor = torch.tensor(warmup, dtype=base_trans.dtype, device=base_trans.device)
        delta_t = delta_t.to(device=base_trans.device, dtype=base_trans.dtype)
        delta_r = delta_r.to(device=base_rot.device, dtype=base_rot.dtype)
        delta_t_used = warmup_tensor * delta_t
        delta_r_used = warmup_tensor * delta_r

        refined_trans = base_trans + delta_t_used
        refined_rot = quaternion_raw_multiply_theta(base_rot, delta_r_used)
        refined_rot = torch.nn.functional.normalize(refined_rot.unsqueeze(0), dim=-1).squeeze(0)

        self._current_delta_t.append(delta_t_used)
        self._current_delta_r.append(delta_r_used)
        self._current_warmup.append(warmup)
        return refined_rot, refined_trans, True

    def regularization_loss(self, iteration=None, return_stats=False):
        if len(self._current_delta_t) == 0:
            loss = torch.zeros((), dtype=torch.float32, device=self.device)
            if not return_stats:
                return loss
            stats = {
                'motion_delta_t_mean': 0.0,
                'motion_delta_t_max': 0.0,
                'motion_delta_r_mean': 0.0,
                'motion_delta_r_max': 0.0,
                'motion_reg_loss': 0.0,
                'motion_warmup': self.compute_warmup(iteration=iteration),
                'motion_mode_emd_pose_lite': 1.0,
            }
            return loss, stats

        delta_t = torch.stack(self._current_delta_t, dim=0)
        delta_r = torch.stack(self._current_delta_r, dim=0).reshape(-1)
        loss_t = delta_t.pow(2).sum(dim=-1).mean()
        loss_r = delta_r.pow(2).mean()
        loss = self.lambda_motion_reg_t * loss_t + self.lambda_motion_reg_r * loss_r

        if not return_stats:
            return loss

        with torch.no_grad():
            delta_t_norm = delta_t.detach().norm(dim=-1)
            delta_r_abs = delta_r.detach().abs()
            warmup = sum(self._current_warmup) / max(len(self._current_warmup), 1)
            stats = {
                'motion_delta_t_mean': float(delta_t_norm.mean().item()),
                'motion_delta_t_max': float(delta_t_norm.max().item()),
                'motion_delta_r_mean': float(delta_r_abs.mean().item()),
                'motion_delta_r_max': float(delta_r_abs.max().item()),
                'motion_reg_loss': float(loss.detach().item()),
                'motion_warmup': float(warmup),
                'motion_mode_emd_pose_lite': 1.0,
            }
        return loss, stats
