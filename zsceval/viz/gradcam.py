import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        # Allow single layer or multiple layers
        self.target_layers = target_layer if isinstance(target_layer, (list, tuple)) else [target_layer]

        # Per-layer caches
        self.activations_per_layer = [None] * len(self.target_layers)
        self.gradients_per_layer = [None] * len(self.target_layers)

        # Register hooks for all target layers
        self.forward_hooks = [
            layer.register_forward_hook(self._make_forward_hook(idx))
            for idx, layer in enumerate(self.target_layers)
        ]
        self.backward_hooks = [
            layer.register_full_backward_hook(self._make_backward_hook(idx))
            for idx, layer in enumerate(self.target_layers)
        ]

    def _make_forward_hook(self, idx):
        def _hook(module, input, output):
            self.activations_per_layer[idx] = output
        return _hook

    def _make_backward_hook(self, idx):
        def _hook(module, grad_input, grad_output):
            self.gradients_per_layer[idx] = grad_output[0]
        return _hook

    def __call__(self, obs, available_actions, rnn_states, masks, target_action=None):
        actions, action_log_probs, rnn_states = self.model.forward(obs, rnn_states, masks, available_actions)

        self.model.zero_grad()
        score = action_log_probs.sum()
        score.backward()

        cams = []
        target_size = None

        # Compute CAM per layer
        for acts, grads in zip(self.activations_per_layer, self.gradients_per_layer):
            if acts is None or grads is None:
                continue
            weights = grads.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
            cam = (weights * acts).sum(dim=1)  # (B, H, W)
            cam = F.relu(cam)

            # Normalize per-layer CAM to [0,1]
            cam_min = cam.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
            cam_max = cam.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            cams.append(cam)
            if target_size is None:
                target_size = cam.shape[-2:]
            else:
                # choose the largest spatial size across layers to aggregate
                target_size = (
                    max(target_size[0], cam.shape[-2]),
                    max(target_size[1], cam.shape[-1]),
                )

        # Upsample cams to a common size and average
        cams_up = []
        for c in cams:
            # c: (B, H, W) -> (B, 1, H, W) for interpolate
            c4 = c.unsqueeze(1)
            c4_up = F.interpolate(c4, size=target_size, mode="bilinear", align_corners=False)
            c_up = c4_up.squeeze(1)  # (B, H, W)
            cams_up.append(c_up)

        cam_agg = torch.stack(cams_up, dim=0).mean(dim=0)  # (B, H, W)

        cam_agg = cam_agg[0]  # first in batch
        cam_agg = cam_agg.clamp(0.0, 1.0)
        return cam_agg.detach().cpu().numpy()

    def remove_hooks(self):
        for h in self.forward_hooks:
            h.remove()
        for h in self.backward_hooks:
            h.remove()
