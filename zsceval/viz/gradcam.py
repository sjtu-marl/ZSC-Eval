import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, obs, available_actions, rnn_states, masks, target_action=None):
        # Forward pass
        actions, action_log_probs, rnn_states = self.model.forward(obs, rnn_states, masks, available_actions)
        
        # Backward pass
        self.model.zero_grad()        
        score = action_log_probs.sum()
        score.backward()
        
        # GradCAM 계산
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1)  # (B, H, W)
        cam = F.relu(cam)  # ReLU
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam[0].detach().cpu().numpy()
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()