import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from .util import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class MIXBase(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params=None):
        super().__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_maxpool2d = args.use_maxpool2d
        self.hidden_size = args.hidden_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.use_resnet = args.use_resnet
        self.use_original_size = args.use_original_size
        self.pretrained_global_resnet = args.pretrained_global_resnet
        self.cnn_keys = []
        self.local_cnn_keys = []
        self.embed_keys = []
        self.mlp_keys = []
        self.n_cnn_input = 0
        self.n_embed_input = 0
        self.n_mlp_input = 0

        for key in obs_shape:
            if obs_shape[key].__class__.__name__ == "Box":
                key_obs_shape = obs_shape[key].shape
                if len(key_obs_shape) == 3:
                    if key in ["local_obs", "local_merge_obs"]:
                        self.local_cnn_keys.append(key)
                    else:
                        self.cnn_keys.append(key)
                else:
                    if "orientation" in key:
                        self.embed_keys.append(key)
                    else:
                        self.mlp_keys.append(key)
            else:
                raise NotImplementedError

        if len(self.cnn_keys) > 0:
            if self.use_resnet:
                self.cnn = self._build_resnet_model(obs_shape, self.cnn_keys, self.hidden_size)
            else:
                if self.use_original_size:
                    cnn_layers_params = "32,7,1,1 64,5,1,1 128,3,1,1 64,3,1,1 32,3,1,1"
                self.cnn = self._build_cnn_model(
                    obs_shape,
                    self.cnn_keys,
                    cnn_layers_params,
                    self.hidden_size,
                    self._use_orthogonal,
                    self._activation_id,
                )

        if len(self.local_cnn_keys) > 0:
            if self.use_resnet:
                self.local_cnn = self._build_resnet_model(obs_shape, self.local_cnn_keys, self.hidden_size)
            else:
                self.local_cnn = self._build_cnn_model(
                    obs_shape,
                    self.local_cnn_keys,
                    cnn_layers_params,
                    self.hidden_size,
                    self._use_orthogonal,
                    self._activation_id,
                )

        if len(self.embed_keys) > 0:
            self.embed = self._build_embed_model(obs_shape)

        if len(self.mlp_keys) > 0:
            self.mlp = self._build_mlp_model(
                obs_shape,
                self.mlp_hidden_size,
                self._use_orthogonal,
                self._activation_id,
            )

    def forward(self, x):
        out_x = x
        if len(self.cnn_keys) > 0:
            cnn_input = self._build_cnn_input(x, self.cnn_keys)
            cnn_x = self.cnn(cnn_input)
            out_x = cnn_x

        if len(self.local_cnn_keys) > 0:
            local_cnn_input = self._build_cnn_input(x, self.local_cnn_keys)
            local_cnn_x = self.local_cnn(local_cnn_input)
            out_x = torch.cat([out_x, local_cnn_x], dim=1)

        if len(self.embed_keys) > 0:
            embed_input = self._build_embed_input(x)
            embed_x = self.embed(embed_input.long()).view(embed_input.size(0), -1)
            out_x = torch.cat([out_x, embed_x], dim=1)

        if len(self.mlp_keys) > 0:
            mlp_input = self._build_mlp_input(x)
            mlp_x = self.mlp(mlp_input).view(mlp_input.size(0), -1)
            out_x = torch.cat([out_x, mlp_x], dim=1)  # ! wrong

        return out_x

    def _build_resnet_model(self, obs_shape, cnn_keys, hidden_size):
        n_cnn_input = 0
        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                n_cnn_input += obs_shape[key].shape[2]
            elif key in [
                "global_map",
                "local_map",
                "global_obs",
                "local_obs",
                "global_merge_obs",
                "local_merge_obs",
                "trace_image",
                "global_merge_goal",
                "gt_map",
                "vector_cnn",
            ]:
                n_cnn_input += obs_shape[key].shape[0]
            else:
                raise NotImplementedError

        cnn_layers = [nn.Conv2d(int(n_cnn_input), 64, kernel_size=7, stride=2, padding=3, bias=False)]
        resnet = models.resnet18(pretrained=self.pretrained_global_resnet)
        cnn_layers += list(resnet.children())[1:-1]
        cnn_layers += [Flatten(), nn.Linear(resnet.fc.in_features, hidden_size)]

        return nn.Sequential(*cnn_layers)

    def _build_cnn_model(
        self,
        obs_shape,
        cnn_keys,
        cnn_layers_params,
        hidden_size,
        use_orthogonal,
        activation_id,
    ):
        if cnn_layers_params is None:
            cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
        else:

            def _convert(params):
                output = []
                for l in params.split(" "):
                    output.append(tuple(map(int, l.split(","))))
                return output

            cnn_layers_params = _convert(cnn_layers_params)

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        n_cnn_input = 0
        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                n_cnn_input += obs_shape[key].shape[2]
                cnn_dims = np.array(obs_shape[key].shape[:2], dtype=np.float32)
            elif key in [
                "global_map",
                "local_map",
                "global_obs",
                "local_obs",
                "global_merge_obs",
                "local_merge_obs",
                "trace_image",
                "global_merge_goal",
                "gt_map",
                "vector_cnn",
            ]:
                n_cnn_input += obs_shape[key].shape[0]
                cnn_dims = np.array(obs_shape[key].shape[1:3], dtype=np.float32)
            else:
                raise NotImplementedError

        cnn_layers = []
        prev_out_channels = None
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                cnn_layers.append(nn.MaxPool2d(2))

            if i == 0:
                in_channels = n_cnn_input
            else:
                in_channels = prev_out_channels

            cnn_layers.append(
                init_(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            )
            # if i != len(cnn_layers_params) - 1:
            cnn_layers.append(active_func)
            prev_out_channels = out_channels

        for i, (_, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                cnn_dims = self._maxpool_output_dim(
                    dimension=cnn_dims,
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array([2, 2], dtype=np.float32),
                    stride=np.array([2, 2], dtype=np.float32),
                )
            cnn_dims = self._cnn_output_dim(
                dimension=cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )

        if (cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1]) > 20000:
            cnn_layers += [
                Flatten(),
                init_(nn.Linear(cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1], 2048)),
                active_func,
                nn.LayerNorm(2048),
                init_(nn.Linear(2048, hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
            ]
        else:
            cnn_layers += [
                Flatten(),
                init_(
                    nn.Linear(
                        cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1],
                        hidden_size,
                    )
                ),
                active_func,
                nn.LayerNorm(hidden_size),
            ]
        return nn.Sequential(*cnn_layers)

    def _build_embed_model(self, obs_shape):
        self.embed_dim = 0
        for key in self.embed_keys:
            self.n_embed_input = 72
            self.n_embed_output = 8
            self.embed_dim += np.prod(obs_shape[key].shape)

        return nn.Embedding(self.n_embed_input, self.n_embed_output)

    def _build_mlp_model(self, obs_shape, hidden_size, use_orthogonal, activation_id):
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        for key in self.mlp_keys:
            self.n_mlp_input += np.prod(obs_shape[key].shape)

        return nn.Sequential(
            init_(nn.Linear(self.n_mlp_input, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )

    def _maxpool_output_dim(self, dimension, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(((dimension[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1))
            )
        return tuple(out_dimension)

    def _cnn_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1)
                )
            )
        return tuple(out_dimension)

    def _build_cnn_input(self, obs, cnn_keys):
        cnn_input = []

        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                cnn_input.append(obs[key].permute(0, 3, 1, 2) / 255.0)
            elif key in [
                "global_map",
                "local_map",
                "global_obs",
                "local_obs",
                "global_merge_obs",
                "trace_image",
                "local_merge_obs",
                "global_merge_goal",
                "gt_map",
                "vector_cnn",
            ]:
                cnn_input.append(obs[key])
            else:
                raise NotImplementedError

        cnn_input = torch.cat(cnn_input, dim=1)
        return cnn_input

    def _build_embed_input(self, obs):
        embed_input = []
        for key in self.embed_keys:
            embed_input.append(obs[key].view(obs[key].size(0), -1))

        embed_input = torch.cat(embed_input, dim=1)
        return embed_input

    def _build_mlp_input(self, obs):
        mlp_input = []
        for key in self.mlp_keys:
            mlp_input.append(obs[key].view(obs[key].size(0), -1))

        mlp_input = torch.cat(mlp_input, dim=1)
        return mlp_input

    @property
    def output_size(self):
        output_size = 0
        if len(self.cnn_keys) > 0:
            output_size += self.hidden_size

        if len(self.local_cnn_keys) > 0:
            output_size += self.hidden_size

        if len(self.embed_keys) > 0:
            output_size += 8 * self.embed_dim

        if len(self.mlp_keys) > 0:
            output_size += self.mlp_hidden_size
        return output_size
