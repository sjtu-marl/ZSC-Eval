import numpy as np
import torch
import torch.nn as nn

from .util import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params=None):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_maxpool2d = args.use_maxpool2d
        self.hidden_size = args.hidden_size
        self.cnn_keys = ["rgb"]

        self.cnn = self._build_cnn_model(
            obs_shape,
            self.cnn_keys,
            cnn_layers_params,
            self.hidden_size,
            self._use_orthogonal,
            self._activation_id,
        )

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
            cnn_layers_params = [(16, 5, 1, 0), (32, 3, 1, 0), (16, 3, 1, 0)]
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
                n_cnn_input += obs_shape[2]
                cnn_dims = np.array(obs_shape[:2], dtype=np.float32)
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
                n_cnn_input += obs_shape.shape[0]
                cnn_dims = np.array(obs_shape[1:3], dtype=np.float32)
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
                init_(nn.Linear(hidden_size, hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
                init_(nn.Linear(hidden_size, hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
            ]
        return nn.Sequential(*cnn_layers)

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

    def _build_cnn_input(self, obs, cnn_keys):
        cnn_input = []

        for key in cnn_keys:
            if key in ["rgb", "depth", "image", "occupy_image"]:
                cnn_input.append(obs.permute(0, 3, 1, 2) / 255.0)
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
                cnn_input.append(obs)
            else:
                raise NotImplementedError

        cnn_input = torch.cat(cnn_input, dim=1)
        return cnn_input

    def forward(self, x):
        cnn_input = self._build_cnn_input(x, self.cnn_keys)
        cnn_x = self.cnn(cnn_input)
        return cnn_x

    @property
    def output_size(self):
        return self.hidden_size
