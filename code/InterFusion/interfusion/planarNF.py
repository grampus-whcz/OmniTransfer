# import numpy as np
# import torch
# import torch.nn as nn
# from data_config import *

# # def init_a_layer(layer: nn.Module):
# #     print(f"{type(layer)}")
# #     for name, param in layer.named_parameters():
# #         if name.startswith("weight"):
# #             nn.init.kaiming_normal_(param)
# #         else:
# #             nn.init.constant_(param, 0)

# class PlanarNormalizingFlow(nn.Module):
#     """
#     A single layer Planar Normalizing Flow (Danilo 2016) with `tanh` activation
#     function, as well as the invertible trick.  The `x` and `y` are assumed to
#     be 1-D random variable (i.e., ``value_ndims == 1``)

#     .. math::

#         \\begin{aligned}
#             \\mathbf{y} &= \\mathbf{x} +
#                 \\mathbf{\\hat{u}} \\tanh(\\mathbf{w}^\\top\\mathbf{x} + b) \\\\
#             \\mathbf{\\hat{u}} &= \\mathbf{u} +
#                 \\left[m(\\mathbf{w}^\\top \\mathbf{u}) -
#                        (\\mathbf{w}^\\top \\mathbf{u})\\right]
#                 \\cdot \\frac{\\mathbf{w}}{\\|\\mathbf{w}\\|_2^2} \\\\
#             m(a) &= -1 + \\log(1+\\exp(a))
#         \\end{aligned}
#     """
#     def build(self, shape=None):
#         lim_w = np.sqrt(2. / np.prod(shape))
#         lim_u = np.sqrt(2)
        
#         self.shape = shape
#         w = nn.Parameter(torch.empty(shape)[None], requires_grad=True).to(device=global_device)
#         nn.init.uniform_(w, -lim_w, lim_w)
        
#         u = nn.Parameter(torch.empty(shape)[None], requires_grad=True).to(device=global_device)
#         nn.init.uniform_(u, -lim_u, lim_u)

#         b = nn.Parameter(torch.zeros(1), requires_grad=True).to(device=global_device)
        
#         wu = w.matmul(u.transpose(-1, -2))
#         u_hat = u + (-1 + nn.Softplus()(wu) - wu) * w / torch.sum(torch.square(w))  # shape == [1, n_units]

#         self._w, self._b, self._u, self._u_hat = w, b, u, u_hat

#     def __init__(self, shape=None):
#         """
#         Construct a new :class:`PlanarNormalizingFlow`.

#         Args:
#             w_initializer: The initializer for parameter `w`.
#             w_regularizer: The regularizer for parameter `w`.
#             b_regularizer: The regularizer for parameter `b`.
#             b_initializer: The initializer for parameter `b`.
#             u_regularizer: The regularizer for parameter `u`.
#             u_initializer: The initializer for parameter `u`.
#             trainable (bool): Whether or not the parameters are trainable?
#                 (default :obj:`True`)
#         """
#         super(PlanarNormalizingFlow, self).__init__()
#         self.shape = shape
#         self.build(shape=self.shape)
#         # self.apply(init_a_layer)
    
#     def reset_parameters(self):
#         self.build(shape=self.shape)


#     def forward(self, input):

#         x, log_det_previous = input
#         # flatten x for better performance
#         # x_flatten, s1, s2 = flatten_to_ndims(x, 2)  # x.shape == [?, n_units]
#         wxb = torch.matmul(x, self._w.transpose(-1, -2)) + self._b  # shape == [?, 1]
#         tanh_wxb = torch.tanh(wxb)  # shape == [?, 1]

#         # compute y = f(x)
#         y = x + self._u_hat * tanh_wxb  # shape == [?, n_units]
#         # y = unflatten_from_ndims(y, s1, s2)

#         # compute log(det|df/dz|)
#         grad = 1. - torch.square(tanh_wxb)  # dtanh(x)/dx = 1 - tanh^2(x)
#         phi = grad * self._w  # shape == [?, n_units]
#         u_phi = torch.matmul(phi, self._u_hat.transpose(-1, -2))  # shape == [?, 1]
#         det_jac = 1. + u_phi  # shape == [?, 1]
#         log_det = torch.log(torch.abs(det_jac))  # shape == [?, 1]
#         # log_det = unflatten_from_ndims(tf.squeeze(log_det, -1), s1, s2)
#         # now returns the transformed sample and log-determinant
#         return y, log_det



import numpy as np
import torch
import torch.nn as nn
from data_config import *  # 保留原数据配置导入

class PlanarNormalizingFlow(nn.Module):
    """
    A single layer Planar Normalizing Flow (Danilo 2016) with `tanh` activation
    function, as well as the invertible trick.  The `x` and `y` are assumed to
    be 1-D random variable (i.e., ``value_ndims == 1``)

    .. math::

        \\begin{aligned}
            \\mathbf{y} &= \\mathbf{x} +
                \\mathbf{\\hat{u}} \\tanh(\\mathbf{w}^\\top\\mathbf{x} + b) \\\\
            \\mathbf{\\hat{u}} &= \\mathbf{u} +
                \\left[m(\\mathbf{w}^\\top \\mathbf{u}) -
                       (\\mathbf{w}^\\top \\mathbf{u})\\right]
                \\cdot \\frac{\\mathbf{w}}{\\|\\mathbf{w}\\|_2^2} \\\\
            m(a) &= -1 + \\log(1+\\exp(a))
        \\end{aligned}
    """
    def build(self, shape=None):
        lim_w = np.sqrt(2. / np.prod(shape))
        lim_u = np.sqrt(2)
        
        self.shape = shape
        # 初始化可学习参数，直接作为类属性（而非局部变量），确保梯度追踪
        self._w = nn.Parameter(torch.empty(shape)[None], requires_grad=True).to(device=global_device)
        nn.init.uniform_(self._w, -lim_w, lim_w)
        
        self._u = nn.Parameter(torch.empty(shape)[None], requires_grad=True).to(device=global_device)
        nn.init.uniform_(self._u, -lim_u, lim_u)

        self._b = nn.Parameter(torch.zeros(1), requires_grad=True).to(device=global_device)

        # 移除：不再在build中计算u_hat，避免参数版本固化
        # 移除：self._u_hat相关赋值

    def __init__(self, shape=None):
        """
        Construct a new :class:`PlanarNormalizingFlow`.

        Args:
            w_initializer: The initializer for parameter `w`.
            w_regularizer: The regularizer for parameter `w`.
            b_regularizer: The regularizer for parameter `b`.
            b_initializer: The initializer for parameter `b`.
            u_regularizer: The regularizer for parameter `u`.
            u_initializer: The initializer for parameter `u`.
            trainable (bool): Whether or not the parameters are trainable?
                (default :obj:`True`)
        """
        super(PlanarNormalizingFlow, self).__init__()
        self.shape = shape
        self.build(shape=self.shape)
        # self.apply(init_a_layer)  # 保留原初始化函数（若需要可启用）
    
    def reset_parameters(self):
        self.build(shape=self.shape)

    def forward(self, input):
        x, log_det_previous = input
        
        # 1. 动态计算u_hat（核心修改：每次前向都基于当前_w/_u计算，不修改原始参数）
        wu = self._w.matmul(self._u.transpose(-1, -2))  # 基于当前参数计算w·u
        softplus_wu = nn.Softplus()(wu)  # 非原地操作，生成新张量
        term = (-1 + softplus_wu - wu) * self._w  # 分子部分，不修改_w
        # 分母加1e-8避免除零，同时用keepdim=True确保维度匹配（防止广播错误）        
        # dim=1 表示在特征维度（z_dim）上求和，keepdim=True 保持 batch 维度（1）
        denominator = torch.sum(torch.square(self._w), dim=1, keepdim=True) + 1e-8
        
        u_hat = self._u + term / denominator  # 生成新的u_hat，不修改_u

        # 2. 原前向传播逻辑（仅将self._u_hat替换为动态计算的u_hat）
        wxb = torch.matmul(x, self._w.transpose(-1, -2)) + self._b  # shape == [?, 1]
        tanh_wxb = torch.tanh(wxb)  # shape == [?, 1]

        # 计算y = f(x)
        y = x + u_hat * tanh_wxb  # 用动态u_hat，不依赖固化的类属性

        # 计算log(det|df/dz|)
        grad = 1. - torch.square(tanh_wxb)  # dtanh(x)/dx = 1 - tanh^2(x)
        phi = grad * self._w  # shape == [?, n_units]
        u_phi = torch.matmul(phi, u_hat.transpose(-1, -2))  # 用动态u_hat
        det_jac = 1. + u_phi  # shape == [?, 1]
        log_det = torch.log(torch.abs(det_jac))  # shape == [?, 1]

        return y, log_det