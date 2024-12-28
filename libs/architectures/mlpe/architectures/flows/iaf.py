from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import random
import torch.distributions as dist
from lightning import pytorch as pl
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import ConditionalAffineAutoregressive
from pyro.nn import ConditionalAutoRegressiveNN
import torch.nn.utils as nn_utils
from mlpe.architectures.flows import utils
from mlpe.architectures.flows.flow import NormalizingFlow

class InverseAutoRegressiveFlow(pl.LightningModule, NormalizingFlow):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        embedding_net: torch.nn.Module,
        opt: torch.optim.SGD,
        sched: torch.optim.lr_scheduler.ConstantLR,
        inference_params: list,
        num_samples_draw: int = 3000,
        num_plot_corner: int = 20,
        hidden_features: int = 50,
        num_transforms: int = 5,
        num_blocks: int = 2,
        activation: Callable = torch.tanh,
        base_epsilon: float = 0.05,  # Base epsilon for adaptive training
        c: float = 0.000001,  # Hardness coefficient for adaptive step size
        beta: float = 0.5  # Momentum for gradient norm
    ):
        super().__init__()
        self.automatic_optimization = False
        self.param_dim, self.n_ifos, self.strain_dim = shape
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.activation = activation
        self.optimizer = opt
        self.scheduler = sched
        self.inference_params = inference_params
        self.num_samples_draw = num_samples_draw
        self.num_plot_corner = num_plot_corner
        self.base_epsilon = base_epsilon
        self.c = c
        self.beta = beta
        self.gdnorms = None  # To store gradient norms across batches
        # define embedding net and base distribution
        self.embedding_net = embedding_net
        # build the transform - sets the transforms attrib
        self.build_flow()

    def transform_block(self):
        """Returns single autoregressive transform"""
        arn = ConditionalAutoRegressiveNN(
            self.param_dim,
            self.context_dim,
            self.num_blocks * [self.hidden_features],
            nonlinearity=self.activation,
        )
        transform = ConditionalAffineAutoregressive(arn).inv
        return transform

    def distribution(self):
        """Returns the base distribution for the flow"""
        return dist.Normal(
            torch.zeros(self.param_dim, device=self.device),
            torch.ones(self.param_dim, device=self.device),
        )

    def build_flow(self):
        """Build the transform"""
        self.transforms = []
        for idx in range(self.num_transforms):
            _transform = self.transform_block()
            self.transforms.extend([_transform])
        self.transforms = ConditionalComposeTransformModule(self.transforms)

    def fgsm_attack(self, data, epsilon, data_grad):
        """
        FGSM attack: generate adversarial examples
        """
        random_noise = torch.randn_like(data) * 0.001  # 可以调整随机噪声大小
        data = data + random_noise
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon * sign_data_grad
        return perturbed_data

    def adaptive_epsilon(self, grad_norm):
        """
        根据梯度范数自适应调整 epsilon，梯度越大 epsilon 越小，控制 epsilon 在 [0.01, 0.1] 之间
        使用对数缩放来处理梯度的多个数量级差异
        """
        epsilon_min = 0.01
        epsilon_max = 0.1
    
        # 初始化或更新 grad_min 和 grad_max，使用 momentum 平滑
        if not hasattr(self, 'grad_min') or not hasattr(self, 'grad_max'):
            self.grad_min = grad_norm.min()  # 初始时设置为当前 batch 的最小值
            self.grad_max = grad_norm.max()  # 初始时设置为当前 batch 的最大值
        else:
            momentum = 0.99  # 动态更新的滑动平均系数
            self.grad_min = momentum * self.grad_min + (1 - momentum) * grad_norm.min()
            self.grad_max = momentum * self.grad_max + (1 - momentum) * grad_norm.max()
    
        # 对 grad_norm 进行对数变换，以处理梯度的数量级问题
        log_grad_min = torch.log(self.grad_min + 1e-12)  # 防止 log(0)
        log_grad_max = torch.log(self.grad_max + 1e-12)
        log_grad_norm = torch.log(grad_norm + 1e-12)
    
        # 使用对数缩放公式，将梯度大的样本对应较小的epsilon，梯度小的样本对应较大的epsilon
        scaled_epsilon = epsilon_min + ((log_grad_max - log_grad_norm) / (log_grad_max - log_grad_min)) * (epsilon_max - epsilon_min)
        
        # 确保 epsilon 值在 [epsilon_min, epsilon_max] 范围内
        scaled_epsilon = torch.clamp(scaled_epsilon, epsilon_min, epsilon_max)
        
        return scaled_epsilon

    def training_step(self, batch, batch_idx):
        strain, parameters = batch

        # Ensure strain requires gradient computation
        strain.requires_grad = True

        # Normal forward pass and loss calculation
        loss = -self.log_prob(parameters, context=strain).mean()

        # Get optimizer
        optimizer = self.optimizers()

        # Backpropagate and get the gradient of the input
        self.manual_backward(loss, retain_graph=True)
        strain_grad = strain.grad.data

        # Calculate the gradient norm (L2 norm)
        grad_norm = torch.norm(strain_grad.view(len(strain), -1), dim=1).detach() ** 2
        self.gdnorms = grad_norm
        # If using momentum for gradient norm
        #if self.gdnorms is None:
        #    self.gdnorms = grad_norm  # Initialize on the first iteration
        #else:
        #    self.gdnorms = (1 - self.beta) * grad_norm + self.beta * self.gdnorms

        # Calculate adaptive step size (epsilon) based on gradient norm
        adaptive_epsilons = self.adaptive_epsilon(self.gdnorms)

        # 实时打印 grad_norm 和 adaptive_epsilons，打印前 5 个样本
        #if batch_idx % 10 == 0:  # 每10个batch打印一次，避免过多输出
        #    print(f"Batch {batch_idx} Grad Norms: {grad_norm[:5]}")
        #    print(f"Batch {batch_idx} Adaptive Epsilons: {adaptive_epsilons[:5]}")

        # Apply FGSM attack with adaptive epsilon
        perturbed_strain = []
        for i in range(len(strain)):
            perturbed_strain.append(self.fgsm_attack(strain[i], adaptive_epsilons[i], strain_grad[i]))
        perturbed_strain = torch.stack(perturbed_strain)

        # Compute adversarial loss
        adv_loss = -self.log_prob(parameters, context=perturbed_strain).mean()

        # Combine original loss and adversarial loss
        total_loss = (loss + adv_loss) / 2

        # Reset gradients and perform backpropagation on the total loss
        optimizer.zero_grad()
        self.manual_backward(total_loss)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)

        # Perform the optimization step
        optimizer.step()

        # Log the training loss
        self.log("train_loss", total_loss, on_step=True, prog_bar=True, sync_dist=False)

        return total_loss

    def configure_optimizers(self):
        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.001)
        
        # 使用学习率余弦衰减调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
    def validation_step(self, batch, batch_idx):
        strain, parameters = batch
        loss = -self.log_prob(parameters, context=strain).mean()
        self.log(
            "valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def on_test_epoch_start(self):
        self.test_results = []
        self.num_plotted = 0

    def test_step(self, batch, batch_idx):
        strain, parameters = batch
        res = utils.draw_samples_from_model(
            strain,
            parameters,
            self,
            self.inference_params,
            self.num_samples_draw,
        )
        self.test_results.append(res)
        if batch_idx % 10 == 0 and self.num_plotted < self.num_plot_corner:
            skymap_filename = f"{self.num_plotted}_mollview.png"
            res.plot_corner(
                save=True,
                filename=f"{self.num_plotted}_corner.png",
                levels=(0.5, 0.9),
            )
            utils.plot_mollview(
                res.posterior["phi"] - torch.pi,  # between -pi to pi in healpy
                res.posterior["dec"],
                truth=(
                    res.injection_parameters["phi"] - torch.pi,
                    res.injection_parameters["dec"],
                ),
                outpath=skymap_filename,
            )
            self.num_plotted += 1
            self.print("Made corner plots and skymap for:", batch_idx)

    def on_test_epoch_end(self):
        import bilby

        bilby.result.make_pp_plot(
            self.test_results,
            save=True,
            filename="pp-plot.png",
            keys=self.inference_params,
        )
        del self.test_results, self.num_plotted
    
