"""
Diffusion process: noise schedule, forward noising, and DDIM/DDPM sampling.

The model uses a standard linear beta schedule over 1000 steps.
Inference uses DDIM (deterministic at eta=0, stochastic at eta=1).
Training uses a short DDIM chain that returns proxy losses from the first step.

EMA helper is also included here for stabilizing training.
"""

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow model whose weights are a running average of the
    main model's weights. Reduces noise in gradient updates.
    """

    def __init__(self, beta: float = 0.9999):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model: torch.nn.Module,
                              current_model: torch.nn.Module):
        for current_params, ema_params in zip(
            current_model.parameters(), ema_model.parameters()
        ):
            ema_params.data = self._update(ema_params.data, current_params.data)

    def _update(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: torch.nn.Module, model: torch.nn.Module,
                 step_start: int = 2000):
        """
        Update the EMA model. Before step_start, simply copy weights.
        """
        if self.step < step_start:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model: torch.nn.Module, model: torch.nn.Module):
        ema_model.load_state_dict(model.state_dict())


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------

class Diffusion:
    """
    Latent diffusion process with linear noise schedule.

    Args:
        noise_steps:  Total number of diffusion steps T (default 1000)
        noise_offset: Optional structured noise offset for diversity
        beta_start:   Starting beta value
        beta_end:     Ending beta value
        device:       Torch device for schedule tensors
    """

    def __init__(self, noise_steps: int = 1000, noise_offset: float = 0.0,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 device=None):
        self.noise_steps = noise_steps
        self.noise_offset = noise_offset
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def noise_images(self, x: torch.Tensor, t: torch.Tensor):
        """
        Add noise to x at timestep t.

        Returns:
            x_noisy: noised input
            eps:     the noise that was added
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        if self.noise_offset > 0:
            eps = eps + self.noise_offset * torch.randn(
                x.shape[0], x.shape[1], 1, 1, device=x.device
            )
        return sqrt_alpha_hat * x + sqrt_one_minus * eps, eps

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=0, high=self.noise_steps, size=(n,))

    # ------------------------------------------------------------------
    # DDIM update step
    # ------------------------------------------------------------------

    @staticmethod
    def _ddim_step(x: torch.Tensor, predicted_noise: torch.Tensor,
                   alpha_hat: torch.Tensor, alpha_hat_next: torch.Tensor,
                   beta: torch.Tensor, eta: float) -> torch.Tensor:
        """Apply one DDIM reverse step from x_t to x_{t-1}."""
        x_start = (x - (1 - alpha_hat).sqrt() * predicted_noise) / alpha_hat.sqrt()
        sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
        c = (1 - alpha_hat_next - sigma ** 2).sqrt()
        noise = torch.randn_like(x)
        return x_start * alpha_hat_next.sqrt() + c * predicted_noise + sigma * noise

    # ------------------------------------------------------------------
    # Training-time DDIM (returns proxy losses from first step)
    # ------------------------------------------------------------------

    def train_ddim(self, model, x: torch.Tensor, styles: torch.Tensor,
                   content: torch.Tensor, total_t: int, wid: torch.Tensor,
                   sampling_timesteps: int = 6, eta: float = 0.0):
        """
        Short DDIM chain used during training.

        The first denoising step runs with tag='train' to obtain proxy losses.
        Subsequent steps run without proxy loss computation.

        Returns:
            x_denoised, first_step_noise_pred, vertical_proxy_loss, horizontal_proxy_loss
        """
        times = torch.linspace(-1, total_t - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_start = None
        first_noise = None
        v_loss = h_loss = None

        for idx, (time, time_next) in enumerate(time_pairs):
            t = (torch.ones(x.shape[0]) * time).long().to(self.device)
            t_next = (torch.ones(x.shape[0]) * time_next).long().to(self.device)

            if idx == 0:
                noise_pred, v_loss, h_loss = model(x, t, styles, content, wid, tag="train")
                first_noise = noise_pred
            else:
                noise_pred = model(x, t, styles, content, tag="test")

            beta = self.beta[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]

            if time_next < 0:
                x = (x - (1 - alpha_hat).sqrt() * noise_pred) / alpha_hat.sqrt()
                break

            alpha_hat_next = self.alpha_hat[t_next][:, None, None, None]
            x = self._ddim_step(x, noise_pred, alpha_hat, alpha_hat_next, beta, eta)

        return x, first_noise, v_loss, h_loss

    # ------------------------------------------------------------------
    # Inference DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, model, vae, n: int, x: torch.Tensor,
                    styles: torch.Tensor, content: torch.Tensor,
                    sampling_timesteps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        Full DDIM sampling loop followed by VAE decode.

        Args:
            model:             UNetModel
            vae:               Frozen SD v1.5 AutoencoderKL
            n:                 Batch size
            x:                 Initial noise [n, 4, H/8, W/8]
            styles:            [n, 1, H, W] style images
            content:           [n, T, 16, 16] content glyphs
            sampling_timesteps: number of DDIM steps
            eta:               stochasticity (0=deterministic)

        Returns:
            images: [n, C, H, W] in [0, 1]
        """
        model.eval()

        times = torch.linspace(-1, self.noise_steps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            t = (torch.ones(n) * time).long().to(self.device)
            t_next = (torch.ones(n) * time_next).long().to(self.device)

            noise_pred = model(x, t, styles, content, tag="test")

            beta = self.beta[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]

            if time_next < 0:
                x = (x - (1 - alpha_hat).sqrt() * noise_pred) / alpha_hat.sqrt()
                break

            alpha_hat_next = self.alpha_hat[t_next][:, None, None, None]
            x = self._ddim_step(x, noise_pred, alpha_hat, alpha_hat_next, beta, eta)

        model.train()

        # Decode latents with VAE
        latents = x / 0.18215
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).contiguous()
        image = torch.from_numpy(image.numpy()).permute(0, 3, 1, 2).contiguous()
        return image

    # ------------------------------------------------------------------
    # DDPM sampling (slower, full 1000-step chain)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddpm_sample(self, model, vae, n: int, x: torch.Tensor,
                    styles: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """Full DDPM reverse process. Slower than DDIM but equivalent in expectation."""
        model.eval()

        for i in reversed(range(0, self.noise_steps)):
            t = (torch.ones(n) * i).long().to(self.device)
            noise_pred = model(x, t, styles, content, tag="test")

            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1 / alpha.sqrt()) * (
                x - ((1 - alpha) / (1 - alpha_hat).sqrt()) * noise_pred
            ) + beta.sqrt() * noise

        model.train()

        latents = x / 0.18215
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).contiguous()
        image = torch.from_numpy(image.numpy()).permute(0, 3, 1, 2).contiguous()
        return image
