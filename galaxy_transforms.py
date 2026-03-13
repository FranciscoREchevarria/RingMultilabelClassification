import math
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur
from visualizations import Transformations

class LuptonRgbTransform:
    def __init__(self, stretch=0.5, Q=10):
        self.stretch = stretch
        self.Q = Q
        
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            # Transformations.channels_to_rgb expects numpy arrays.
            image_np = image.detach().cpu().numpy()
        else:
            image_np = np.asarray(image)

        rgb_image = Transformations.channels_to_rgb(image_np, stretch=self.stretch, Q=self.Q)
        #enforce CHW and float32 output
        rgb_image = np.transpose(rgb_image, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(rgb_image)

class MultiScaleUnsharpMaskTransform:
    def __init__(
        self,
        sigmas=(2.5, 5.5),
        amounts=(1.0, 2.0),
        thresholds=(0.005, 0.002),
        clip_percentiles=(0.1, 99.9),
        z_amount_boost=0.0,
    ):
        self.sigmas = sigmas
        self.amounts = torch.tensor(amounts)
        self.thresholds = torch.tensor(thresholds)
        self.clip_percentiles = clip_percentiles
        self.z_amount_boost = z_amount_boost

    @torch.no_grad()
    def __call__(self, image):
        device = image.device

        boosts = torch.tensor([0.0, 0.0, self.z_amount_boost], device=device).view(3, 1, 1)
        detail_sum = torch.zeros_like(image)

        for sigma, amount, thr in zip(self.sigmas, self.amounts, self.thresholds):
            k_size = int(4 * sigma + 1)
            if k_size % 2 == 0:
                k_size += 1

            blurred = gaussian_blur(image, [k_size, k_size], [sigma, sigma])
            diff = image - blurred
            mask = torch.abs(diff) > thr
            curr_amount = amount + boosts
            detail_sum += diff * mask * curr_amount

        out = image + detail_sum

        # Percentile clipping via kthvalue — O(n) instead of O(n log n) quantile
        q_low = self.clip_percentiles[0] / 100.0
        q_high = self.clip_percentiles[1] / 100.0

        flat_out = out.view(3, -1)
        n = flat_out.shape[1]
        k_low = max(1, int(q_low * n))
        k_high = min(n, int(q_high * n))
        lo = torch.kthvalue(flat_out, k_low, dim=1, keepdim=True).values.view(3, 1, 1)
        hi = torch.kthvalue(flat_out, k_high, dim=1, keepdim=True).values.view(3, 1, 1)

        return torch.clamp(out, lo, hi)


class SkySubstractTransform:
    @torch.no_grad()
    def __call__(self, image):
        # image: (3, H, W) torch.Tensor
        sky = image.flatten(1).median(dim=1).values.view(-1, 1, 1)
        return image - sky

class EnsureCHWTransform:
    def __call__(self, image):
        return image.permute(2, 0, 1) if isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[-1] == 3 else image

class ScaleToUnitIntervalTransform:
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:  # Avoid division by zero
                return (image - min_val) / (max_val - min_val)
            else:
                return image  # If all values are the same, return the original image
        else:
            return image

