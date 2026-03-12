import numpy as np
import torch
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

        g_band, r_band, z_band = image_np[0], image_np[1], image_np[2]
        rgb_image = Transformations.channels_to_rgb(r_band, g_band, z_band, stretch=self.stretch, Q=self.Q)
        return torch.from_numpy(rgb_image)

class UnsharpMaskTransform:
    def __init__(self, sigma=1.0, amount=1.0, threshold=0):
        self.sigma = sigma
        self.amount = amount
        self.threshold = threshold
    def __call__(self, image):
        
        g_band = image[1]
        r_band = image[0]
        z_band = image[2]

        _, sharpened_r = Transformations.unsharp_mask(r_band, sigma=self.sigma, amount=self.amount, threshold=self.threshold)
        _, sharpened_g = Transformations.unsharp_mask(g_band, sigma=self.sigma, amount=self.amount, threshold=self.threshold)
        _, sharpened_z = Transformations.unsharp_mask(z_band, sigma=self.sigma, amount=self.amount, threshold=self.threshold)

        #combine sharpened channels back into a single ndarray of shape (3, H, W):
        output = np.stack([sharpened_r, sharpened_g, sharpened_z], axis=0)

        return output
    
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
        self.amounts = amounts
        self.thresholds = thresholds
        self.clip_percentiles = clip_percentiles
        self.z_amount_boost = z_amount_boost

    def _apply_multiscale(self, band2d, amount_boost=0.0):
        # base band
        #cast tensor to float32 to avoid UFuncTypeError in gaussian_filter
        base = band2d.astype(np.float32, copy=False) if not isinstance(band2d, torch.Tensor) else band2d.detach().cpu().numpy().astype(np.float32, copy=False)

        # build multi-scale detail sum
        detail_sum = np.zeros_like(base, dtype=np.float32)
        for sigma, amount, thr in zip(self.sigmas, self.amounts, self.thresholds):
            _, sharp = Transformations.unsharp_mask(base, sigma=sigma, amount=amount + amount_boost, threshold=thr)
            detail_sum += (sharp - base)  # this is the (thresholded) detail mask at that scale

        out = base + detail_sum

        # gentle clipping to avoid extreme halos/outliers (optional but helps “not damaging too much”)
        lo, hi = np.percentile(out, self.clip_percentiles)
        out = np.clip(out, lo, hi)
        return out

    def __call__(self, image):
        # your convention: image is (3, H, W) with r,g,z (based on your existing class)
        r = image[0]
        g = image[1]
        z = image[2]

        r2 = self._apply_multiscale(r, amount_boost=0.0)
        g2 = self._apply_multiscale(g, amount_boost=0.0)
        z2 = self._apply_multiscale(z, amount_boost=self.z_amount_boost)

        return np.stack([r2, g2, z2], axis=0)

class SkySubstractTransform:
    def __call__(self, image):
        g_band = image[1]
        r_band = image[0]
        z_band = image[2]

        # Estimate sky background using median of the image
        sky_g = np.median(g_band)
        sky_r = np.median(r_band)
        sky_z = np.median(z_band)

        # Subtract sky background from each channel
        subtracted_g = g_band - sky_g
        subtracted_r = r_band - sky_r
        subtracted_z = z_band - sky_z   

        # Combine subtracted channels back into a single ndarray of shape (3, H, W):
        output = np.stack([subtracted_r, subtracted_g, subtracted_z], axis=0)

        return output

class EnsureCHWTransform:
    def __call__(self, image):
        return image.permute(2, 0, 1) if isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[-1] == 3 else image

#Apply dynamic rescaling to FITS band data by using asinh
class AsinhRescaleTransform:
    def __call__(self, image):
        # Apply arcsinh scaling to each channel
        rescaled_channels = []
        for i in range(image.shape[0]):
            band = image[i]
            band = np.nan_to_num(band)  # Handle NaNs
            band = np.arcsinh(band)  # Apply arcsinh scaling
            # Normalize to [0, 1]
            d_min, d_max = band.min(), band.max()
            if d_max > d_min:
                band = (band - d_min) / (d_max - d_min)
            rescaled_channels.append(band.astype(np.float32))
        
        return np.stack(rescaled_channels, axis=0)
