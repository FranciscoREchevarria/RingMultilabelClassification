"""
RingDetectionZoobot: Custom PyTorch Lightning module for multilabel ring classification.

This module provides a fine-tunable model for detecting inner and outer rings in galaxy images
using a pretrained Zoobot encoder.
"""

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import Accuracy, F1Score, FBetaScore, Precision, Recall, HammingDistance


def tune_thresholds_on_val(
    model,
    val_dataloader,
    device,
    threshold_range=(0.2, 0.9),
    step=0.02,
    metric="f2",
):
    """
    Optimize inner and outer ring thresholds independently (per-label) on validation set.

    Each label's threshold is tuned separately to maximize the chosen metric for that label,
    allowing different recall/precision tradeoffs for inner vs outer rings.

    Args:
        model: RingDetectionZoobot (or any with predict_proba and inner/outer_ring_threshold).
        val_dataloader: DataLoader yielding batches with 'image' and 'ring_class'.
        device: torch device.
        threshold_range: (low, high) for threshold grid.
        step: Grid step size.
        metric: Optimization objective per label. 'f2' (recall-weighted) or 'recall'.

    Returns:
        (best_t_inner, best_t_outer), and sets model.inner_ring_threshold, model.outer_ring_threshold.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            x = batch['image'].to(device)
            y = batch['ring_class'].to(device)
            probs = model.predict_proba(x)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    low, high = threshold_range
    steps = max(1, int((high - low) / step) + 1)
    thresholds = [
        low + k * (high - low) / max(1, steps - 1)
        for k in range(steps)
    ]

    if metric == "recall":
        rec_metric = Recall(task='multilabel', num_labels=2, average='none')
    elif metric == "f2":
        # Default: f2 (weights recall 2x precision)
        rec_metric = FBetaScore(task='multilabel', num_labels=2, average='none', beta=2.0)
    elif metric == "f1":
        rec_metric = F1Score(task='multilabel', num_labels=2, average='none')
    elif metric == "precision":
        rec_metric = Precision(task='multilabel', num_labels=2, average='none')
    elif metric == "hamming":
        rec_metric = HammingDistance(task='multilabel', num_labels=2, average='none')
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    best_t_inner, best_t_outer = 0.5, 0.5

    # Optimize inner ring threshold (label 0) holding outer at 0.5
    best_score_inner = -1.0
    for t in thresholds:
        preds = (probs > torch.tensor([t, 0.5])).float()
        rec_metric.reset()
        score = rec_metric(preds, labels)[0].item()
        if score > best_score_inner:
            best_score_inner = score
            best_t_inner = t

    # Optimize outer ring threshold (label 1) using best_t_inner
    best_score_outer = -1.0
    for t in thresholds:
        preds = (probs > torch.tensor([best_t_inner, t])).float()
        rec_metric.reset()
        score = rec_metric(preds, labels)[1].item()
        if score > best_score_outer:
            best_score_outer = score
            best_t_outer = t

    model.inner_ring_threshold = best_t_inner
    model.outer_ring_threshold = best_t_outer
    return best_t_inner, best_t_outer


def _focal_bce_loss(logits, targets, pos_weight=None, gamma=2.0):
    """Focal loss for multilabel: (1 - pt)^gamma * BCE, with optional pos_weight per label."""
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    focal_weight = (1 - pt).clamp(min=1e-6).pow(gamma)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = focal_weight * bce
    if pos_weight is not None:
        pw = pos_weight.to(logits.device).expand_as(targets)
        loss = loss * torch.where(targets == 1, pw, 1.0)
    return loss.mean()


class RingDetectionZoobot(pl.LightningModule):
    def __init__(
        self,
        encoder,
        encoder_dim: int = 512,
        hidden_dim: int = 256,
        dropout_rate: float = 0.4,
        encoder_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.0,
        pos_weight=None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        use_head_batchnorm: bool = False,
        **kwargs,
    ):
        """
        Custom Zoobot model for multilabel ring classification.
        
        Args:
            encoder: Pretrained Zoobot encoder (already instantiated).
            encoder_dim: Dimension of encoder feature output.
            hidden_dim: Hidden layer dimension in the classification head.
            dropout_rate: Dropout probability in the head.
            encoder_lr: Learning rate for encoder parameters during fine-tuning.
            head_lr: Learning rate for the classification head.
            weight_decay: Weight decay for the optimizer (AdamW).
            pos_weight: Optional tensor/array of shape [2] for BCEWithLogitsLoss
                        to handle class imbalance between inner/outer rings.
            use_focal_loss: If True, use focal loss instead of BCE (for hard/rare positives).
            focal_gamma: Gamma for focal loss (default 2.0).
            use_head_batchnorm: If True, add BatchNorm1d after first linear in head.
        """
        super().__init__()

        self.inner_ring_threshold = 0.5
        self.outer_ring_threshold = 0.5
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self._pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None

        # Save lightweight hyperparameters for reproducibility / checkpoints
        self.save_hyperparameters(ignore=["encoder"])

        # Store encoder without calling parent init
        self.encoder = encoder

        # Define classification head (optionally with BatchNorm)
        head_layers = [
            nn.Dropout(p=dropout_rate),
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
        ]
        if use_head_batchnorm:
            head_layers.append(nn.BatchNorm1d(hidden_dim))
        head_layers.extend([
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(hidden_dim, 2),
        ])
        self.head = nn.Sequential(*head_layers)

        # Loss and metrics
        if not use_focal_loss and pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight_tensor)
        elif not use_focal_loss:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = None  # use _focal_bce_loss in steps

        # Core metrics
        self.train_accuracy = Accuracy(task='multilabel', num_labels=2)
        self.val_accuracy = Accuracy(task='multilabel', num_labels=2)

        # More informative multilabel metrics (macro over labels)
        self.train_f1_macro = F1Score(task='multilabel', num_labels=2, average='macro')
        self.val_f1_macro = F1Score(task='multilabel', num_labels=2, average='macro')
        self.train_precision_macro = Precision(task='multilabel', num_labels=2, average='macro')
        self.val_precision_macro = Precision(task='multilabel', num_labels=2, average='macro')
        self.train_recall_macro = Recall(task='multilabel', num_labels=2, average='macro')
        self.val_recall_macro = Recall(task='multilabel', num_labels=2, average='macro')
        

        # Optimizer hyperparameters
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay

        # Track freeze state
        self.encoder_frozen = True
        self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
        print("✓ Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
        print("✓ Encoder unfrozen")
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['ring_class']
        logits = self.forward(x)
        if self.loss_fn is not None:
            loss = self.loss_fn(logits, y)
        else:
            loss = _focal_bce_loss(logits, y, self._pos_weight_tensor, self.focal_gamma)
        
        #prediction threshold is performed by class, allowing for different thresholds for inner vs outer ring detection if desired
        threshold = torch.tensor([self.inner_ring_threshold, self.outer_ring_threshold], device=logits.device)

        preds = (logits.sigmoid() > threshold).float()
        acc = self.train_accuracy(preds, y)
        f1 = self.train_f1_macro(preds, y)
        prec = self.train_precision_macro(preds, y)
        rec = self.train_recall_macro(preds, y)

        self.log('finetuning/train_loss', loss, on_epoch=True)
        self.log('finetuning/train_acc', acc, on_epoch=True)
        self.log('finetuning/train_f1_macro', f1, on_epoch=True)
        self.log('finetuning/train_precision_macro', prec, on_epoch=True)
        self.log('finetuning/train_recall_macro', rec, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['ring_class']
        logits = self.forward(x)
        if self.loss_fn is not None:
            loss = self.loss_fn(logits, y)
        else:
            loss = _focal_bce_loss(logits, y, self._pos_weight_tensor, self.focal_gamma)
        
        #prediction threshold is performed by class, allowing for different thresholds for inner vs outer ring detection if desired
        threshold = torch.tensor([self.inner_ring_threshold, self.outer_ring_threshold], device=logits.device)

        preds = (logits.sigmoid() > threshold).float()
        acc = self.val_accuracy(preds, y)
        f1 = self.val_f1_macro(preds, y)
        prec = self.val_precision_macro(preds, y)
        rec = self.val_recall_macro(preds, y)

        self.log('finetuning/val_loss', loss, on_epoch=True)
        self.log('finetuning/val_acc', acc, on_epoch=True)
        self.log('finetuning/val_f1_macro', f1, on_epoch=True)
        self.log('finetuning/val_precision_macro', prec, on_epoch=True)
        self.log('finetuning/val_recall_macro', rec, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with separate parameter groups for head/encoder."""
        # Head parameters are always trainable
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        # Encoder parameters may be frozen during stage 1
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]

        if not head_params and not encoder_params:
            raise ValueError("No trainable parameters found!")

        param_groups = []

        if head_params:
            param_groups.append(
                {"params": head_params, "lr": self.head_lr}
            )

        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": self.encoder_lr}
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'finetuning/val_loss',
            },
        }
    
    def predict(self, x, threshold=0.5):
        probs = self.forward(x).sigmoid()
        return (probs > threshold).float()

    def predict_proba(self, x):
        """
        Get probability predictions for binary multilabel classification.
        
        Args:
            x: Input image tensor
            
        Returns:
            Probability tensor of shape (batch_size, 2) with values in [0, 1]
        """
        logits = self.forward(x)
        probabilities = logits.sigmoid()
        return probabilities

    def predict_proba_tta(self, x, n_rotations=4, flip=True):
        """
        Test-time augmentation: average predictions over rotations and optional flips.
        Views: 0°, 90°, 180°, 270° and, if flip=True, their horizontal flips (8 views).
        """
        self.eval()
        probs_list = []
        with torch.no_grad():
            for k in range(n_rotations):
                x_rot = torch.rot90(x, k=k, dims=(-2, -1))
                probs_list.append(self.predict_proba(x_rot))
            if flip:
                for k in range(n_rotations):
                    x_rot = torch.rot90(x, k=k, dims=(-2, -1))
                    x_flip = torch.flip(x_rot, dims=[-1])  # horizontal
                    probs_list.append(self.predict_proba(x_flip))
        return torch.stack(probs_list, dim=0).mean(dim=0)

    def batch_to_supervised_tuple(self, batch):
        """Convert batch dictionary to (x, y) tuple for training."""
        # Your dataset returns {'image': tensor, 'ring_class': tensor}
        x = batch['image']
        y = batch['ring_class']
        return x, y
