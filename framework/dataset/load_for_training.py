import logging
from typing import List, Optional
from functools import partial
import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)

SPECTRAL_INDICES = {
    "MSI": lambda bands: (bands["B8A"] - bands["B11"]) / (bands["B8A"] + bands["B11"] + 1e-6),
    "NDSI": lambda bands: (bands["B03"] - bands["B11"]) / (bands["B03"] + bands["B11"] + 1e-6),
    "NDWI": lambda bands: (bands["B03"] - bands["B08"]) / (bands["B03"] + bands["B08"] + 1e-6),
    "NDVI": lambda bands: (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + 1e-6),
    "GNDVI": lambda bands: (bands["B08"] - bands["B03"]) / (bands["B08"] + bands["B03"] + 1e-6),
    "BNDVI": lambda bands: (bands["B08"] - bands["B02"]) / (bands["B08"] + bands["B02"] + 1e-6),
    "NDRE": lambda bands: (bands["B08"] - bands["B05"]) / (bands["B08"] + bands["B05"] + 1e-6),
    "NDCI": lambda bands: (bands["B05"] - bands["B04"]) / (bands["B05"] + bands["B04"] + 1e-6),
    "NBR": lambda bands: (bands["B08"] - bands["B12"]) / (bands["B08"] + bands["B12"] + 1e-6),
    "EVI": lambda bands, G=2.5, C1=6.0, C2=7.5, L=1.0: G * (bands["B08"] - bands["B04"]) / (
            bands["B08"] + C1 * bands["B04"] - C2 * bands["B02"] + L + 1e-6),
    "SAVI": lambda bands, L=0.5: ((bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + L + 1e-6)) * (1 + L),
    "NDMI": lambda bands: (bands["B08"] - bands["B11"]) / (bands["B08"] + bands["B11"] + 1e-6),
    "ARI": lambda bands: (1 / (bands["B03"] + 1e-6)) - (1 / (bands["B05"] + 1e-6)),
    "BUI": lambda bands, L=0.5: ((bands["B11"] - bands["B08"]) / (bands["B11"] + bands["B08"] + 1e-6) - (
            ((bands["B08"] - bands["B04"]) * (1 + L)) / (bands["B08"] + bands["B04"] + L + 1e-6) + (
            bands["B03"] - bands["B11"]) / (bands["B03"] + bands["B11"] + 1e-6)) / 2)
}

BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

euroSAT_classes = {"AnnualCrop": 0, "Forest": 1, "HerbaceousVegetation": 2, "Highway": 3, "Industrial": 4, "Pasture": 5,
                   "PermanentCrop": 6, "Residential": 7, "River": 8, "SeaLake": 9}

_VALID_BANDS = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}
_VALID_INDICES = set(SPECTRAL_INDICES.keys())


def load_eurosat_hf_splits(dataset_hf_repo_id: str) -> tuple[Dataset | None, Dataset | None, Dataset | None]:
    """
    This function loads the EuroSAT HF splits dataset.
    It returns the train, val and test splits.
    """
    log.info(f"Loading Dataset: {dataset_hf_repo_id}")
    ds = load_dataset(dataset_hf_repo_id)
    train_ds = ds.get("train", None)
    val_ds = ds.get("validation", None) or ds.get("val", None)
    test_ds = ds.get("test", None)
    log.info("Finished loading EuroSAT HF splits")
    return train_ds, val_ds, test_ds


def _pick_image(sample: dict):
    """
    This function picks an image from the sample.
    """
    if "image" in sample:
        return np.array(sample["image"])
    raise KeyError("Could not find image array in sample.")


def _pick_label(sample: dict):
    """
    This function loads the label of the given sample.
    """
    if "label" in sample:
        return int(sample["label"])
    raise KeyError("Could not find label in sample.")


def _ensure_chw(cube: np.ndarray) -> np.ndarray:
    """
    Ensures that the input cube has a CHW dimension and 12 Sentinel-2 bands (B10 dropped).
    The order after this function is BAND_ORDER.
    """
    cube = np.asarray(cube)
    if cube.ndim == 2:
        cube = cube[None, ...]
    if cube.shape[0] not in (12, 13) and cube.shape[-1] in (12, 13):
        cube = np.moveaxis(cube, -1, 0)
    if cube.shape[0] == 13:  # drop B10
        cube = np.concatenate([cube[:9], cube[10:]], axis=0)
    return cube.astype(np.float32)


def _bands_dict(cube: np.ndarray) -> dict:
    """
    Build a {band_name: 2D array} dict from the 12-channel cube in BAND_ORDER.
    """
    return {b: cube[i] for i, b in enumerate(BAND_ORDER[:cube.shape[0]])}


def _select_bands(cube: np.ndarray, selected_bands: Optional[List[str]]) -> np.ndarray:
    """
    This function selects all bands specified by the selected_bands argument.
    """
    if not selected_bands:
        return cube  # keep all 12 in BAND_ORDER
    for b in selected_bands:
        if b not in _VALID_BANDS:
            raise ValueError(f"Unknown band '{b}'. Valid: {sorted(_VALID_BANDS)}")
    # map band -> index in BAND_ORDER
    idxs = [BAND_ORDER.index(b) for b in selected_bands]
    return cube[idxs, :, :].astype(np.float32)


def _compute_indices_only(cube: np.ndarray, indices: List[str]) -> Optional[np.ndarray]:
    """
    This function computes indices only for a given set of bands.
    """
    if not indices:
        return None
    for name in indices:
        if name not in _VALID_INDICES:
            raise ValueError(f"Unknown index '{name}'. Valid: {sorted(_VALID_INDICES)}")
    bands = _bands_dict(cube)  # compute indices from the FULL base cube (all available bands)
    idx_maps = [SPECTRAL_INDICES[name](bands) for name in indices]
    return np.stack(idx_maps, axis=0).astype(np.float32)


def _assemble_channels(cube: np.ndarray,
                       selected_bands: Optional[List[str]],
                       indices: Optional[List[str]]) -> np.ndarray:
    """
    Returns channels in the order:
      selected_bands (or all 12 if None/empty) + indices (in the given order).
    Indices are computed from the full base cube so they don't depend on selection.
    """
    base = _select_bands(cube, selected_bands)
    extra = _compute_indices_only(cube, indices or [])
    if extra is not None:
        return np.concatenate([base, extra], axis=0)
    return base


def _preprocess_sample(sample: dict, image_size: int,
                       indices: Optional[List[str]], augment: bool,
                       mean=None, std=None,
                       selected_bands: Optional[List[str]] = None):
    """
    This function preprocesses the given sample. If augment is enabled it will be applied.
    """
    cube = _ensure_chw(_pick_image(sample))  # full 12 base bands in BAND_ORDER
    cube = _assemble_channels(cube, selected_bands, indices)  # apply selection + indices
    x = torch.from_numpy(cube).float()
    x = TF.resize(x, [image_size, image_size])
    if (mean is not None) and (std is not None):
        x = (x - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    if augment:
        if torch.rand(1).item() < 0.5:
            x = TF.hflip(x)
        if torch.rand(1).item() < 0.5:
            x = TF.vflip(x)
    y = _pick_label(sample)
    return x, torch.tensor(y, dtype=torch.long)


def _collate_hf(ds, batch_indices, image_size: int,
                indices: Optional[List[str]], augment: bool,
                mean, std, selected_bands: Optional[List[str]]):
    """
    This function collates the dataset into batches and preprocesses the images.
    """
    xs, ys = [], []
    for i in batch_indices:
        x, y = _preprocess_sample(
            ds[int(i)], image_size, indices, augment, mean, std, selected_bands
        )
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def _calculate_mean_std(loader,desc):
    """
    Calculates the mean and std of the given dataset.
    """
    channel_sum, channel_sq_sum, num = 0, 0, 0
    for images, _ in tqdm(loader, desc=f"Calculating mean and std for {len(loader)} batches for {desc}"):
        channel_sum = channel_sum + images.sum(dim=[0, 2, 3])
        channel_sq_sum = channel_sq_sum + (images ** 2).sum(dim=[0, 2, 3])
        num += images.size(0) * images.size(2) * images.size(3)
    mean = channel_sum / num
    std = (channel_sq_sum / num - mean ** 2).clamp(min=1e-6).sqrt()
    return mean, std


def build_loaders_from_hf(train_ds, val_ds, batch_size: int, image_size: int,
                          indices: Optional[List[str]],
                          augment: bool, normalize: bool = False,
                          norm_mean=None, norm_std=None,
                          stats_samples: int = 0,
                          selected_bands: Optional[List[str]] = None,
                          num_workers: int = 4,
                          pin_memory: bool = False):
    """
    selected_bands: list of raw bands to include (order respected). If None/empty, keeps all 12.
    indices: list of spectral index names to append as extra channels (order respected).
    """
    if normalize and (norm_mean is None or norm_std is None):
        log.info("Start normalization.")
        sample_range = range(len(train_ds) if stats_samples <= 0 else min(stats_samples, len(train_ds)))

        tmp_collate = partial(
            _collate_hf,
            train_ds,
            image_size=image_size,
            indices=indices,
            augment=False,
            mean=None,
            std=None,
            selected_bands=selected_bands,
        )

        tmp_loader = DataLoader(
            sample_range,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=tmp_collate,
            num_workers=num_workers,
            pin_memory=False,
        )

        norm_mean, norm_std = _calculate_mean_std(tmp_loader, desc="Training Dataset before Normalization")
        log.info(
            f"Before normalization --> "
            f"norm_mean: {norm_mean.tolist()!r}\n"
            f"norm_std: {norm_std.tolist()!r}"
        )

    train_collate = partial(
        _collate_hf,
        train_ds,
        image_size=image_size,
        indices=indices,
        augment=augment,
        mean=(norm_mean if normalize else None),
        std=(norm_std if normalize else None),
        selected_bands=selected_bands,
    )

    train_loader = DataLoader(
        range(len(train_ds)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    # Log after normalization only in DEBUG mode because it is time intensive
    if logging.getLevelName(log.getEffectiveLevel()) == "DEBUG":
        mean_after_normalization, std_after_normalization = _calculate_mean_std(train_loader, desc="Training Dataset after Normalization")
        log.info(f"After normalization (Train) --> "
                 f"norm_mean: {mean_after_normalization.tolist()!r}\n"
                 f"norm_std: {std_after_normalization.tolist()!r}"
                 )

    val_loader = None
    if val_ds is not None:
        val_collate = partial(
            _collate_hf,
            val_ds,
            image_size=image_size,
            indices=indices,
            augment=False,
            mean=(norm_mean if normalize else None),
            std=(norm_std if normalize else None),
            selected_bands=selected_bands,
        )
        val_loader = DataLoader(
            range(len(val_ds)),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )

    # Log after normalization only in DEBUG mode because it is time intensive
    if logging.getLevelName(log.getEffectiveLevel()) == "DEBUG":
        # mean and std of validation dataset are not perfectly normalized since the used std and mean is only as good as the train dataset generalizes
        mean_after_normalization, std_after_normalization = _calculate_mean_std(val_loader, desc="Validation Dataset after Normalization")
        log.info(f"After normalization (Validation) --> "
                 f"norm_mean: {mean_after_normalization.tolist()!r}\n"
                 f"norm_std: {std_after_normalization.tolist()!r}"
                 )

    return train_loader, val_loader


def _mixup_data(x, y, alpha: float = 1.0):
    if alpha > 0.:
        lam = torch._standard_gamma(torch.tensor([alpha])).item()
        lam = lam / (lam + torch._standard_gamma(torch.tensor([alpha])).item())
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
