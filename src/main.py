from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Literal, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pyiqa
import SimpleITK as sitk
import torch
from PIL import Image
from skimage.filters import threshold_otsu

from constants import INPUT, TARGET, REPORT


def _to_normalized_channel_tensor(depth_first_array: np.ndarray) -> torch.Tensor:
    arr = depth_first_array.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(arr)
    return torch.from_numpy(arr).unsqueeze(1)


def _load_pil(path: Path) -> torch.Tensor:
    grayscale = np.asarray(Image.open(path).convert("L"))
    return _to_normalized_channel_tensor(grayscale[np.newaxis])


def _dicom_array_to_depth_first(pixel_array: np.ndarray, photometric: str) -> np.ndarray:
    pixel_array = np.squeeze(pixel_array)
    if pixel_array.ndim == 2:
        return pixel_array[np.newaxis]
    if pixel_array.ndim == 3:
        if pixel_array.shape[-1] in (3, 4) and photometric.startswith("RGB"):
            luminance = (
                0.2989 * pixel_array[..., 0].astype(np.float32)
                + 0.5870 * pixel_array[..., 1].astype(np.float32)
                + 0.1140 * pixel_array[..., 2].astype(np.float32)
            )
            return luminance[np.newaxis]
        return pixel_array
    raise ValueError(f"Unsupported DICOM pixel_array shape {pixel_array.shape}")


def _load_dicom(path: Path) -> torch.Tensor:
    dicom_dataset = pydicom.dcmread(str(path))
    photometric = str(getattr(dicom_dataset, "PhotometricInterpretation", "MONOCHROME2"))
    pixel_array = _dicom_array_to_depth_first(
        dicom_dataset.pixel_array, photometric
    ).astype(np.float32)
    slope = float(getattr(dicom_dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dicom_dataset, "RescaleIntercept", 0.0) or 0.0)
    pixel_array = pixel_array * slope + intercept
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array
    return _to_normalized_channel_tensor(pixel_array)


def _load_nifti(path: Path) -> torch.Tensor:
    data = nib.as_closest_canonical(nib.load(str(path))).get_fdata()
    if data.ndim == 3:
        depth_first = np.transpose(data, (2, 0, 1))
    elif data.ndim == 4:
        depth_first = np.transpose(data, (3, 2, 0, 1)).reshape(-1, data.shape[0], data.shape[1])
    else:
        raise ValueError(f"Unsupported NIfTI ndim {data.ndim} for {path}")
    return _to_normalized_channel_tensor(depth_first)


def _load_sitk(path: Path) -> torch.Tensor:
    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if volume.ndim == 2:
        volume = volume[np.newaxis]
    elif volume.ndim != 3:
        raise ValueError(f"Unsupported SimpleITK array shape {volume.shape} for {path}")
    return _to_normalized_channel_tensor(volume)


_LOADERS: dict[str, Callable[[Path], torch.Tensor]] = {
    ".png":  _load_pil,
    ".jpg":  _load_pil,
    ".jpeg": _load_pil,
    ".dcm":  _load_dicom,
    ".nii":  _load_nifti,
    ".nrrd": _load_sitk,
    ".mha":  _load_sitk,
    ".mhd":  _load_sitk,
}


def _canonical_suffix(path: Path) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return ".nii"
    return path.suffix.lower()


def _is_supported(path: Path) -> bool:
    return _canonical_suffix(path) in _LOADERS


_MIN_MATCH_PREFIX_LENGTH = 4


def _strip_all_extensions(path: Path) -> str:
    return path.name.split(".")[0]


def _shared_prefix_length(a: str, b: str) -> int:
    length = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        length += 1
    return length


def _list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and _is_supported(p))


def _find_matching_target(input_path: Path, targets: list[Path]) -> Optional[Path]:
    input_stem = _strip_all_extensions(input_path)
    best_match: Optional[Path] = None
    longest_prefix = 0
    for candidate in targets:
        length = _shared_prefix_length(input_stem, _strip_all_extensions(candidate))
        if length > longest_prefix:
            best_match, longest_prefix = candidate, length
    return best_match if longest_prefix >= _MIN_MATCH_PREFIX_LENGTH else None


class ImageLoader:
    def __init__(self, path: Path):
        self.path = path
        self.suffix = _canonical_suffix(path)
        if self.suffix not in _LOADERS:
            raise ValueError(f"Unsupported format: {path}")
        self._tensor: Optional[torch.Tensor] = None

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            self._tensor = _LOADERS[self.suffix](self.path)
        return self._tensor

    @property
    def rgb_tensor(self) -> torch.Tensor:
        return self.tensor.expand(-1, 3, -1, -1)

    @property
    def empty_slice_mask(self) -> torch.Tensor:
        volume = self.tensor.squeeze(1)
        return (volume.mean(dim=(1, 2)) < 1e-3) | (volume.std(dim=(1, 2)) < 1e-3)

    def log_tensor_shape(self) -> torch.Size:
        shape = self.tensor.shape
        print(f"[{self.path.name}] tensor size: {tuple(shape)}")
        return shape


_METRIC_CACHE: dict[str, torch.nn.Module] = {}


def _get_metric(name: str) -> torch.nn.Module:
    if name not in _METRIC_CACHE:
        _METRIC_CACHE[name] = pyiqa.create_metric(name, as_loss=False)
    return _METRIC_CACHE[name]


def _segment_otsu(grayscale_slice: np.ndarray) -> np.ndarray:
    if float(grayscale_slice.max()) <= float(grayscale_slice.min()):
        return np.zeros_like(grayscale_slice, dtype=bool)
    return grayscale_slice > threshold_otsu(grayscale_slice)


_SEGMENTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "otsu": _segment_otsu,
}

SEGMENTER = "otsu"
MASK_DIR = Path("report") / "masks"


def _active_segmenter() -> Callable[[np.ndarray], np.ndarray]:
    return _SEGMENTERS[SEGMENTER]


_MetricDirection = Literal["higher_is_better", "lower_is_better"]

_METRIC_DIRECTION: dict[str, _MetricDirection] = {
    "psnr":    "higher_is_better",
    "ssim":    "higher_is_better",
    "clipiqa": "higher_is_better",
    "lpips":   "lower_is_better",
    "dists":   "lower_is_better",
    "brisque": "lower_is_better",
    "niqe":    "lower_is_better",
}


def _slice_to_uint8(grayscale_float: np.ndarray) -> np.ndarray:
    return (np.clip(grayscale_float, 0.0, 1.0) * 255).astype(np.uint8)


def _mask_to_uint8(binary_mask: np.ndarray) -> np.ndarray:
    return (binary_mask.astype(bool) * 255).astype(np.uint8)


def _apply_color_overlay(
    grayscale_slice: np.ndarray,
    foreground_mask: np.ndarray,
    tint_color: tuple = (255, 0, 0),
    tint_strength: float = 0.4,
) -> np.ndarray:
    rgb = np.stack([_slice_to_uint8(grayscale_slice)] * 3, axis=-1).astype(np.float32)
    foreground = foreground_mask.astype(bool)
    for channel, color_value in enumerate(tint_color):
        rgb[foreground, channel] = (
            rgb[foreground, channel] * (1.0 - tint_strength) + color_value * tint_strength
        )
    return np.clip(rgb, 0, 255).astype(np.uint8)


@dataclass
class ImageEvaluatorRecord:
    image_id: str
    source_model: Optional[str] = None
    mode: str = "no_reference"
    slice_index: int = 0
    is_empty: bool = False
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    dists: Optional[float] = None
    clipiqa: Optional[float] = None
    brisque: Optional[float] = None
    niqe: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class IQAEvaluator:
    def __init__(
        self,
        input_image: ImageLoader,
        target_image: Optional[ImageLoader],
        source_model: Optional[str] = None,
    ):
        self.input = input_image
        self.target = target_image
        self.source_model = source_model

        if self.target is not None and self.input.tensor.shape != self.target.tensor.shape:
            raise ValueError(
                f"shape mismatch: input {tuple(self.input.tensor.shape)} "
                f"vs target {tuple(self.target.tensor.shape)}"
            )

    def _compute_psnr(self, slice_index: int) -> float:
        return float(_get_metric("psnr")(
            self.input.tensor[slice_index:slice_index+1],
            self.target.tensor[slice_index:slice_index+1],
        ).item())

    def _compute_ssim(self, slice_index: int) -> float:
        return float(_get_metric("ssim")(
            self.input.tensor[slice_index:slice_index+1],
            self.target.tensor[slice_index:slice_index+1],
        ).item())

    def _compute_lpips(self, slice_index: int) -> float:
        return float(_get_metric("lpips")(
            self.input.rgb_tensor[slice_index:slice_index+1],
            self.target.rgb_tensor[slice_index:slice_index+1],
        ).item())

    def _compute_dists(self, slice_index: int) -> float:
        return float(_get_metric("dists")(
            self.input.rgb_tensor[slice_index:slice_index+1],
            self.target.rgb_tensor[slice_index:slice_index+1],
        ).item())

    def _compute_clipiqa(self, slice_index: int) -> float:
        return float(_get_metric("clipiqa")(self.input.rgb_tensor[slice_index:slice_index+1]).item())

    def _compute_brisque(self, slice_index: int) -> float:
        return float(_get_metric("brisque")(self.input.rgb_tensor[slice_index:slice_index+1]).item())

    def _compute_niqe(self, slice_index: int) -> float:
        return float(_get_metric("niqe")(self.input.rgb_tensor[slice_index:slice_index+1]).item())

    def _run_safely(self, metric_name: str, compute: Callable[[], float]) -> Optional[float]:
        try:
            return compute()
        except Exception as exc:
            print(f"[{self.input.path}] metric '{metric_name}' failed: {exc}")
            return None

    def _format_slice_id(self, slice_index: int) -> str:
        base = f"{_strip_all_extensions(self.input.path)}_s{slice_index:03d}"
        return f"{self.source_model}/{base}" if self.source_model else base

    def run_evaluation(self) -> list[ImageEvaluatorRecord]:
        records: list[ImageEvaluatorRecord] = []
        empty_slices = self.input.empty_slice_mask
        num_slices = self.input.tensor.shape[0]
        has_target = self.target is not None

        for slice_index in range(num_slices):
            record = ImageEvaluatorRecord(
                image_id=self._format_slice_id(slice_index),
                source_model=self.source_model,
                mode="full_reference" if has_target else "no_reference",
                slice_index=slice_index,
                is_empty=bool(empty_slices[slice_index].item()),
            )
            if not record.is_empty:
                record.clipiqa = self._run_safely("clipiqa", lambda i=slice_index: self._compute_clipiqa(i))
                record.brisque = self._run_safely("brisque", lambda i=slice_index: self._compute_brisque(i))
                record.niqe    = self._run_safely("niqe",    lambda i=slice_index: self._compute_niqe(i))
                if has_target:
                    record.psnr  = self._run_safely("psnr",  lambda i=slice_index: self._compute_psnr(i))
                    record.ssim  = self._run_safely("ssim",  lambda i=slice_index: self._compute_ssim(i))
                    record.lpips = self._run_safely("lpips", lambda i=slice_index: self._compute_lpips(i))
                    record.dists = self._run_safely("dists", lambda i=slice_index: self._compute_dists(i))
            records.append(record)

        self._run_safely("segmentation", lambda: self.segment_and_save_best_slices(records, MASK_DIR))
        return records

    def _best_slice_per_metric(
        self, records: list[ImageEvaluatorRecord]
    ) -> dict[str, int]:
        value_index_pairs: dict[str, list[tuple[float, int]]] = {
            metric: [] for metric in _METRIC_DIRECTION
        }
        for record in records:
            if record.is_empty:
                continue
            for metric in _METRIC_DIRECTION:
                value = getattr(record, metric, None)
                if value is not None:
                    value_index_pairs[metric].append((value, record.slice_index))

        best_slice_per_metric: dict[str, int] = {}
        for metric, pairs in value_index_pairs.items():
            if not pairs:
                continue
            if _METRIC_DIRECTION[metric] == "higher_is_better":
                _, slice_index = max(pairs, key=lambda pair: pair[0])
            else:
                _, slice_index = min(pairs, key=lambda pair: pair[0])
            best_slice_per_metric[metric] = slice_index
        return best_slice_per_metric

    def segment_and_save_best_slices(
        self, records: list[ImageEvaluatorRecord], output_dir: Path
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _strip_all_extensions(self.input.path)
        segmenter = _active_segmenter()
        volume = self.input.tensor[:, 0].numpy()

        saved_paths: list[Path] = []
        for metric, slice_index in self._best_slice_per_metric(records).items():
            grayscale_slice = volume[slice_index]
            segmentation_mask = segmenter(grayscale_slice)

            output_prefix = str(output_dir / f"{stem}_{metric}_s{slice_index:03d}")
            slice_path   = Path(output_prefix + "_slice.png")
            mask_path    = Path(output_prefix + "_mask.png")
            overlay_path = Path(output_prefix + "_overlay.png")

            Image.fromarray(_slice_to_uint8(grayscale_slice), mode="L").save(slice_path)
            Image.fromarray(_mask_to_uint8(segmentation_mask), mode="L").save(mask_path)
            Image.fromarray(_apply_color_overlay(grayscale_slice, segmentation_mask), mode="RGB").save(overlay_path)

            saved_paths.extend([slice_path, mask_path, overlay_path])
            print(
                f"[{self.input.path.name}] {metric} best slice={slice_index}"
                f" -> {slice_path.name}, {mask_path.name}, {overlay_path.name}"
            )

        return saved_paths


def evaluate_dataset(report_path: Path) -> pd.DataFrame:
    evaluation_rows: list[dict] = []

    def _evaluate_and_collect(input_path: Path, target_path: Optional[Path]) -> None:
        try:
            input_image = ImageLoader(input_path)
            input_image.log_tensor_shape()
            target_image: Optional[ImageLoader] = None
            if target_path is not None:
                target_image = ImageLoader(target_path)
                target_image.log_tensor_shape()
            records = IQAEvaluator(input_image, target_image).run_evaluation()
            evaluation_rows.extend(record.to_dict() for record in records)
        except Exception as exc:
            print(f"[{input_path}] evaluator init/run failed: {exc}")

    if INPUT.is_file():
        _evaluate_and_collect(INPUT, TARGET if TARGET.is_file() else None)
    elif INPUT.is_dir():
        available_targets: list[Path] = []
        if TARGET.is_dir():
            available_targets = _list_images(TARGET)
        elif TARGET.is_file():
            available_targets = [TARGET]
        for input_path in _list_images(INPUT):
            _evaluate_and_collect(input_path, _find_matching_target(input_path, available_targets))
    else:
        print(f"No input file or directory at {INPUT}")
        return pd.DataFrame(columns=list(ImageEvaluatorRecord.__annotations__.keys()))

    columns = list(ImageEvaluatorRecord.__annotations__.keys())
    report = pd.DataFrame(evaluation_rows, columns=columns)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    return report


def main():
    report = evaluate_dataset(REPORT)
    print(report.describe())
    print(f"Report written: {REPORT}")


if "__main__" == __name__:
    main()
