"""Offline bilinear cost models from Section 3.3 of the Helix paper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch


@dataclass(frozen=True)
class ProfileSample:
    micro_batch_size: int
    submodel_size: float
    compute_seconds: float
    memory_bytes: float

    def __post_init__(self) -> None:
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        if not 0.0 < self.submodel_size <= 1.0:
            raise ValueError("submodel_size must be in (0, 1]")
        if self.compute_seconds <= 0.0 or self.memory_bytes <= 0.0:
            raise ValueError("profile measurements must be positive")


@dataclass(frozen=True)
class BilinearModel:
    """y(b, s) = c0 + c1*b + c2*s + c3*b*s."""

    c0: float
    c_batch: float
    c_size: float
    c_batch_size: float
    mape: float = 0.0

    def predict(self, micro_batch_size: float, submodel_size: float) -> float:
        return (
            self.c0
            + self.c_batch * micro_batch_size
            + self.c_size * submodel_size
            + self.c_batch_size * micro_batch_size * submodel_size
        )

    def size_at_budget(self, micro_batch_size: int, budget: float) -> float:
        """Solve the bilinear equation for s at a fixed b and y budget."""

        denominator = self.c_size + self.c_batch_size * micro_batch_size
        if denominator <= 0.0:
            raise ValueError("cost model is not increasing with submodel size")
        return (budget - self.c0 - self.c_batch * micro_batch_size) / denominator


@dataclass(frozen=True)
class DeviceProfile:
    rank: int
    device_name: str
    memory_budget_bytes: float
    compute: BilinearModel
    memory: BilinearModel
    samples: Sequence[ProfileSample]
    model_name: str = ""
    sequence_length: int = 0
    dtype: str = ""

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["samples"] = [asdict(sample) for sample in self.samples]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "DeviceProfile":
        return cls(
            rank=int(payload["rank"]),
            device_name=str(payload["device_name"]),
            memory_budget_bytes=float(payload["memory_budget_bytes"]),
            compute=BilinearModel(**payload["compute"]),
            memory=BilinearModel(**payload["memory"]),
            samples=[ProfileSample(**sample) for sample in payload.get("samples", [])],
            model_name=str(payload.get("model_name", "")),
            sequence_length=int(payload.get("sequence_length", 0)),
            dtype=str(payload.get("dtype", "")),
        )


def _design_matrix(samples: Sequence[ProfileSample]) -> torch.Tensor:
    return torch.tensor(
        [
            [
                1.0,
                float(sample.micro_batch_size),
                float(sample.submodel_size),
                float(sample.micro_batch_size) * float(sample.submodel_size),
            ]
            for sample in samples
        ],
        dtype=torch.float64,
    )


def fit_bilinear_model(samples: Sequence[ProfileSample], metric: str) -> BilinearModel:
    """Fit the paper's four-coefficient model and report training-set MAPE."""

    if metric not in {"compute_seconds", "memory_bytes"}:
        raise ValueError(f"Unsupported profile metric: {metric}")
    if len(samples) < 4:
        raise ValueError("At least four samples are required to fit a bilinear model")
    design = _design_matrix(samples)
    if int(torch.linalg.matrix_rank(design)) < 4:
        raise ValueError("Profile samples do not span the four bilinear features")
    targets = torch.tensor([float(getattr(sample, metric)) for sample in samples], dtype=torch.float64)
    coefficients = torch.linalg.lstsq(design, targets).solution
    predictions = design @ coefficients
    mape = torch.mean(torch.abs(predictions - targets) / torch.clamp(torch.abs(targets), min=1e-12))
    return BilinearModel(
        c0=float(coefficients[0]),
        c_batch=float(coefficients[1]),
        c_size=float(coefficients[2]),
        c_batch_size=float(coefficients[3]),
        mape=float(mape),
    )


def build_device_profile(
    rank: int,
    device_name: str,
    memory_budget_bytes: float,
    samples: Sequence[ProfileSample],
    model_name: str = "",
    sequence_length: int = 0,
    dtype: str = "",
) -> DeviceProfile:
    return DeviceProfile(
        rank=rank,
        device_name=device_name,
        memory_budget_bytes=memory_budget_bytes,
        compute=fit_bilinear_model(samples, "compute_seconds"),
        memory=fit_bilinear_model(samples, "memory_bytes"),
        samples=list(samples),
        model_name=model_name,
        sequence_length=sequence_length,
        dtype=dtype,
    )


def save_profiles(path: str, profiles: Iterable[DeviceProfile]) -> None:
    payload = {
        "schema_version": 1,
        "profiles": [profile.to_dict() for profile in profiles],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_profiles(path: str) -> List[DeviceProfile]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported profile schema: {payload.get('schema_version')}")
    profiles = [DeviceProfile.from_dict(item) for item in payload["profiles"]]
    ranks = [profile.rank for profile in profiles]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Profile file contains duplicate ranks")
    return sorted(profiles, key=lambda profile: profile.rank)
