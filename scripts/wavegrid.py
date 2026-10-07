"""Rééchantillonnage des champs de vagues sur la grille Web Mercator de la carte marine, et écriture des grilles de valeurs.

Format des grilles (identique à AROME / MFWAM) : fichier gzip, en-tête « HKV1 » (largeur, hauteur, minimum,
maximum), puis un entier non signé 16 bits par cellule (65535 = pas de valeur). Les lignes suivent la
projection Web Mercator (EPSG:3857), colonnes régulières en longitude.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def mercator(latitude):
    radians = np.radians(np.clip(latitude, -85.0, 85.0))
    return np.log(np.tan(np.pi / 4.0 + radians / 2.0))


def inverse_mercator(value):
    return np.degrees(2.0 * np.arctan(np.exp(value)) - np.pi / 2.0)


@dataclass(frozen=True)
class RegularGrid:
    """Grille régulière lat/lon d'un GRIB2 : premier point, pas signés et dimensions."""

    lat_first: float
    lon_first: float  # en [-180, 180[
    lat_step: float  # négatif si la grille balaie du nord vers le sud
    lon_step: float
    nj: int
    ni: int


class MercatorResampler:
    """Interpole une grille régulière lat/lon sur une grille Mercator (largeur fixe, hauteur déduite de l'emprise)."""

    def __init__(self, bounds: dict[str, float], width: int, grid: RegularGrid) -> None:
        span_x = np.radians(bounds["east"] - bounds["west"])
        span_y = float(mercator(np.asarray(bounds["north"])) - mercator(np.asarray(bounds["south"])))
        self.width = int(width)
        self.height = int(round(width * span_y / span_x))
        lat_t = inverse_mercator(np.linspace(mercator(np.asarray(float(bounds["north"]))), mercator(np.asarray(float(bounds["south"]))), self.height))
        lon_t = np.linspace(bounds["west"], bounds["east"], self.width)
        rows = (lat_t[:, None] - grid.lat_first) / grid.lat_step
        cols = (lon_t[None, :] - grid.lon_first) / grid.lon_step
        self.rows = np.broadcast_to(rows, (self.height, self.width))
        self.cols = np.broadcast_to(cols, (self.height, self.width))
        self.coverage = (self.rows >= 0) & (self.rows <= grid.nj - 1) & (self.cols >= 0) & (self.cols <= grid.ni - 1)

    def sample(self, values: np.ndarray, nearest: bool = False) -> np.ndarray:
        """Valeurs sur la grille Mercator ; NaN sur terre / hors domaine. `nearest` pour les directions (0-360°)."""
        invalid = ~np.isfinite(values)
        if invalid.any() and not invalid.all():
            idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
            filled = values[tuple(idx)]
        else:
            filled = values
        out = map_coordinates(filled, [self.rows, self.cols], order=0 if nearest else 1, mode="constant", cval=np.nan, prefilter=False).astype(np.float32, copy=False)
        bad = map_coordinates(invalid.astype(np.float32), [self.rows, self.cols], order=0, mode="constant", cval=1.0, prefilter=False)
        out[bad >= 0.5] = np.nan
        out[~self.coverage] = np.nan
        return out


def write_hkv(path: Path, values: np.ndarray) -> bool:
    """Écrit une grille HKV1 compressée ; renvoie False si le champ est entièrement vide."""
    finite = np.isfinite(values)
    if not finite.any():
        return False
    vmin = float(values[finite].min())
    vmax = float(values[finite].max())
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    raw = np.full(values.shape, 65535, dtype="<u2")
    raw[finite] = np.rint((values[finite] - vmin) / (vmax - vmin) * 65534).astype("<u2")
    height, width = values.shape
    header = b"HKV1" + struct.pack("<HHff", width, height, vmin, vmax)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=9) as handle:
        handle.write(header + raw.tobytes())
    return True
