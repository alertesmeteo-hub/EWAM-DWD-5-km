#!/usr/bin/env python3
"""Vagues EWAM (DWD, modèle européen à 0,05°) : grilles de valeurs pour la carte marine du site.

Source : https://opendata.dwd.de/weather/maritime/wave_models/ewam/grib/ (un fichier GRIB2 compressé par
variable et par échéance, runs 00 et 12 UTC). Les champs lus sont rééchantillonnés sur une grille Web Mercator
et publiés au format « HKV1 » (voir wavegrid.py), avec un manifeste maps/index.json de même forme que celui
de MFWAM (échéances, grilles de valeurs par échéance).
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from eccodes import codes_get, codes_get_double_array, codes_new_from_message, codes_release

from wavegrid import MercatorResampler, RegularGrid, write_hkv

LOGGER = logging.getLogger("ewam")
BASE = "https://opendata.dwd.de/weather/maritime/wave_models/ewam/grib/"
USER_AGENT = "alertes-meteo.com/ewam-dwd/1.0"
PIPELINE_VERSION = "1.0.0"
DEFAULT_CURRENT_METADATA_URL = "https://raw.githubusercontent.com/alertesmeteo-hub/EWAM-DWD-5-km/data/index.json"

# Emprise publiée : Méditerranée occidentale et centrale, golfe de Gascogne, Manche.
BOUNDS = {"south": 30.0, "west": -10.0, "north": 50.0, "east": 20.0}
PROBE_WIDTH = 400  # ~0,075° par cellule

# Clé de la grille publiée -> (dossier DWD, direction ?). Mêmes clés que MFWAM.
PROBES = {
    "hs": ("swh", False),
    "tp": ("tm10", False),
    "dir": ("mwd", True),
    "wind_h": ("shww", False),
    "wind_tp": ("mpww", False),
    "wind_dir": ("mdww", True),
    "swell_h": ("shts", False),
    "swell_tp": ("mpts", False),
    "swell_dir": ("mdts", True),
}
# Échéances publiées : toutes les heures jusqu'à +24 h, puis toutes les 3 h jusqu'à +78 h.
LEADS = list(range(0, 25)) + list(range(27, 79, 3))
MISSING = 9990.0


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def file_url(run: datetime, variable: str, lead: int) -> str:
    stamp = run.strftime("%Y%m%d%H")
    return f"{BASE}{run:%H}/{variable}/EWAM_{variable.upper()}_{stamp}_{lead:03d}.grib2.bz2"


def run_is_complete(session: requests.Session, run: datetime) -> bool:
    """Un run est utilisable quand la dernière échéance de chaque variable est publiée."""
    for variable, _ in PROBES.values():
        response = session.head(file_url(run, variable, LEADS[-1]), timeout=30, allow_redirects=True)
        if response.status_code != 200:
            return False
    return True


def latest_complete_run(session: requests.Session) -> datetime | None:
    now = datetime.now(timezone.utc)
    candidates = []
    for days_back in (0, 1):
        day = (now - timedelta(days=days_back)).replace(minute=0, second=0, microsecond=0)
        for hour in (12, 0):
            run = day.replace(hour=hour)
            if run <= now:
                candidates.append(run)
    for run in sorted(candidates, reverse=True):
        if run_is_complete(session, run):
            return run
    return None


def already_published(url: str, run: datetime) -> bool:
    try:
        response = requests.get(url, timeout=30, headers={"Cache-Control": "no-cache"})
        if response.status_code != 200:
            return False
        published = response.json().get("model", {}).get("run_time")
        return published == iso(run)
    except (requests.RequestException, ValueError):
        return False


def fetch_message(session: requests.Session, url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(url, timeout=(15, 120))
            if response.status_code == 200:
                return bz2.decompress(response.content)
            last = RuntimeError(f"HTTP {response.status_code}")
        except (requests.RequestException, OSError) as exc:
            last = exc
        if attempt < 3:
            import time

            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Téléchargement impossible : {url} ({last})")


def read_field(data: bytes) -> tuple[np.ndarray, RegularGrid]:
    gid = codes_new_from_message(data)
    try:
        ni = int(codes_get(gid, "Ni"))
        nj = int(codes_get(gid, "Nj"))
        lat_first = float(codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
        lat_last = float(codes_get(gid, "latitudeOfLastGridPointInDegrees"))
        lon_first = float(codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
        di = float(codes_get(gid, "iDirectionIncrementInDegrees"))
        dj = float(codes_get(gid, "jDirectionIncrementInDegrees"))
        values = codes_get_double_array(gid, "values").reshape(nj, ni).astype(np.float64)
    finally:
        codes_release(gid)
    values[values >= MISSING] = np.nan
    lon_first = (lon_first + 180.0) % 360.0 - 180.0
    grid = RegularGrid(lat_first=lat_first, lon_first=lon_first, lat_step=-dj if lat_first > lat_last else dj, lon_step=di, nj=nj, ni=ni)
    return values, grid


def build(run: datetime, workdir: Path) -> Path:
    result = workdir / "result"
    maps = result / "maps"
    maps.mkdir(parents=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Géométrie de la grille : lue une fois (première échéance de la hauteur significative).
    _, grid = read_field(fetch_message(session, file_url(run, "swh", LEADS[0])))
    resampler = MercatorResampler(BOUNDS, PROBE_WIDTH, grid)
    LOGGER.info("Grille source %sx%s -> grille Mercator %sx%s", grid.ni, grid.nj, resampler.width, resampler.height)

    steps = []
    for position, lead in enumerate(LEADS, start=1):
        urls = {key: file_url(run, variable, lead) for key, (variable, _) in PROBES.items()}
        with ThreadPoolExecutor(max_workers=6) as pool:
            payloads = dict(zip(urls, pool.map(lambda u: fetch_message(session, u), urls.values())))
        written: dict[str, str] = {}
        for key, (variable, is_direction) in PROBES.items():
            values, _ = read_field(payloads[key])
            sampled = resampler.sample(values, nearest=is_direction)
            relative = f"maps/values/{key}/{lead:03d}.hkv.gz"
            if write_hkv(result / relative, sampled):
                written[key] = relative
        if "hs" not in written:
            raise RuntimeError(f"Hauteur significative vide à +{lead} h")
        steps.append({"lead_hour": lead, "valid_time": iso(run + timedelta(hours=lead)), "files": {}, "probes": written})
        LOGGER.info("Échéance +%03d h (%s/%s)", lead, position, len(LEADS))

    generated_at = iso(datetime.now(timezone.utc))
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "module_version": PIPELINE_VERSION,
        "generated_at": generated_at,
        "run_time": iso(run),
        "projection": "EPSG:3857",
        "bounds": BOUNDS,
        "probe_grid": {"width": resampler.width, "height": resampler.height},
        "layers": {},
        "steps": steps,
    }
    with (maps / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    index = {
        "schema_version": 1,
        "status": "ok",
        "generated_at": generated_at,
        "model": {
            "name": "EWAM",
            "provider": "DWD (Deutscher Wetterdienst)",
            "dataset": "European Wave Model, maille 0,05°",
            "resolution_km": 5,
            "run_time": iso(run),
            "pipeline_version": PIPELINE_VERSION,
            "source_url": BASE,
            "license": "Open data DWD (GeoNutzV)",
        },
        "coverage": {"label": "Méditerranée occidentale et centrale, golfe de Gascogne, Manche (30N-50N, 10W-20E)"},
        "maps": {"status": "ok", "manifest": "maps/index.json", "steps": len(steps)},
    }
    with (result / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="build/national")
    parser.add_argument("--current-metadata-url", default=DEFAULT_CURRENT_METADATA_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    run = latest_complete_run(session)
    if run is None:
        LOGGER.info("Aucun run EWAM complet pour le moment.")
        return 0
    LOGGER.info("Run EWAM complet le plus récent : %s", iso(run))
    if not args.force and already_published(args.current_metadata_url, run):
        LOGGER.info("Ce run est déjà publié, rien à faire.")
        return 0
    with tempfile.TemporaryDirectory(prefix="ewam-build-") as tmp:
        result = build(run, Path(tmp))
        destination = Path(args.output_dir)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(result, destination)
    LOGGER.info("Fichiers prêts dans %s", args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.exception("Échec de la mise à jour EWAM")
        raise SystemExit(1)
