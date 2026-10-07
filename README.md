# EWAM (DWD) — vagues 5 km pour la carte marine

Grilles de valeurs des vagues du modèle **EWAM** du Deutscher Wetterdienst (Méditerranée, golfe de Gascogne,
Manche), lues par la carte « Météo marine » de https://www.alertes-meteo.com/meteo-cotier.

- Source : `https://opendata.dwd.de/weather/maritime/wave_models/ewam/grib/` (runs 00 et 12 UTC, +78 h).
- Sortie : branche `data` — `index.json` (modèle, run) et `maps/index.json` (échéances, grilles de valeurs :
  hauteur significative, période, direction, mer du vent, houle) au format HKV1 de la carte interactive.
- Workflow : `.github/workflows/update-ewam.yml` (horaire, relancé aussi par le déclencheur du site).

```bash
pip install -r requirements.txt
cd scripts && python update_ewam.py --output-dir ../build/national --force
```
