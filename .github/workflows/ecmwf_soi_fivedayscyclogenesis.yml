#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Génération quotidienne des produits de cyclogenèse à 5 jours
pour le Sud-Ouest océan Indien à partir des données ECMWF,
en réutilisant les fonctions de TropiDash.

Résultat : pour chaque système identifié dans le bassin,
on produit dans output/<storm_id>/ :
  - ensemble_tracks.geojson
  - mean_track.geojson
  - strike_probability.tif
  - strike_probability.png  (carte prête à être affichée sur e-monsite)
"""

from pathlib import Path
from datetime import datetime
import os
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
import rasterio

from tropidash_utils import utils_tracks as tracks


# ================== PARAMÈTRES GÉNÉRAUX ==================

# Si tu veux forcer une date de run depuis le workflow :
#   env:
#     COI_RUN_DATE: ${{ steps.date.outputs.date }}
# au format YYYYMMDD
run_date_str = os.environ.get("COI_RUN_DATE")

if run_date_str:
    RUN_DATE = datetime.strptime(run_date_str, "%Y%m%d")
else:
    # par défaut : date UTC du jour à 00Z
    now = datetime.utcnow()
    RUN_DATE = datetime(now.year, now.month, now.day, 0, 0)

# Dossiers
BASE_OUTPUT = Path("output") / RUN_DATE.strftime("%Y%m%d")
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
TRACKS_DIR = DATA_DIR / "tracks"
TRACKS_DIR.mkdir(parents=True, exist_ok=True)  # important pour le téléchargement BUFR

# Boîte géographique Sud-Ouest océan Indien
LON_MIN = 20.0
LON_MAX = 120.0
LAT_MIN = -45.0   # Sud
LAT_MAX = 0.0     # Équateur

# On enlève les "faux" systèmes (recommandation du tuto TropiDash)
MIN_STORM_ID = 70  # stormIdentifier >= 70


# ================== FONCTIONS UTILITAIRES ==================

def to_geojson_linestring_list(locs_list, properties_list=None):
    """
    Convertit une liste de trajectoires (listes de (lat, lon))
    en FeatureCollection GeoJSON de LineString.
    """
    features = []

    for i, locs in enumerate(locs_list):
        coords = [[float(lon), float(lat)] for (lat, lon) in locs]
        props = {"member": i}
        if properties_list is not None and i < len(properties_list):
            props.update(properties_list[i])

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": props,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def save_strike_map_png(tif_path, png_path):
    """
    Convertit le GeoTIFF de strike probability en PNG simple
    (sans axes) pour affichage direct sur le site.
    """
    with rasterio.open(tif_path) as r:
        data = r.read(1)
        bounds = r.bounds

    # On masque les zéros (pas de probabilité)
    data = np.ma.masked_where(data <= 0, data)

    # Palette inspirée de TropiDash
    palette = [
        "#8df52c", "#6ae24c", "#61bb30", "#508b15",
        "#057941", "#2397d1", "#557ff3", "#143cdc",
        "#3910b4", "#1e0063"
    ]

    plt.figure(figsize=(8, 6))
    from matplotlib.colors import ListedColormap
    plt.imshow(
        data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap=ListedColormap(palette),
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


def process_storm(df_storms_forecast, storm_id):
    """
    Traite un système donné (stormIdentifier) :
      - calcule les trajectoires
      - calcule la strike probability map
      - sauvegarde GeoJSON + TIF + PNG
    """
    df_storm = df_storms_forecast[df_storms_forecast.stormIdentifier == storm_id].copy()
    if df_storm.empty:
        return

    # Dossier de sortie pour ce système
    storm_dir = BASE_OUTPUT / f"storm_{storm_id}"
    storm_dir.mkdir(parents=True, exist_ok=True)

    print(f"  > Système {storm_id} : {len(df_storm)} points")

    # === Trajectoires d'ensemble ===
    locations_f, timesteps_f, pressures_f, wind_speeds_f = tracks.forecast_tracks_locations(df_storm)

    ens_props = []
    for i in range(len(locations_f)):
        ens_props.append(
            {
                "timesteps": [str(t) for t in timesteps_f[i]],
                "pressure_hpa": [float(p) for p in pressures_f[i]],
                "wind_ms": [float(w) for w in wind_speeds_f[i]],
            }
        )

    ens_geojson = to_geojson_linestring_list(locations_f, ens_props)
    (storm_dir / "ensemble_tracks.geojson").write_text(
        json.dumps(ens_geojson, ensure_ascii=False)
    )

    # === Trajectoire moyenne ===
    locations_avg, timesteps_avg, pressures_avg, wind_speeds_avg = tracks.mean_forecast_track(
        df_storm
    )

    mean_props = [
        {
            "timesteps": [str(t) for t in timesteps_avg],
            "pressure_percentiles_hpa": [[float(x) for x in row] for row in pressures_avg],
            "wind_percentiles_ms": [[float(x) for x in row] for row in wind_speeds_avg],
        }
    ]
    mean_geojson = to_geojson_linestring_list([locations_avg], mean_props)
    (storm_dir / "mean_track.geojson").write_text(
        json.dumps(mean_geojson, ensure_ascii=False)
    )

    # === Strike probability map ===
    strike_map_xr, tif_path = tracks.strike_probability_map(df_storm)
    tif_path = Path(tif_path)

    target_tif = storm_dir / "strike_probability.tif"
    shutil.copy(tif_path, target_tif)

    # PNG pour le site
    target_png = storm_dir / "strike_probability.png"
    save_strike_map_png(target_tif, target_png)

    print(f"    -> fichiers générés dans {storm_dir}")


# ================== MAIN ==================

def main():
    print(f"=== Génération produits SOI pour run {RUN_DATE} ===")

    # 1) Télécharger les données ECMWF (BUFR) si nécessaire
    print("Téléchargement des données ECMWF (download_tracks_forecast)…")
    start_date = tracks.download_tracks_forecast(RUN_DATE)

    # 2) Charger toutes les tempêtes à partir du fichier BUFR
    print("Téléchargement / chargement des données ECMWF (create_storms_df)…")
    df_storms = tracks.create_storms_df(start_date)

    if df_storms.empty:
        print("Aucune tempête détectée dans les données ECMWF.")
        return

    # 3) Filtrer le bassin Sud-Ouest océan Indien
    print("Filtrage sur le bassin Sud-Ouest océan Indien…")

    df_basin = df_storms[
        (df_storms.latitude < LAT_MAX) &
        (df_storms.latitude > LAT_MIN) &
        (df_storms.longitude >= LON_MIN) &
        (df_storms.longitude <= LON_MAX) &
        (df_storms.stormIdentifier.astype(str) >= str(MIN_STORM_ID))
    ].copy()

    if df_basin.empty:
        print("Aucun système suivi dans le bassin SOI pour ce run.")
        return

    storm_ids = sorted(df_basin.stormIdentifier.unique())
    print(f"Systèmes identifiés dans le SOI : {storm_ids}")

    # 4) Traiter chaque système
    for sid in storm_ids:
        process_storm(df_basin, sid)

    print("\n✅ Génération terminée.")


if __name__ == "__main__":
    main()
