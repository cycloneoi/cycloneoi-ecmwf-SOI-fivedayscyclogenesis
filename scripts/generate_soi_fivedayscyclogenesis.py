#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Génération quotidienne des produits de cyclogenèse à 5 jours
pour l'océan Indien Sud à partir des données ECMWF,
en réutilisant les fonctions de TropiDash.

Résultat : pour chaque système identifié dans le bassin,
on produit dans output/YYYYMMDD/storm_<id>/ :
  - ensemble_tracks.geojson
  - mean_track.geojson
  - strike_probability.tif
  - strike_probability.png
  - ensemble_tracks.png  (vue des trajectoires d'ensemble)

En plus, on crée toujours dans output/latest/ :
  - cyclogenesis.png      (carte de probabilité principale)
  - ensemble_tracks.png   (vue des trajectoires d'ensemble)
  - strike_probability.png (alias de cyclogenesis.png pour compat)

Si aucun système n'est détecté, ces images sont remplacées par
un visuel "Aucun système suspecté" au style CycloneOI.
"""

from pathlib import Path
from datetime import datetime
import os
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio

from tropidash_utils import utils_tracks as tracks


# ================== PARAMÈTRES GÉNÉRAUX ==================

# Style CycloneOI
COI_BG = "#050608"
COI_GOLD = "#f4c542"
COI_TEXT = "#e6eaf0"

# Si tu veux forcer une date de run depuis le workflow :
#   env:
#     COI_RUN_DATE: YYYYMMDD
run_date_str = os.environ.get("COI_RUN_DATE")

if run_date_str:
    RUN_DATE = datetime.strptime(run_date_str, "%Y%m%d")
else:
    # par défaut : date UTC du jour (00Z pour ce produit ECMWF)
    now = datetime.utcnow()
    RUN_DATE = datetime(now.year, now.month, now.day, 0, 0)

# Dossiers de sortie
BASE_OUTPUT = Path("output") / RUN_DATE.strftime("%Y%m%d")
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

# Dossier "latest" qui contiendra toujours les images à afficher sur le site
LATEST_DIR = Path("output") / "latest"
LATEST_DIR.mkdir(parents=True, exist_ok=True)

# Dossiers de données brutes
DATA_DIR = Path("data")
TRACKS_DIR = DATA_DIR / "tracks"
TRACKS_DIR.mkdir(parents=True, exist_ok=True)

# Boîte géographique pour TOUT l'océan Indien Sud
# (grosso modo Afrique de l'Est -> Australie, équateur -> 60°S)
LON_MIN = 20.0
LON_MAX = 120.0
LAT_MIN = -60.0
LAT_MAX = 0.0

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


def save_strike_map_png(tif_path, png_path, title="Cyclogenèse à 5 jours – Océan Indien Sud"):
    """
    Convertit le GeoTIFF de strike probability en PNG stylé CycloneOI
    pour affichage direct sur le site.
    """
    with rasterio.open(tif_path) as r:
        data = r.read(1)
        bounds = r.bounds

    # On masque les zéros (pas de probabilité)
    data = np.ma.masked_where(data <= 0, data)

    # Palette inspirée de TropiDash mais adaptée CycloneOI
    palette = [
        "#8df52c", "#6ae24c", "#61bb30", "#508b15",
        "#057941", "#2397d1", "#557ff3", "#143cdc",
        "#3910b4", COI_GOLD
    ]

    plt.figure(figsize=(8, 6))
    plt.gcf().patch.set_facecolor(COI_BG)

    plt.imshow(
        data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap=ListedColormap(palette),
    )
    plt.axis("off")

    # Titre CycloneOI
    plt.title(
        title,
        fontsize=14,
        color=COI_GOLD,
        pad=12
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.1, dpi=150, facecolor=COI_BG)
    plt.close()


def create_placeholder_png(path, subtitle):
    """
    Crée une image simple indiquant qu'aucun système n'est suivi
    dans l'océan Indien Sud sur les 5 prochains jours.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(COI_BG)
    ax.set_facecolor(COI_BG)

    ax.text(
        0.5,
        0.65,
        "Aucun système suspecté\npour les 5 prochains jours\nsur l'océan Indien Sud",
        ha="center",
        va="center",
        fontsize=14,
        color=COI_TEXT,
        wrap=True,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.35,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
        color="#aaaaaa",
        transform=ax.transAxes,
    )

    ax.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.3, dpi=150, facecolor=COI_BG)
    plt.close()


def create_ensemble_overview_png(locations_f, locations_avg, png_path, storm_id):
    """
    Crée une vue simple des trajectoires d'ensemble + trajectoire moyenne
    au style CycloneOI (pas une carte complète, mais un aperçu clair).
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.patch.set_facecolor(COI_BG)
    ax.set_facecolor(COI_BG)

    # Trajectoires d'ensemble
    for locs in locations_f:
        lats = [lat for (lat, lon) in locs]
        lons = [lon for (lat, lon) in locs]
        ax.plot(lons, lats, linewidth=0.8, alpha=0.4, color="#4ade80")  # vert clair

    # Trajectoire moyenne
    if locations_avg:
        lat_avg = [lat for (lat, lon) in locations_avg]
        lon_avg = [lon for (lat, lon) in locations_avg]
        ax.plot(lon_avg, lat_avg, linewidth=2.0, color=COI_GOLD, label="Trajectoire moyenne")

    ax.set_xlabel("Longitude", color=COI_TEXT)
    ax.set_ylabel("Latitude", color=COI_TEXT)

    # Limites auto + petite marge
    all_lats = [lat for locs in locations_f for (lat, lon) in locs]
    all_lons = [lon for locs in locations_f for (lat, lon) in locs]
    if all_lats and all_lons:
        margin = 2.0
        ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
        ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)

    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.tick_params(colors="#bbbbbb", labelsize=8)

    ax.set_title(
        f"Ensembles ECMWF – Système {storm_id}",
        fontsize=13,
        color=COI_GOLD,
        pad=10,
    )

    if locations_avg:
        ax.legend(facecolor=COI_BG, edgecolor="#555555", labelcolor=COI_TEXT)

    plt.tight_layout(pad=0.7)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.2, dpi=150, facecolor=COI_BG)
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

    # PNG cyclogenèse
    target_png = storm_dir / "strike_probability.png"
    save_strike_map_png(
        target_tif,
        target_png,
        title="Cyclogenèse à 5 jours – Océan Indien Sud"
    )

    # PNG ensembles
    ens_png = storm_dir / "ensemble_tracks.png"
    create_ensemble_overview_png(locations_f, locations_avg, ens_png, storm_id)

    print(f"    -> fichiers générés dans {storm_dir}")


# ================== MAIN ==================

def main():
    print(f"=== Génération produits IO Sud pour run {RUN_DATE:%Y-%m-%d} ===")

    # 1) Télécharger les données ECMWF (BUFR) si nécessaire
    print("Téléchargement des données ECMWF (download_tracks_forecast)…")
    start_date = tracks.download_tracks_forecast(RUN_DATE)

    # 2) Charger toutes les tempêtes à partir du fichier BUFR
    print("Téléchargement / chargement des données ECMWF (create_storms_df)…")
    df_storms = tracks.create_storms_df(start_date)

    subtitle = f"Run ECMWF : {start_date:%Y-%m-%d}"

    if df_storms.empty:
        print("Aucune tempête détectée dans les données ECMWF.")
        # Placeholders dans /latest
        create_placeholder_png(LATEST_DIR / "cyclogenesis.png", subtitle)
        create_placeholder_png(LATEST_DIR / "ensemble_tracks.png", subtitle)
        # compat ancien chemin
        create_placeholder_png(LATEST_DIR / "strike_probability.png", subtitle)
        print(f"  -> Placeholders générés dans {LATEST_DIR}")
        return

    # 3) Filtrer le bassin Océan Indien Sud
    print("Filtrage sur le bassin Océan Indien Sud…")

    df_basin = df_storms[
        (df_storms.latitude < LAT_MAX) &
        (df_storms.latitude > LAT_MIN) &
        (df_storms.longitude >= LON_MIN) &
        (df_storms.longitude <= LON_MAX) &
        (df_storms.stormIdentifier.astype(str) >= str(MIN_STORM_ID))
    ].copy()

    if df_basin.empty:
        print("Aucun système suivi dans l'océan Indien Sud pour ce run.")
        create_placeholder_png(LATEST_DIR / "cyclogenesis.png", subtitle)
        create_placeholder_png(LATEST_DIR / "ensemble_tracks.png", subtitle)
        create_placeholder_png(LATEST_DIR / "strike_probability.png", subtitle)
        print(f"  -> Placeholders générés dans {LATEST_DIR}")
        return

    storm_ids = sorted(df_basin.stormIdentifier.unique())
    print(f"Systèmes identifiés dans l'océan Indien Sud : {storm_ids}")

    # 4) Traiter chaque système
    for sid in storm_ids:
        process_storm(df_basin, sid)

    # 5) Copier la carte du premier système vers output/latest/
    first_storm_dir = BASE_OUTPUT / f"storm_{storm_ids[0]}"

    # cyclogenèse
    src_cyclo = first_storm_dir / "strike_probability.png"
    latest_cyclo = LATEST_DIR / "cyclogenesis.png"
    compat_old = LATEST_DIR / "strike_probability.png"

    if src_cyclo.exists():
        shutil.copy(src_cyclo, latest_cyclo)
        shutil.copy(src_cyclo, compat_old)
    else:
        create_placeholder_png(latest_cyclo, subtitle)
        create_placeholder_png(compat_old, subtitle)

    # ensembles
    src_ens = first_storm_dir / "ensemble_tracks.png"
    latest_ens = LATEST_DIR / "ensemble_tracks.png"

    if src_ens.exists():
        shutil.copy(src_ens, latest_ens)
    else:
        create_placeholder_png(latest_ens, subtitle)

    print(f"\nImages 'latest' mises à jour dans {LATEST_DIR}")
    print("\n✅ Génération terminée.")


if __name__ == "__main__":
    main()
