#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Produits ECMWF à 5 jours pour l'océan Indien Sud (style CycloneOI).

Pour chaque run :
- Filtre les systèmes tropicaux dans l'océan Indien Sud.
- Pour chaque système, génère dans output/YYYYMMDD/storm_<id>/ :
    * ensemble_tracks.geojson
    * mean_track.geojson
    * strike_probability.tif
    * strike_probability.png          (carte de cyclogenèse)
    * ensemble_tracks.png             (vue des trajectoires d'ensemble)
    * max_wind.png                    (heatmap du vent max prévu)

- Met également à jour output/latest/ :
    * cyclogenesis.png      (copie du strike_probability du système principal)
    * ensemble_tracks.png   (copie de la vue ensembles du système principal)
    * max_wind.png          (copie de la heatmap vent max du système principal)
    * strike_probability.png (alias pour compat avec l'ancien code)

Si aucun système n'est détecté, ces images sont remplacées par
un visuel "aucun système suspecté" avec un fond bleu océan
centré sur l'océan Indien Sud.
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

# Palette CycloneOI
COI_BG = "#050816"     # fond bleu nuit
COI_GOLD = "#f4c542"   # or CycloneOI
COI_TEXT = "#e6eaf0"   # texte clair

# Dossier & date de run
run_date_str = os.environ.get("COI_RUN_DATE")
if run_date_str:
    RUN_DATE = datetime.strptime(run_date_str, "%Y%m%d")
else:
    now = datetime.utcnow()
    RUN_DATE = datetime(now.year, now.month, now.day, 0, 0)

BASE_OUTPUT = Path("output") / RUN_DATE.strftime("%Y%m%d")
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

LATEST_DIR = Path("output") / "latest"
LATEST_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
TRACKS_DIR = DATA_DIR / "tracks"
TRACKS_DIR.mkdir(parents=True, exist_ok=True)

# Boîte géographique : TOUT l'océan Indien Sud
LON_MIN = 20.0
LON_MAX = 120.0
LAT_MIN = -60.0
LAT_MAX = 0.0

# On enlève les "faux" systèmes
MIN_STORM_ID = 70


# ================== FONCTIONS UTILITAIRES ==================

def to_geojson_linestring_list(locs_list, properties_list=None):
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


def base_axes_with_basin(ax):
    """Applique un look 'carte océan Indien Sud' pour les placeholders."""
    ax.set_facecolor(COI_BG)
    # cadre du bassin
    ax.plot(
        [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
        [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
        color="#274060",
        linewidth=1.2,
        linestyle="--",
    )
    ax.set_xlim(LON_MIN - 5, LON_MAX + 5)
    ax.set_ylim(LAT_MIN - 5, LAT_MAX + 5)
    ax.set_xlabel("Longitude", color="#9ca3af", fontsize=8)
    ax.set_ylabel("Latitude", color="#9ca3af", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.grid(color="#1f2933", linestyle=":", linewidth=0.5, alpha=0.6)


def save_strike_map_png(tif_path, png_path, title):
    """Carte de probabilité de cyclogenèse (raster ECMWF)."""
    with rasterio.open(tif_path) as r:
        data = r.read(1)
        bounds = r.bounds

    data = np.ma.masked_where(data <= 0, data)

    palette = [
        "#8df52c", "#6ae24c", "#61bb30", "#508b15",
        "#057941", "#2397d1", "#557ff3", "#143cdc",
        "#3910b4", COI_GOLD
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COI_BG)
    ax.set_facecolor(COI_BG)

    im = ax.imshow(
        data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap=ListedColormap(palette),
    )

    base_axes_with_basin(ax)
    ax.set_title(title, fontsize=13, color=COI_GOLD, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=COI_TEXT)
    cbar.outline.set_edgecolor("#4b5563")
    cbar.set_label("Probabilité de cyclogenèse (%)", color=COI_TEXT, fontsize=8)

    plt.tight_layout(pad=0.6)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.2, dpi=150, facecolor=COI_BG)
    plt.close(fig)


def create_placeholder_png(path, subtitle, message=None):
    """Visuel 'aucun système' avec fond océan et cadre du bassin."""
    if message is None:
        message = (
            "Aucun système suspecté\n"
            "pour les 5 prochains jours\n"
            "sur l'océan Indien Sud"
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor(COI_BG)

    base_axes_with_basin(ax)

    ax.text(
        0.5,
        0.65,
        message,
        ha="center",
        va="center",
        fontsize=13,
        color=COI_TEXT,
        transform=ax.transAxes,
        wrap=True,
    )
    ax.text(
        0.5,
        0.25,
        subtitle,
        ha="center",
        va="center",
        fontsize=9,
        color="#9ca3af",
        transform=ax.transAxes,
    )

    plt.tight_layout(pad=0.6)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.25, dpi=150, facecolor=COI_BG)
    plt.close(fig)


def create_ensemble_overview_png(locations_f, locations_avg, png_path, storm_id):
    """Vue ensembles : trajectoires + trajectoire moyenne."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.patch.set_facecolor(COI_BG)
    ax.set_facecolor(COI_BG)

    # trajectoires d'ensemble
    for locs in locations_f:
        lats = [lat for (lat, lon) in locs]
        lons = [lon for (lat, lon) in locs]
        ax.plot(lons, lats, linewidth=0.7, alpha=0.45, color="#4ade80")

    # trajectoire moyenne
    if locations_avg:
        lat_avg = [lat for (lat, lon) in locations_avg]
        lon_avg = [lon for (lat, lon) in locations_avg]
        ax.plot(lon_avg, lat_avg, linewidth=2.0, color=COI_GOLD, label="Moyenne ECMWF")

    base_axes_with_basin(ax)

    ax.set_title(
        f"Trajectoires d'ensemble – Système {storm_id}",
        fontsize=13,
        color=COI_GOLD,
        pad=10,
    )

    if locations_avg:
        leg = ax.legend(facecolor=COI_BG, edgecolor="#4b5563", fontsize=8)
        for text in leg.get_texts():
            text.set_color(COI_TEXT)

    plt.tight_layout(pad=0.7)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.25, dpi=150, facecolor=COI_BG)
    plt.close(fig)


def create_max_wind_heatmap(df_storm, png_path, storm_id):
    """Heatmap vent max prévu (m/s) sur le bassin pour ce système."""
    df = df_storm.dropna(subset=["latitude", "longitude", "windSpeedAt10M"]).copy()
    if df.empty:
        subtitle = f"Système {storm_id} – données vent indisponibles"
        create_placeholder_png(png_path, subtitle, message="Vent max ECMWF indisponible")
        return

    lats = df.latitude.to_numpy()
    lons = df.longitude.to_numpy()
    winds = df.windSpeedAt10M.to_numpy()

    # grille grossière suffisante pour un produit web
    lat_bins = np.arange(max(lats.min() - 2, LAT_MIN), min(lats.max() + 2, LAT_MAX) + 0.1, 1.0)
    lon_bins = np.arange(max(lons.min() - 2, LON_MIN), min(lons.max() + 2, LON_MAX) + 0.1, 1.0)

    grid = np.full((len(lat_bins) - 1, len(lon_bins) - 1), np.nan)

    lat_idx = np.digitize(lats, lat_bins) - 1
    lon_idx = np.digitize(lons, lon_bins) - 1

    for i in range(len(lats)):
        ii = lat_idx[i]
        jj = lon_idx[i]
        if ii < 0 or jj < 0 or ii >= grid.shape[0] or jj >= grid.shape[1]:
            continue
        w = winds[i]
        if np.isnan(grid[ii, jj]) or w > grid[ii, jj]:
            grid[ii, jj] = w

    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.patch.set_facecolor(COI_BG)
    ax.set_facecolor(COI_BG)

    data = np.ma.masked_invalid(grid)
    cmap = plt.cm.inferno

    im = ax.imshow(
        data,
        extent=(lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]),
        origin="lower",
        cmap=cmap,
    )

    base_axes_with_basin(ax)
    ax.set_title(
        f"Vent max prévu (m/s) – Système {storm_id}",
        fontsize=13,
        color=COI_GOLD,
        pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=COI_TEXT)
    cbar.outline.set_edgecolor("#4b5563")
    cbar.set_label("Vent max (m/s)", color=COI_TEXT, fontsize=8)

    plt.tight_layout(pad=0.7)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.25, dpi=150, facecolor=COI_BG)
    plt.close(fig)


def process_storm(df_storms_forecast, storm_id):
    """Traite un système (trajectoires, cyclogenèse, vent max)."""
    df_storm = df_storms_forecast[df_storms_forecast.stormIdentifier == storm_id].copy()
    if df_storm.empty:
        return

    storm_dir = BASE_OUTPUT / f"storm_{storm_id}"
    storm_dir.mkdir(parents=True, exist_ok=True)

    print(f"  > Système {storm_id} : {len(df_storm)} points")

    # --- Trajectoires d'ensemble
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

    # --- Trajectoire moyenne
    locations_avg, timesteps_avg, pressures_avg, wind_speeds_avg = tracks.mean_forecast_track(df_storm)
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

    # --- Strike probability map
    strike_map_xr, tif_path = tracks.strike_probability_map(df_storm)
    tif_path = Path(tif_path)
    target_tif = storm_dir / "strike_probability.tif"
    shutil.copy(tif_path, target_tif)

    cyclo_png = storm_dir / "strike_probability.png"
    save_strike_map_png(
        target_tif,
        cyclo_png,
        title="Probabilité de cyclogenèse à 5 jours – Océan Indien Sud",
    )

    # --- Vue ensembles
    ens_png = storm_dir / "ensemble_tracks.png"
    create_ensemble_overview_png(locations_f, locations_avg, ens_png, storm_id)

    # --- Heatmap vent max
    maxwind_png = storm_dir / "max_wind.png"
    create_max_wind_heatmap(df_storm, maxwind_png, storm_id)

    print(f"    -> fichiers générés dans {storm_dir}")


# ================== MAIN ==================

def main():
    print(f"=== Génération produits IO Sud pour run {RUN_DATE:%Y-%m-%d} ===")

    print("Téléchargement des données ECMWF (download_tracks_forecast)…")
    start_date = tracks.download_tracks_forecast(RUN_DATE)

    print("Chargement des données ECMWF (create_storms_df)…")
    df_storms = tracks.create_storms_df(start_date)

    subtitle = f"Run ECMWF : {start_date:%Y-%m-%d}"

    if df_storms.empty:
        print("Aucune tempête détectée dans les données ECMWF.")
        create_placeholder_png(LATEST_DIR / "cyclogenesis.png", subtitle)
        create_placeholder_png(LATEST_DIR / "ensemble_tracks.png", subtitle)
        create_placeholder_png(LATEST_DIR / "max_wind.png", subtitle)
        create_placeholder_png(LATEST_DIR / "strike_probability.png", subtitle)
        print(f"  -> Placeholders générés dans {LATEST_DIR}")
        return

    print("Filtrage sur l'océan Indien Sud…")
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
        create_placeholder_png(LATEST_DIR / "max_wind.png", subtitle)
        create_placeholder_png(LATEST_DIR / "strike_probability.png", subtitle)
        print(f"  -> Placeholders générés dans {LATEST_DIR}")
        return

    storm_ids = sorted(df_basin.stormIdentifier.unique())
    print(f"Systèmes identifiés dans l'océan Indien Sud : {storm_ids}")

    for sid in storm_ids:
        process_storm(df_basin, sid)

    # Système principal = premier de la liste
    first = BASE_OUTPUT / f"storm_{storm_ids[0]}"

    src_cyclo = first / "strike_probability.png"
    src_ens = first / "ensemble_tracks.png"
    src_maxwind = first / "max_wind.png"

    latest_cyclo = LATEST_DIR / "cyclogenesis.png"
    latest_ens = LATEST_DIR / "ensemble_tracks.png"
    latest_maxwind = LATEST_DIR / "max_wind.png"
    compat_cyclo = LATEST_DIR / "strike_probability.png"

    for src, dests in [
        (src_cyclo, [latest_cyclo, compat_cyclo]),
        (src_ens, [latest_ens]),
        (src_maxwind, [latest_maxwind]),
    ]:
        if src.exists():
            for d in dests:
                shutil.copy(src, d)
        else:
            for d in dests:
                create_placeholder_png(d, subtitle)

    print(f"\nImages 'latest' mises à jour dans {LATEST_DIR}")
    print("\n✅ Génération terminée.")


if __name__ == "__main__":
    main()
