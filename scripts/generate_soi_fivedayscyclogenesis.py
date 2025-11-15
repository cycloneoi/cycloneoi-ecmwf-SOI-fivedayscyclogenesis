#!/usr/bin/env python3
"""
CycloneOI – SOI 5-day wind hazard (proxy cyclone)
-------------------------------------------------
Produit automatique basé sur l'Open Data ECMWF :

- Paramètre : 10fgg25 = prob. rafales 10 m > 25 m/s (%)
- Domaine : Océan Indien Sud
- Sorties :
    output/soi_wg25_prob_map.png
    output/soi_wg25_prob_timeseries.png
"""

import os
import datetime as dt

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from ecmwf.opendata import Client

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Domaine Océan Indien Sud
DOMAIN = {
    "north": 5.0,
    "south": -35.0,
    "west": 20.0,
    "east": 100.0,
}

# Steps de prévision (heures) disponibles pour les probabilités ENS
STEPS = [24, 48, 72, 96, 120, 144]

# Paramètre Open Data : probabilité rafales >= 25 m/s
PARAM = "10fgg25"

GRIB_FILE = os.path.join(OUTPUT_DIR, "wg25_prob.grib2")


def latest_run_date():
    """Prend le run 00Z le plus récent, en restant prudent sur la dispo."""
    now = dt.datetime.utcnow()
    if now.hour < 6:
        run = now - dt.timedelta(days=1)
    else:
        run = now
    return run.strftime("%Y-%m-%d")


def download_data():
    date = latest_run_date()
    print(f"[INFO] Downloading ENS probabilities for {date} 00 UTC")

    client = Client(source="ecmwf")

    client.retrieve(
        date=date,
        time=0,
        stream="enfo",
        type="ep",          # ensemble probabilities
        step=STEPS,
        param=PARAM,
        target=GRIB_FILE,
    )
    print(f"[INFO] GRIB saved to {GRIB_FILE}")


def load_domain():
    if not os.path.exists(GRIB_FILE):
        raise FileNotFoundError(GRIB_FILE)

    print("[INFO] Opening GRIB with xarray/cfgrib")
    ds = xr.open_dataset(
        GRIB_FILE,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
    )

    # Le nom de variable ne peut pas commencer par un chiffre en Python.
    # On prend donc la première variable trouvée.
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]

    lat = da.latitude
    lon = da.longitude

    # Gestion du sens des latitudes
    if lat[0] > lat[-1]:
        lat_slice = slice(DOMAIN["north"], DOMAIN["south"])
    else:
        lat_slice = slice(DOMAIN["south"], DOMAIN["north"])

    lon_slice = slice(DOMAIN["west"], DOMAIN["east"])

    da_dom = da.sel(latitude=lat_slice, longitude=lon_slice)

    print(
        "[INFO] Domain subset:",
        f"lat {float(da_dom.latitude.max()):.1f} to {float(da_dom.latitude.min()):.1f},",
        f"lon {float(da_dom.longitude.min()):.1f} to {float(da_dom.longitude.max()):.1f}",
    )

    return da_dom


def build_windows(da):
    """Construit les fenêtres 24–48, 48–72, 72–96, 96–120, 120–144."""
    steps = da.step.values

    def max_range(a, b):
        valid_steps = [s for s in steps if a <= int(s) <= b]
        return da.sel(step=valid_steps).max(dim="step")

    windows = {
        "24-48": max_range(24, 48),
        "48-72": max_range(48, 72),
        "72-96": max_range(72, 96),
        "96-120": max_range(96, 120),
        "120-144": max_range(120, 144),
    }
    return windows


def make_map(da):
    """Carte du max sur 5 jours de la probabilité rafales >= 25 m/s."""
    prob_max = da.max(dim="step")

    lats = prob_max.latitude.values
    lons = prob_max.longitude.values

    lat_min = float(lats.min())
    lat_max = float(lats.max())
    lon_min = float(lons.min())
    lon_max = float(lons.max())

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        prob_max.values,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        vmin=0,
        vmax=100,
        aspect="auto",
        cmap="plasma",  # tu pourras changer la palette ici
    )
    plt.colorbar(im, label="Probabilité rafales ≥ 25 m/s (%)")
    plt.title("ECMWF ENS – Signal vent violent 5 jours (Océan Indien Sud)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "soi_wg25_prob_map.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved map to {out_path}")


def make_timeseries(windows):
    """Courbe d’évolution du signal (max spatial par fenêtre temporelle)."""
    x_days = [1.5, 2.5, 3.5, 4.5, 5.5]  # centres approx des fenêtres
    labels = ["24–48", "48–72", "72–96", "96–120", "120–144"]

    y_probs = []
    for key in ["24-48", "48-72", "72-96", "96-120", "120-144"]:
        y_probs.append(float(windows[key].max().values))

    plt.figure(figsize=(7, 4))
    plt.plot(x_days, y_probs, marker="o")
    plt.xticks(x_days, labels, rotation=30)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Fenêtre temporelle (h)")
    plt.ylabel("Proba max rafales ≥ 25 m/s (%)")
    plt.title("Évolution du signal vent violent (Océan Indien Sud)")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "soi_wg25_prob_timeseries.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved time series to {out_path}")


def main():
    try:
        download_data()
        da_dom = load_domain()
        windows = build_windows(da_dom)
        make_map(da_dom)
        make_timeseries(windows)
        print("[INFO] All products generated successfully.")
    except Exception as e:
        print("[ERROR]", e)


if __name__ == "__main__":
    main()
