#!/usr/bin/env python3
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

# Steps disponibles via ECMWF OpenData
STEPS = [24, 48, 72, 96, 120, 144]

# Paramètre de probabilité de tempête tropicale
PARAM = "ptc"


def latest_run():
    now = dt.datetime.utcnow()
    if now.hour < 6:
        return (now - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d")


def download_data():
    date = latest_run()
    print(f"[INFO] Downloading ENS probabilities for {date} 00 UTC")

    client = Client()

    client.retrieve(
        date=date,
        time=0,
        stream="enfo",
        type="ep",
        step=STEPS,
        param=PARAM,
        target=f"{OUTPUT_DIR}/data.grib2"
    )


def load_domain():
    ds = xr.open_dataset(
        f"{OUTPUT_DIR}/data.grib2",
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": PARAM}},
    )

    da = ds[PARAM]

    lat = da.latitude
    lon = da.longitude

    if lat[0] > lat[-1]:
        lat_slice = slice(DOMAIN["north"], DOMAIN["south"])
    else:
        lat_slice = slice(DOMAIN["south"], DOMAIN["north"])

    lon_slice = slice(DOMAIN["west"], DOMAIN["east"])

    return da.sel(latitude=lat_slice, longitude=lon_slice)


def build_windows(da):
    """Construit les fenêtres 24–72, 48–96, 72–120, 96–144"""
    steps = da.step.values

    def max_range(a, b):
        return da.sel(step=[s for s in steps if a <= s <= b]).max(dim="step")

    win = {
        "24-72": max_range(24, 72),
        "48-96": max_range(48, 96),
        "72-120": max_range(72, 120),
        "96-144": max_range(96, 144),
    }

    return win


def make_map(da):
    """Carte max sur 5 jours"""
    prob = da.max(dim="step")

    plt.figure(figsize=(8, 6))
    plt.imshow(
        prob,
        origin="lower",
        extent=[
            float(prob.longitude.min()),
            float(prob.longitude.max()),
            float(prob.latitude.min()),
            float(prob.latitude.max())
        ],
        cmap="plasma",
        vmin=0,
        vmax=100
    )
    plt.colorbar(label="Probabilité (%)")
    plt.title("Signal cyclonique 5 jours – Probabilité de tempête tropicale")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/soi_tc_prob_map.png", dpi=150)
    plt.close()


def make_timeseries(windows):
    """Courbe d’évolution du signal dans le domaine"""
    x = [2, 3, 4, 5]  # centres approximatifs des fenêtres (jours)
    y = [
        float(windows["24-72"].max().values),
        float(windows["48-96"].max().values),
        float(windows["72-120"].max().values),
        float(windows["96-144"].max().values),
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Échéance (jours)")
    plt.ylabel("Proba max (%)")
    plt.title("Évolution du signal cyclonique (max domaine)")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/soi_tc_prob_timeseries.png", dpi=150)
    plt.close()


def main():
    try:
        download_data()
        da = load_domain()
        windows = build_windows(da)
        make_map(da)
        make_timeseries(windows)
        print("[INFO] All plots generated successfully.")
    except Exception as e:
        print("[ERROR]", e)


if __name__ == "__main__":
    main()
