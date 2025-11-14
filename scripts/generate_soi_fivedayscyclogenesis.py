#!/usr/bin/env python3
"""
Generate 5-day tropical cyclone signal (Southern Indian Ocean)
from ECMWF ENS open data, and export:
  1) a map (max probability over next 5 days)
  2) a time series (max probability vs lead time)

Outputs:
  output/soi_tc_prob_map.png
  output/soi_tc_prob_timeseries.png
"""

import os
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ecmwf.opendata import Client

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

# Domaine Océan Indien Sud (ajuste si tu veux)
# (Nord, Ouest, Sud, Est)
DOMAIN = {
    "north": 5.0,    # 5°N
    "south": -35.0,  # 35°S
    "west": 20.0,    # 20°E
    "east": 100.0,   # 100°E
}

# Fenêtres de prévision pour le "signal 5 jours"
# (format ECMWF step range)
STEP_RANGES = ["24-72", "48-96", "72-120", "96-144"]

# Paramètre probabilité de tempête tropicale
# (shortName = pts, paramId = 131089)  :contentReference[oaicite:1]{index=1}
PARAM = "pts"

OUTPUT_DIR = "output"
GRIB_FILE = os.path.join(OUTPUT_DIR, "soi_tc_prob.grib2")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Helper : choisir la date du run 00 UTC le plus récent
# --------------------------------------------------------------------
def latest_00z_run_date(now_utc: dt.datetime) -> dt.date:
    """
    Si on est avant 06 UTC, on prend le 00 UTC de la veille
    (les données open data sont parfois un peu en retard).
    Sinon on prend le 00 UTC du jour.
    """
    if now_utc.hour < 6:
        return (now_utc - dt.timedelta(days=1)).date()
    return now_utc.date()


# --------------------------------------------------------------------
# 1. Télécharger les probabilités de tempête tropicale ENS (type=ep)
# --------------------------------------------------------------------
def download_tc_probabilities():
    now = dt.datetime.utcnow()
    run_date = latest_00z_run_date(now)
    date_str = run_date.strftime("%Y-%m-%d")

    print(f"[INFO] Using ENS run {date_str} 00 UTC")

    client = Client(source="ecmwf")

    # Construction de la liste de pas de temps (step ranges)
    steps = STEP_RANGES

    print(f"[INFO] Downloading probability fields: param={PARAM}, steps={steps}")

    client.retrieve(
        date=date_str,
        time=0,              # 00 UTC
        stream="enfo",       # ENS
        type="ep",           # ensemble probability products
        param=PARAM,
        step=steps,
        target=GRIB_FILE,
    )

    print(f"[INFO] Downloaded GRIB file to {GRIB_FILE}")


# --------------------------------------------------------------------
# 2. Charger les données GRIB et découper sur le domaine SOI
# --------------------------------------------------------------------
def load_domain_data():
    if not os.path.exists(GRIB_FILE):
        raise FileNotFoundError(f"GRIB file not found: {GRIB_FILE}")

    print(f"[INFO] Opening GRIB file with xarray/cfgrib: {GRIB_FILE}")
    ds = xr.open_dataset(
        GRIB_FILE,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": PARAM}},
    )

    # Le nom de variable peut être "pts" ou "unknown"
    if PARAM in ds.data_vars:
        var_name = PARAM
    else:
        # fallback: on prend la première variable
        var_name = list(ds.data_vars)[0]
        print(f"[WARN] Variable '{PARAM}' not found, using '{var_name}' instead.")

    da = ds[var_name]

    # Découpe sur le domaine (attention, latitude est souvent décroissante)
    lat = da.latitude
    lon = da.longitude

    north = DOMAIN["north"]
    south = DOMAIN["south"]
    west = DOMAIN["west"]
    east = DOMAIN["east"]

    if lat[0] > lat[-1]:
        lat_slice = slice(north, south)
    else:
        lat_slice = slice(south, north)

    if lon[0] < lon[-1]:
        lon_slice = slice(west, east)
    else:
        lon_slice = slice(east, west)

    da_dom = da.sel(latitude=lat_slice, longitude=lon_slice)

    print(
        f"[INFO] Domain subset: "
        f"lat {float(da_dom.latitude.max()):.1f} to {float(da_dom.latitude.min()):.1f}, "
        f"lon {float(da_dom.longitude.min()):.1f} to {float(da_dom.longitude.max()):.1f}"
    )

    return da_dom


# --------------------------------------------------------------------
# 3. Construire la carte "signal 5 jours" (max de proba sur les steps)
# --------------------------------------------------------------------
def make_map(da_dom: xr.DataArray):
    # da_dom dims: step, latitude, longitude
    prob_max = da_dom.max(dim="step")  # max proba sur les 4 fenêtres

    # Récupérer les bornes géographiques pour l'affichage
    lats = prob_max.latitude.values
    lons = prob_max.longitude.values

    lat_min = float(lats.min())
    lat_max = float(lats.max())
    lon_min = float(lons.min())
    lon_max = float(lons.max())

    data = prob_max.values

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        data,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        vmin=0,
        vmax=100,
        aspect="auto",
        cmap="plasma",  # Palette personnalisable
    )
    plt.colorbar(im, label="Probabilité de tempête tropicale (%)")

    plt.title(
        "ECMWF ENS – Signal cyclonique 5 jours\n"
        "Max probabilité de tempête tropicale (H+24 à H+144)"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "soi_tc_prob_map.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved map to {out_path}")


# --------------------------------------------------------------------
# 4. Construire la courbe d’évolution du signal dans le domaine
# --------------------------------------------------------------------
def make_timeseries(da_dom: xr.DataArray):
    # Max spatial pour chaque fenêtre :
    #   -> "intensité du signal" par pas de temps
    prob_step_max = da_dom.max(dim=("latitude", "longitude"))

    # Les coord 'step' sont des Timedelta (en heures) ou des chaînes.
    # On les convertit en heures puis jours pour l'axe X.
    step_coord = prob_step_max["step"]

    if np.issubdtype(step_coord.dtype, np.timedelta64):
        step_hours = step_coord.values / np.timedelta64(1, "h")
    else:
        # step peut être "24-72", "48-96", ... -> on prend le centre
        step_strs = [str(s) for s in step_coord.values]
        centres = []
        for s in step_strs:
            if "-" in s:
                a, b = s.split("-")
                centres.append((float(a) + float(b)) / 2.0)
            else:
                centres.append(float(s))
        step_hours = np.array(centres)

    step_days = step_hours / 24.0
    probs = prob_step_max.values

    plt.figure(figsize=(7, 4))
    plt.plot(step_days, probs, marker="o")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Échéance (jours depuis le run 00 UTC)")
    plt.ylabel("Proba max de tempête tropicale (%)")
    plt.title("Évolution du signal cyclonique sur 5 jours\n(SOI – max spatial de la probabilité)")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "soi_tc_prob_timeseries.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved time series to {out_path}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    try:
        download_tc_probabilities()
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        return

    try:
        da_dom = load_domain_data()
    except Exception as exc:
        print(f"[ERROR] Failed to load GRIB data: {exc}")
        return

    try:
        make_map(da_dom)
        make_timeseries(da_dom)
    except Exception as exc:
        print(f"[ERROR] Failed to generate plots: {exc}")
        return


if __name__ == "__main__":
    main()

