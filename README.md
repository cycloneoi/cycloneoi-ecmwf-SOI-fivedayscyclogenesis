# cycloneoi-ecmwf-SOI-fivedayscyclogenesis

Cartes automatiques du signal cyclonique 5 jours (Océan Indien Sud) à partir de l'ENS ECMWF (open data).

## Produits générés

Le workflow GitHub Actions génère chaque jour :

- `output/soi_tc_prob_map.png`  
  Carte du maximum de probabilité de tempête tropicale (%) sur H+24 à H+144.

- `output/soi_tc_prob_timeseries.png`  
  Courbe montrant, pour chaque fenêtre de temps (24–72, 48–96, 72–120, 96–144 h),
  la probabilité maximale dans le domaine Océan Indien Sud.

## Intégration sur Cycloneoi

Une fois que le dépôt est public et que le workflow a tourné au moins une fois,  
tu peux intégrer les images via des URL de ce type :

- `https://raw.githubusercontent.com/<TON-USER>/cycloneoi-ecmwf-SOI-fivedayscyclogenesis/main/output/soi_tc_prob_map.png`
- `https://raw.githubusercontent.com/<TON-USER>/cycloneoi-ecmwf-SOI-fivedayscyclogenesis/main/output/soi_tc_prob_timeseries.png`
