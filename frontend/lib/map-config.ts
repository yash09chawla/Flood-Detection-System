// MapLibre style definitions using free tile providers (no API key needed).

import type { StyleSpecification } from "maplibre-gl";

const ESRI_ATTRIBUTION =
  'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, ' +
  "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community";

const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export const SATELLITE_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    "esri-imagery": {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution: ESRI_ATTRIBUTION,
      maxzoom: 19,
    },
  },
  layers: [
    {
      id: "esri-imagery-layer",
      type: "raster",
      source: "esri-imagery",
    },
  ],
};

export const STREET_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: [
        "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution: OSM_ATTRIBUTION,
      maxzoom: 19,
    },
  },
  layers: [
    {
      id: "osm-layer",
      type: "raster",
      source: "osm",
    },
  ],
};

export const INITIAL_VIEW = {
  longitude: 80.0,
  latitude: 22.0,
  zoom: 3.5,
};

// IDs we'll add to the map dynamically for prediction overlays + drawn shapes
export const WATER_SOURCE_ID    = "prediction-water-src";    // red/blue, constant opacity
export const WATER_LAYER_ID     = "prediction-water-layer";
export const LANDMASK_SOURCE_ID = "prediction-landmask-src"; // gray non-water, slider-controlled
export const LANDMASK_LAYER_ID  = "prediction-landmask-layer";
// Legacy single-overlay IDs (kept for fallback when the backend doesn't ship two layers)
export const PREDICTION_SOURCE_ID = "prediction-overlay-src";
export const PREDICTION_LAYER_ID = "prediction-overlay-layer";
export const POLYGON_SOURCE_ID = "user-polygon-src";
export const POLYGON_OUTLINE_LAYER_ID = "user-polygon-outline";
export const POLYGON_FILL_LAYER_ID = "user-polygon-fill";
