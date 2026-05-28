"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import maplibregl, { type Map as MLMap, type MapMouseEvent } from "maplibre-gl";

import {
  SATELLITE_STYLE,
  STREET_STYLE,
  INITIAL_VIEW,
  WATER_SOURCE_ID,
  WATER_LAYER_ID,
  LANDMASK_SOURCE_ID,
  LANDMASK_LAYER_ID,
} from "@/lib/map-config";
import type { Bbox } from "@/lib/types";
import { Globe2, Map as MapIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type Basemap = "satellite" | "street";

interface MapViewProps {
  onBboxDrawn?: (bbox: Bbox | null) => void;
  /** Two-layer overlay: water (red/blue, constant opacity) + landmask (gray, slider-controlled). */
  predictionOverlay?: { waterUrl: string; landmaskUrl: string; bbox: Bbox } | null;
  /** Opacity (0..1) applied ONLY to the gray non-water landmask layer. Water stays at 1.0. */
  landmaskOpacity?: number;
  drawingEnabled: boolean;
}

const PREVIEW_SOURCE = "bbox-draw-preview-src";
const PREVIEW_FILL   = "bbox-draw-preview-fill";
const PREVIEW_LINE   = "bbox-draw-preview-line";

function bboxAsPolygon(lonMin: number, latMin: number, lonMax: number, latMax: number) {
  return {
    type: "Feature" as const,
    properties: {},
    geometry: {
      type: "Polygon" as const,
      coordinates: [[
        [lonMin, latMin], [lonMax, latMin],
        [lonMax, latMax], [lonMin, latMax],
        [lonMin, latMin],
      ]],
    },
  };
}

export function MapView({
  onBboxDrawn,
  predictionOverlay,
  landmaskOpacity = 0.6,
  drawingEnabled,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const corner1Ref = useRef<[number, number] | null>(null);
  const [basemap, setBasemap] = useState<Basemap>("satellite");
  const [hint, setHint] = useState<string>("Click two corners on the map to draw a rectangle");

  // -------- map init --------
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: SATELLITE_STYLE,
      center: [INITIAL_VIEW.longitude, INITIAL_VIEW.latitude],
      zoom: INITIAL_VIEW.zoom,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 120 }), "bottom-left");
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // -------- basemap swap (preserve overlay/preview after style reload) --------
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.setStyle(basemap === "satellite" ? SATELLITE_STYLE : STREET_STYLE);
  }, [basemap]);

  // -------- click-twice rectangle drawing --------
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const cleanup = () => {
      corner1Ref.current = null;
      if (map.getLayer(PREVIEW_FILL)) map.removeLayer(PREVIEW_FILL);
      if (map.getLayer(PREVIEW_LINE)) map.removeLayer(PREVIEW_LINE);
      if (map.getSource(PREVIEW_SOURCE)) map.removeSource(PREVIEW_SOURCE);
      map.getCanvas().style.cursor = "";
      setHint("Click two corners on the map to draw a rectangle");
    };

    if (!drawingEnabled) {
      cleanup();
      onBboxDrawn?.(null);
      return;
    }

    map.getCanvas().style.cursor = "crosshair";
    setHint("Click the first corner");

    const upsertPreview = (lonMin: number, latMin: number, lonMax: number, latMax: number) => {
      const data = bboxAsPolygon(lonMin, latMin, lonMax, latMax);
      const src = map.getSource(PREVIEW_SOURCE) as maplibregl.GeoJSONSource | undefined;
      if (src) {
        src.setData(data as GeoJSON.Feature);
      } else {
        map.addSource(PREVIEW_SOURCE, { type: "geojson", data: data as GeoJSON.Feature });
        map.addLayer({
          id: PREVIEW_FILL,
          type: "fill",
          source: PREVIEW_SOURCE,
          paint: { "fill-color": "#2383E2", "fill-opacity": 0.15 },
        });
        map.addLayer({
          id: PREVIEW_LINE,
          type: "line",
          source: PREVIEW_SOURCE,
          paint: { "line-color": "#2383E2", "line-width": 2 },
        });
      }
    };

    const onClick = (e: MapMouseEvent) => {
      const { lng, lat } = e.lngLat;
      if (!corner1Ref.current) {
        corner1Ref.current = [lng, lat];
        setHint("Click the second corner");
        return;
      }
      const [x1, y1] = corner1Ref.current;
      const lonMin = Math.min(x1, lng);
      const lonMax = Math.max(x1, lng);
      const latMin = Math.min(y1, lat);
      const latMax = Math.max(y1, lat);
      upsertPreview(lonMin, latMin, lonMax, latMax);
      onBboxDrawn?.([lonMin, latMin, lonMax, latMax]);
      corner1Ref.current = null;
      setHint("Drawn. Click again to redraw, or hit \"Predict on selection\".");
    };

    const onMove = (e: MapMouseEvent) => {
      if (!corner1Ref.current) return;
      const [x1, y1] = corner1Ref.current;
      const { lng, lat } = e.lngLat;
      upsertPreview(
        Math.min(x1, lng), Math.min(y1, lat),
        Math.max(x1, lng), Math.max(y1, lat),
      );
    };

    map.on("click", onClick);
    map.on("mousemove", onMove);

    return () => {
      map.off("click", onClick);
      map.off("mousemove", onMove);
      cleanup();
    };
  }, [drawingEnabled, onBboxDrawn]);

  // -------- prediction overlay (two layers: landmask underneath, water on top) --------
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const apply = () => {
      // Tear down anything previously added (covers both new + legacy IDs)
      [WATER_LAYER_ID, LANDMASK_LAYER_ID].forEach((id) => {
        if (map.getLayer(id)) map.removeLayer(id);
      });
      [WATER_SOURCE_ID, LANDMASK_SOURCE_ID].forEach((id) => {
        if (map.getSource(id)) map.removeSource(id);
      });

      if (!predictionOverlay) return;
      const { waterUrl, landmaskUrl, bbox } = predictionOverlay;
      const [lonMin, latMin, lonMax, latMax] = bbox;
      const coords: [[number, number], [number, number], [number, number], [number, number]] = [
        [lonMin, latMax], [lonMax, latMax],
        [lonMax, latMin], [lonMin, latMin],
      ];

      // Layer 1 (added first → renders BELOW): gray landmask, slider-controlled
      map.addSource(LANDMASK_SOURCE_ID, {
        type: "image", url: landmaskUrl, coordinates: coords,
      });
      map.addLayer({
        id: LANDMASK_LAYER_ID,
        type: "raster",
        source: LANDMASK_SOURCE_ID,
        paint: { "raster-opacity": landmaskOpacity },
      });

      // Layer 2 (added second → renders ON TOP): red/blue water, ALWAYS opaque
      map.addSource(WATER_SOURCE_ID, {
        type: "image", url: waterUrl, coordinates: coords,
      });
      map.addLayer({
        id: WATER_LAYER_ID,
        type: "raster",
        source: WATER_SOURCE_ID,
        paint: { "raster-opacity": 1.0 },
      });

      map.fitBounds(
        [[lonMin, latMin], [lonMax, latMax]],
        { padding: 60, duration: 1500 },
      );
    };

    if (map.isStyleLoaded()) apply();
    else map.once("load", apply);
    // landmaskOpacity intentionally omitted — handled by dedicated effect below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionOverlay]);

  // -------- live landmask opacity (water layer never changes) --------
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!map.getLayer(LANDMASK_LAYER_ID)) return;
    map.setPaintProperty(LANDMASK_LAYER_ID, "raster-opacity", landmaskOpacity);
  }, [landmaskOpacity]);

  return (
    <div className="absolute inset-0">
      <div ref={containerRef} className="h-full w-full" />

      {/* Floating basemap toggle — pill-style segment with sliding indicator */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 z-10 animate-slide-in-t">
        <div
          role="tablist"
          aria-label="Basemap"
          className="relative inline-flex glass-pill rounded-full p-1"
        >
          {/* sliding thumb */}
          <span
            aria-hidden
            className="absolute top-1 bottom-1 rounded-full bg-canvas shadow-soft
                       transition-all duration-300 ease-out-expo"
            style={{
              left: basemap === "satellite" ? "4px" : "calc(50% - 0px)",
              width: "calc(50% - 4px)",
            }}
          />
          {(["satellite", "street"] as const).map((bm) => (
            <button
              key={bm}
              role="tab"
              type="button"
              aria-selected={basemap === bm}
              onClick={() => setBasemap(bm)}
              className={cn(
                "relative z-10 inline-flex items-center gap-1.5 px-4 h-8 rounded-full",
                "text-[11px] font-medium tracking-wide cursor-pointer transition-colors duration-200",
                basemap === bm ? "text-ink" : "text-muted hover:text-text",
              )}
            >
              {bm === "satellite" ? (
                <Globe2 size={12} strokeWidth={1.75} />
              ) : (
                <MapIcon size={12} strokeWidth={1.75} />
              )}
              {bm === "satellite" ? "Satellite" : "Street"}
            </button>
          ))}
        </div>
      </div>

      {/* Drawing hint chip */}
      {drawingEnabled && (
        <div
          className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 px-4 py-2
                     glass-pill rounded-full text-[12px] text-text
                     flex items-center gap-2 animate-slide-in-t"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-dot" />
          {hint}
        </div>
      )}
    </div>
  );
}
