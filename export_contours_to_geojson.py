import json
import shapely.geometry as geom

def export_contours_to_geojson(contour_results: dict, output_filepath: str, metadata: dict = None):
    """
    Exports core and margin contours as standardized GeoJSON polygons.
    """
    features = []

    def _contour_to_polygon(contour, label, hex_color, roi_id):
        coords = contour.squeeze()
        if len(coords) < 3:
            return None  # Polygons require at least 3 distinct vertices

        # Close the polygon loop if not already closed
        poly_points = coords.tolist()
        if poly_points[0] != poly_points[-1]:
            poly_points.append(poly_points[0])

        polygon = geom.Polygon(poly_points)
        # Simplify polygon slightly to eliminate pixel stair-stepping artifacts
        simplified = polygon.simplify(tolerance=1.0, preserve_topology=True)

        return {
            "type": "Feature",
            "id": f"{label}_{roi_id}",
            "properties": {
                "label": label,
                "color": hex_color,
                "area_pixels": float(simplified.area),
                "is_core": (label == "Core_Nidus"),
                **(metadata or {})
            },
            "geometry": geom.mapping(simplified)
        }

    # Process Core Contours (Red: #FF0000)
    for idx, c in enumerate(contour_results.get("core_contours", [])):
        feat = _contour_to_polygon(c, "Core_Nidus", "#FF0000", idx)
        if feat:
            features.append(feat)

    # Process Margin Contours (Cyan: #00FFFF)
    for idx, c in enumerate(contour_results.get("margin_contours", [])):
        feat = _contour_to_polygon(c, "Peritumoral_Margin", "#00FFFF", idx)
        if feat:
            features.append(feat)

    geojson_dict = {
        "type": "FeatureCollection",
        "metadata": {
            "coordinate_system": "pixel_space",
            "generator": "ViTXAIEnsembler"
        },
        "features": features
    }

    with open(output_filepath, "w") as f:
        json.dump(geojson_dict, f, indent=2)

    print(f"Exported {len(features)} polygon structures to {output_filepath}")