import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape
import json
import contextily as ctx
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Load geopackage data
gdf = gpd.read_file("Data/OSM_exports/Yukon_gpkg_uid_9cf9d9be-ab7a-413b-8de5-50db2e3aa7f2/Yukon.gpkg")

# Load territorial boundary
with open("Data/OSM_exports/georef-canada-province@public.geojson") as f:
    boundary_data = json.load(f)
boundary_poly = shape(boundary_data['features'][0]['geometry'])

# Convert the geometry to a GeoDataFrame
boundary_gdf = gpd.GeoDataFrame(index=[0], geometry=[boundary_poly], crs="EPSG:4326")


# Clip the original GeoDataFrame to the provincial boundary
gdf_clipped = gpd.clip(gdf, boundary_gdf)

# Save clipped gdf as geojson
gdf_clipped.to_file("Data/clipped_data/clipped_geojson.geojson", driver="GeoJSON") 

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 12))

# Plot geometries with transparency
gdf_clipped[gdf_clipped.geometry.type == 'Point'].plot(ax=ax, color='blue', markersize=10, alpha=0.4)
gdf_clipped[gdf_clipped.geometry.type == 'LineString'].plot(ax=ax, color='orange', linewidth=1, alpha=0.4)
gdf_clipped[gdf_clipped.geometry.type == 'MultiLineString'].plot(ax=ax, color='purple', linewidth=1, alpha=0.4)
gdf_clipped[gdf_clipped.geometry.type == 'Polygon'].plot(ax=ax, facecolor='green', edgecolor='black', alpha=0.1)
gdf_clipped[gdf_clipped.geometry.type == 'MultiPolygon'].plot(ax=ax, facecolor='red', edgecolor='black', alpha=0.1)

# Add basemap
ctx.add_basemap(ax, crs=gdf_clipped.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)

# Build custom legend
legend_handles = [
    mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Point'),
    mlines.Line2D([], [], color='orange', linewidth=2, label='LineString'),
    mlines.Line2D([], [], color='purple', linewidth=2, label='MultiLineString'),
    mpatches.Patch(facecolor='green', edgecolor='black', alpha=0.3, label='Polygon'),
    mpatches.Patch(facecolor='red', edgecolor='black', alpha=0.3, label='MultiPolygon')
]

ax.legend(handles=legend_handles, loc='lower left')
ax.set_title("Geometry Types Overlaid on Basemap", fontsize=16)
ax.set_axis_off()

plt.tight_layout()
plt.show()


# Create smaller gdfs for interactive map
gdf_clipped_points = gdf_clipped[gdf_clipped.geometry.type == "Point"]
gdf_clipped_points.to_file("Data/clipped_data/clipped_points.geojson", driver="GeoJSON")
gdf_clipped_poly = gdf_clipped[gdf_clipped.geometry.type == "Polygon"]
gdf_clipped_poly.to_file("Data/clipped_data/clipped_poly.geojson", driver="GeoJSON")
gdf_clipped_buildings = gdf_clipped[gdf_clipped["building"].notnull()]
gdf_clipped_buildings.to_file("Data/clipped_data/clipped_buildings.geojson", driver="GeoJSON")


