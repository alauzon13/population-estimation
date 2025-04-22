// Initialize the map
var basicMap = L.map('myMap').setView([64, -135], 5); // Adjust coordinates and zoom as needed

// Add a tile layer (OpenStreetMap)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}).addTo(basicMap);

// Load the buildings GeoJSON (mixed points and polygons)
fetch('../Data/clipped_data/clipped_buildings.geojson')
    .then(response => response.json())
    .then(data => {
        console.log("Loaded Buildings GeoJSON:", data);
        L.geoJSON(data, {
            style: function (feature) {
                // Apply styles only to polygons
                if (feature.geometry.type === "Polygon" || feature.geometry.type === "MultiPolygon") {
                    return {
                        color: "#FF4136",
                        weight: 1,
                        fillColor: "#FF851B",
                        fillOpacity: 0.5
                    };
                }
            },
            pointToLayer: function (feature, latlng) {
                // Style for point geometries
                return L.circleMarker(latlng, {
                    radius: 5,
                    fillColor: "#2ECC40",
                    color: "#fff",
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                });
            },
            onEachFeature: function (feature, layer) {
                // Display basic property info in a popup
                let popupContent = "";
                if (feature.properties) {
                    popupContent += `<strong>${feature.properties.name || 'Unnamed Building'}</strong><br>`;
                }
                layer.bindPopup(popupContent);
            }
        }).addTo(basicMap);
        // Fit the map view to the bounds of the GeoJSON layer
        const bounds = geoJsonLayer.getBounds();
        basicMap.fitBounds(bounds);

        // Restrict the map view to the Yukon boundary
        basicMap.setMaxBounds(bounds);
    })
    .catch(error => console.error('Error loading clipped_buildings.geojson:', error));
