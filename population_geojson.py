import pdfplumber
import pandas as pd
import geopandas as gpd
import json
import matplotlib.pyplot as plt


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()
    return text

def process_lines(pdf_path):
    text = extract_text(pdf_path)
    lines = text.splitlines()

    # Start after "2019 2020 2021 2022" header
    start_idx = None
    for i, line in enumerate(lines):
        if "2019 2020 2021 2022" in line:
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Start marker not found!")

    # Extract lines after the header, stop when notes/footer start
    relevant_lines = []
    for line in lines[start_idx:]:
        line_clean = line.strip()
        if line_clean.startswith("Population estimates") or line_clean.startswith("January") or line_clean == "":
            break
        relevant_lines.append(line)

    return relevant_lines


def format_lines(lines):
    header = ["Subdivision", "2019", "2020", "2021", "2022"]
    data = []

    for line in lines:
        line = ' '.join(line.split())  # remove extra spaces
        parts = line.split()

        # Sometimes the subdivision name has multiple words
        population_numbers = parts[-4:]  # last four columns are population
        subdivision_name = ' '.join(parts[:-4])  # rest is name

        # Convert numbers
        populations = [int(x.replace(',', '')) for x in population_numbers]
        data.append([subdivision_name] + populations)

    df = pd.DataFrame(data, columns=header)
    return df

# Load geojson data
def load_and_clean_geojson(path):
    """Function to load and clean the geojson file.
    Cannot use gpd.read_file due to array structure of some properties. 
    Manual investigation confirmed these arrays have a single item, so can flatten without loss of information."""

    # Load raw GeoJSON
    with open("Data/georef-canada-census-subdivision@public.geojson") as f:
        data = json.load(f)

    # Flatten any array-type fields (take the first item)
    for feature in data['features']:
        props = feature['properties']
        for k, v in props.items():
            if isinstance(v, list):
                props[k] = v[0] if v else None  # replace list with first value or None
   
    gdf = gpd.GeoDataFrame.from_features(data['features'], crs="EPSG:4326")
    return gdf

if __name__ == "__main__":
    pdf_path = "Data/fin-population-estimates-census-subdivision-2019-to-2022.pdf"
    lines = process_lines(pdf_path)
    pop_df = format_lines(lines)

    # Save to CSV
    pop_df.to_csv("Data/yukon_population_estimates_cleaned.csv", index=False)
    
    # Save 2021 data
    pop_2021 = pop_df[["Subdivision", "2021"]]
    pop_2021.rename(columns={"2021":"population"}, inplace=True)
   
    # Load census subdivisions boundary
    sub_gdf = load_and_clean_geojson("Data/georef-canada-census-subdivision@public.geojson")
    
    # Add 2021 population to subdivision geojson
    merged_total = sub_gdf.merge(pop_2021, how="left", left_on="csd_name_en", right_on="Subdivision")
    merged_inner = sub_gdf.merge(pop_2021, how="inner", left_on="csd_name_en", right_on="Subdivision")


    # Plot just the subdivisions
    fig, ax = plt.subplots(figsize=(10, 8))
    merged_total.plot(
        edgecolor="black",       # Outline all subdivisions
        facecolor="none",        # No fill color
        linewidth=0.8,           # Adjust border thickness
        ax=ax
    )

    # Add subdivision names as labels
    for idx, row in merged_total.iterrows():
        # Use the centroid of each geometry for label placement
        centroid = row.geometry.centroid
        ax.text(
            centroid.x, centroid.y, 
            row["Subdivision"],     # Replace with the column containing subdivision names
            fontsize=8, 
            ha="center"
        )

    plt.title("Yukon Census Subdivisions")
    plt.axis("off")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the census subdivisions
    merged_total.plot(
        column="population",
        cmap="viridis",
        legend=True,
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )

    # Add a white mask for areas outside the subdivisions
    ax.set_facecolor("white")

    plt.title("Yukon Census Subdivisions by Population (2021)")
    plt.axis("off")
    plt.show()

    merged_total.to_file("Data/sub_population_full.geojson", driver="GeoJSON")
    merged_inner.to_file("Data/sub_population_inner.geojson", driver="GeoJSON")




