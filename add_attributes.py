import os
import geopandas as gpd
from helpers import config_parser as config
import helpers.calculate_mass_wasting as mass_wasting
import helpers.calculate_soc as soc

ROOT = "./" # Change this to your project root
TEMP = "./temp"   # Change this to your temp directory
DATA = "/path/to/data" # Change this to your data directory

def main():
    """Add attributes to the polygons based on TanDEM-X DEMs."""
    os.makedirs(f"{ROOT}/attributes", exist_ok=True)
    ini_location = 'ini/attributes.ini'
    process_dict = config.get_config_dict(filename=ini_location,
                                            section='general')
    for location in process_dict['sites'].split():
        print(f"Tibet {location}: {process_dict['year_start']} - {process_dict['year_end']}")
        gdf = gpd.read_file(f"{ROOT}/raw/polygons_{process_dict['data_type']}_{location}_{process_dict['year_start']}_{process_dict['year_end']}.geojson", crs=f"EPSG:{process_dict['crs']}")
        gdf = mass_wasting.calculate_polygon_attributes(process_dict, gdf, location, TEMP, DATA)
        gdf.to_file(f"{ROOT}/attributes/volume_polygons_{process_dict['data_type']}_{location}_{process_dict['year_start']}_{process_dict['year_end']}.geojson", driver="GeoJSON", crs=f"EPSG:{process_dict['crs']}")
        gdf = soc.calculate_soc_attributes(process_dict, location, gdf, TEMP, DATA)
        gdf.to_file(f"{ROOT}/attributes/soc_polygons_{process_dict['data_type']}_{location}_{process_dict['year_start']}_{process_dict['year_end']}.geojson", driver="GeoJSON", crs=f"EPSG:{process_dict['crs']}")
    return

  
if __name__ == "__main__":
    main()  