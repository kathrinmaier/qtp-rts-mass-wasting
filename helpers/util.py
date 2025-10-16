import os
import ast
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio.features import shapes

WKT_ESRI_102025 = ('PROJCS["Asia_North_Albers_Equal_Area_Conic", '
'GEOGCS["WGS 84", '
    'DATUM["WGS_1984", '
        'SPHEROID["WGS 84",6378137,298.257223563, '
            'AUTHORITY["EPSG","7030"]], '
        'AUTHORITY["EPSG","6326"]], '
    'PRIMEM["Greenwich",0, '
        'AUTHORITY["EPSG","8901"]], '
    'UNIT["degree",0.0174532925199433, '
        'AUTHORITY["EPSG","9122"]], '
    'AUTHORITY["EPSG","4326"]], '
'PROJECTION["Albers_Conic_Equal_Area"], '
'PARAMETER["latitude_of_center",30], '
'PARAMETER["longitude_of_center",95], '
'PARAMETER["standard_parallel_1",15], '
'PARAMETER["standard_parallel_2",65], '
'PARAMETER["false_easting",0], '
'PARAMETER["false_northing",0], '
'UNIT["metre",1, '
    'AUTHORITY["EPSG","9001"]], '
'AXIS["Easting",EAST], '
'AXIS["Northing",NORTH], '
'AUTHORITY["ESRI","102025"]]')


def create_temp_txt_files(
    temp_path: str,
    location: str,
    process_dict: dict
) -> dict:
    """
    Creates a dictionary of temporary text file paths for different data types.

    Parameters:
        temp_path (str): Directory path for temporary files.
        location (str): Location identifier.
        process_dict (dict): Dictionary containing 'year_start' and 'year_end'.

    Returns:
        dict: Dictionary mapping data types to their respective file paths.
    """
    tmp_txt_file_dict = {}
    for temp_txt_file in ['soc', 'alt', 'gi']:
        tmp_txt_file_dict[temp_txt_file] = (
            f"{temp_path}/attribute_{location}_{process_dict['year_start']}_{process_dict['year_end']}_{temp_txt_file}.txt"
        )
    return tmp_txt_file_dict


def read_temp_text_file(
    datatype: str,
    file: rio.io.DatasetReader,
    temp_txt_file: str
) -> list | None:
    """
    Reads previously sampled coordinates from a temporary text file and samples values from the raster file.

    Parameters:
        datatype (str): Type of data being read (e.g., 'soc').
        file (rio.io.DatasetReader): The raster file to sample from.
        temp_txt_file (str): Path to the temporary text file containing coordinates.

    Returns:
        list or None: List of sampled values if the file exists, otherwise None.
    """
    if os.path.exists(temp_txt_file):
        print(f"Old {datatype} values used.")
        with open(temp_txt_file, 'r') as txt_file:
            temp_str = txt_file.readline()
            old_coords = ast.literal_eval(temp_str)
            temp_list = []
            for val in file.sample(old_coords):
                temp_list.append(val)
        values = temp_list[0].tolist()
        return values
    else:
        return None



def remove_temp_txt_files(tmp_txt_file_dict: dict) -> None:
    """
    Removes temporary text files specified in the dictionary if they exist.

    Parameters:
        tmp_txt_file_dict (dict): Dictionary mapping keys to temporary text file paths.

    Returns:
        None
    """
    for txt_file in tmp_txt_file_dict.values():
        if os.path.exists(txt_file):
            os.remove(txt_file)

def postprocess_stats(stats: dict) -> dict:
    """
    Rounds and formats the values in the stats dictionary, handling lists, arrays, and numeric values,
    and applies additional logic for specific keys.

    Parameters:
        stats (dict): Dictionary of calculated statistics for a polygon.

    Returns:
        dict: Updated statistics dictionary with rounded and formatted values.
    """
    for key in stats.keys():
        new_value = stats[key]
        try:
            if isinstance(new_value, (list, np.ndarray)):
                new_value = [round(val, 2) for val in new_value]
                new_value = str(new_value)
            else:
                if not key in ["location", "year_start", "year_end", "time_span"]:
                    new_value = round(new_value, 2)
                    if "lower" in key:
                        if new_value < 0:
                            new_value = 0
                    if "dh" in key or "dv" in key:
                        new_value = np.abs(new_value)
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            new_value = np.nan
        stats[key] = new_value
    return stats

def sample_data_from_coords(
    file: rio.io.DatasetReader,
    coords: list,
    datatype: str,
    soc_type: str = None
) -> list | None:
    """
    Samples data from a raster file at specified coordinates and checks for invalid values.

    Parameters:
        file (rio.io.DatasetReader): The raster file to sample from.
        coords (list): List of coordinate tuples to sample.
        datatype (str): Type of data being sampled (e.g., 'soc').
        soc_type (str, optional): Specific SOC type, e.g., 'mishra'.

    Returns:
        list or None: List of sampled values if valid, otherwise None.
    """
    temp_list = []
    for val in file.sample(coords):
        temp_list.append(val)
    file_values = temp_list[0].tolist()
    if np.any(np.array(file_values) < 0.0):
        return None
    else:
        return file_values


def read_and_write_dem_data(
    diff_dem_objects: list,
    polygon: Polygon,
    all_touched: bool = False
) -> tuple | None:
    """
    Reads DEM and error data from a list of diff_dem_objects, masks them with the given polygon,
    and returns the flattened DEM values, error values, and the file path of the selected DEM.

    Parameters:
        diff_dem_objects (list): List of dicts with 'diff_dem' and 'diff_error' file paths.
        polygon (Polygon): Shapely Polygon to mask the DEMs.
        all_touched (bool, optional): If True, all pixels touched by the polygon are included.

    Returns:
        tuple or None: (dh_new, sigma_dh, selected_file) if successful, otherwise (None, None, None).
    """
    diff_dems = []
    files = []
    errors = []
    counts = []
    for diff_dem_obj in diff_dem_objects:
        if isinstance(diff_dem_objects, list) and all(isinstance(item, dict) for item in diff_dem_objects):
            diff_dem_file = diff_dem_obj["diff_dem"]
            diff_error_file = diff_dem_obj["diff_error"]
        files.append(diff_dem_file)
        diff_dem, count = mask_tif(diff_dem_file, polygon, all_touched=all_touched)
        error, _ = mask_tif(diff_error_file, polygon, all_touched=all_touched)
        if diff_dem is not None:
            diff_dems.append(diff_dem)
            errors.append(error)
            counts.append(count)

    if len(diff_dems) > 1:
        max_index =counts.index(max(counts))
        dh = diff_dems[max_index].flatten()
        sigma_dh = errors[max_index].flatten()
    elif len(diff_dems) == 1:
        dh = diff_dems[0].flatten()
        sigma_dh = errors[0].flatten()
    else:
        return None, None, None
    dh_new = dh[~np.isnan(dh)]
    sigma_dh = sigma_dh[~np.isnan(dh)]
    sigma_dh[np.isnan(sigma_dh)] = 0.0
    if len(dh_new) == 0:
        return None, None, None
    return dh_new, sigma_dh, files[0]


def rasterize_polygon(polygon: Polygon, file_path: str) -> Polygon:
    """
    Rasterizes a given polygon using the spatial reference and shape of the provided raster file,
    then returns the largest resulting polygon from the rasterization.

    Parameters:
        polygon (Polygon): The input Shapely polygon to rasterize.
        file_path (str): Path to the raster file used for reference.

    Returns:
        Polygon: The largest polygon resulting from rasterization.
    """
    with rio.open(file_path) as src:
        transform = src.transform
        crs = src.crs

    out_shape = (src.height, src.width)
    rasterized = rio.features.rasterize( # type: ignore
        [(polygon, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    )
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            shapes(rasterized, mask=None, transform=transform)
        )
        if v == 1  # Only keep polygons with value 1
    )
    geoms = list(results)
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
    new_polygon = gdf.geometry.iloc[0]
    return new_polygon


def is_contained_in_mask(
    polygon: Polygon,
    tif_file: str,
    zero_threshold: float = 0.5
) -> bool:
    """
    Checks if the given polygon, when masked against the raster file, contains a percentage of zero values
    greater than or equal to the specified threshold.

    Parameters:
        polygon (Polygon): The polygon to check.
        tif_file (str): Path to the raster file.
        zero_threshold (float, optional): Threshold for zero value percentage (default is 0.5).

    Returns:
        bool: True if zero percentage >= threshold or on error, False otherwise.
    """
    try:
        with rio.open(tif_file) as src:
            geom = mapping(polygon)
            out_img, _ = mask(src, [geom], crop=True)
            out_img = np.where((out_img > 8000) | (out_img < -500), np.nan, out_img)
            zero_count = (out_img == 0.0).sum()
            total_pixels = np.count_nonzero(~np.isnan(out_img))
            zero_percentage = zero_count / total_pixels
        return True if zero_percentage >= zero_threshold else False
    except Exception:
        return True


def try_other_coordinates(
    datatype: str,
    file: rio.io.DatasetReader,
    polygon_center_x: float,
    polygon_center_y: float,
    crs: int,
    dst_crs: int = None,
    soc_type: str = None
) -> list | None:
    """
    Attempts to sample data from a raster file at shifted coordinates around the polygon center.

    Parameters:
        datatype (str): Type of data being sampled.
        file (rio.io.DatasetReader): The raster file to sample from.
        polygon_center_x (float): X coordinate of the polygon center.
        polygon_center_y (float): Y coordinate of the polygon center.
        crs (int): Source coordinate reference system EPSG code.
        dst_crs (int, optional): Destination coordinate reference system EPSG code.
        soc_type (str, optional): Specific SOC type, e.g., 'mishra'.

    Returns:
        list or None: List of sampled values if valid, otherwise None.
    """
    shifts = [
        (1000, 1000), (-1000, -1000), (-1000, 1000), (1000, -1000),
        (5000, 5000), (-5000, -5000), (-5000, 5000), (5000, -5000),
        (10000, 10000), (-10000, -10000), (-10000, 10000), (10000, -10000)
    ]
    for x, y in shifts:
        new_x = polygon_center_x + x
        new_y = polygon_center_y + y
        new_coords = [(new_x, new_y)]
        file_values = sample_data_from_coords(file, new_coords, datatype, soc_type)
        if file_values is not None:
            return file_values
    return None
