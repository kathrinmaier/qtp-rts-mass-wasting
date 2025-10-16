import os
import numpy as np
import rasterio
import pandas as pd
import shapely.geometry
import geopandas as gpd
from scipy.optimize import curve_fit
import helpers.util as util


def calculate_soc_attributes(
    process_dict: dict,
    location: str,
    gdf: gpd.GeoDataFrame,
    temp_path: str,
    data_path: str
) -> gpd.GeoDataFrame:
    """
    Calculates soil organic carbon (SOC) and ground ice attributes for each polygon in the provided GeoDataFrame.

    Parameters:
        process_dict (dict): Dictionary containing processing parameters and file paths.
        location (str): Location identifier for the analysis.
        gdf (pd.DataFrame): GeoDataFrame containing polygon geometries and associated data.
        temp_path (str): Path to store temporary text files.
        data_path (str): Path to input data files.

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with calculated SOC and ground ice statistics.
    """
    polygons_stats = {}
    temp_txt_file_dict = util.create_temp_txt_files(temp_path, location, process_dict)
    crs = int(process_dict['crs'])
    total = gdf.shape[0]
    for i, (idx, polygon) in enumerate(gdf.geometry.items()):
        print(f"{i} of {total}")
        stats = {}
        gdf_dh = gdf.loc[idx].drop('geometry').to_dict()
        geometry = f"'{polygon.wkt}', {crs}"
        if isinstance(gdf_dh["dh"], str):
            dh = np.array(gdf_dh["dh"].split()[1:-1], dtype=float)
        if isinstance(gdf_dh["dh_upper"], str):
            dh_upper = np.array(gdf_dh["dh_upper"].split()[1:-1], dtype=float)
        if isinstance(gdf_dh["dh_lower"], str):
            dh_lower = np.array(gdf_dh["dh_lower"].split()[1:-1], dtype=float)
        
        stats, soc_data_dict = calculate_soc(polygon, stats, temp_txt_file_dict, data_path, dh, dh_upper, dh_lower, soc=process_dict["soc"].split(), crs=crs)
        stats = calcluate_deep_soc(stats, soc_data_dict,  dh, dh_upper, dh_lower)
        stats, alt = calculate_alt(polygon, stats, temp_txt_file_dict, data_path, crs=crs)
        stats, gi_dict = calculate_ground_ice(polygon, stats, temp_txt_file_dict, data_path,  dh, dh_upper, dh_lower, alt, gi=process_dict["gi"].split(), crs=crs)
        stats = adopt_soc_to_ground_ice(stats, gi_dict, alt, soc=process_dict["soc"].split(), gi=process_dict["gi"].split())
        
        stats = util.postprocess_stats(stats)
        polygons_stats[idx] = stats
        if isinstance(polygon, shapely.geometry.Polygon):
            polygon = shapely.geometry.MultiPolygon([polygon])
        gdf.at[idx, 'geometry'] = polygon

    df = pd.DataFrame.from_dict(polygons_stats, orient='index')
    gdf = gdf.merge(df, left_index=True, right_index=True)
    util.remove_temp_txt_files(temp_txt_file_dict)
    return gdf

def calculate_soc(
    polygon: shapely.geometry.base.BaseGeometry,
    stats: dict,
    temp_txt_file_dict: dict,
    data_path: str, 
    dh: np.ndarray,
    dh_upper: np.ndarray,
    dh_lower: np.ndarray,
    soc: list,
    pixel_area: int = 100,
    crs: int = 6933
) -> tuple[dict, dict]:
    """
    Calculates SOC statistics for a given polygon.

    Parameters:
        polygon (shapely.geometry.base.BaseGeometry): Polygon geometry.
        stats (dict): Dictionary to store statistics.
        temp_txt_file_dict (dict): Dictionary of temporary text file paths.
        dh (np.ndarray): Height change values.
        dh_upper (np.ndarray): Upper height change values.
        dh_lower (np.ndarray): Lower height change values.
        soc (list): List of SOC data sources.
        pixel_area (int, optional): Area of a pixel in m^2. Default is 100.
        crs (int, optional): Coordinate reference system. Default is 6933.

    Returns:
        tuple[dict, dict]: Updated stats dictionary and SOC data dictionary.
    """
    polygon_center_x = polygon.centroid.x
    polygon_center_y = polygon.centroid.y
    
    soc_data_dict = {}
    for data in soc:
        if data == "wang":
            soc_data_dict["wang"] = create_soc_value_dict(polygon_center_x, polygon_center_y, temp_txt_file_dict["soc"], data_path, crs=crs)
            for key, value in soc_data_dict["wang"].items():
                stats[f"content_wang_{key}"] = value

    keys = ["0_1m", "1_2m", "2_3m"]
    depths = [(0.0, -1.0), (-1.0, -2.0), (-2.0, -3.0)]
    
    for soc_key, soc_dict in soc_data_dict.items():
        for depth, key in zip(depths, keys):
            if soc_dict is None or dh is None:
                stats[f"{soc_key}_soc{key}"] = stats[f"{soc_key}_soc{key}_upper"] = stats[f"{soc_key}_soc{key}_lower"] = np.nan
                stats[f"{soc_key}_soc{key}_content"] = stats[f"{soc_key}_soc{key}_content_uncert"] = np.nan
            else:
                stats[f"{soc_key}_soc{key}_content"] = len(dh) * soc_dict[f"soc{key}"] * pixel_area
                stats[f"{soc_key}_soc{key}_content_uncert"] = len(dh) * soc_dict[f"uncert{key}"] * pixel_area
                stats[f"{soc_key}_soc{key}"] = len(dh[dh < depth[0]]) * soc_dict[f"soc{key}"] * pixel_area
                stats[f"{soc_key}_soc{key}_upper"] = (len(dh_upper[dh_upper < depth[0]]) * soc_dict[f"soc{key}"] * pixel_area) + (len(dh_upper[dh_upper < depth[0]]) * soc_dict[f"uncert{key}"] * pixel_area)
                stats[f"{soc_key}_soc{key}_lower"] = (len(dh_lower[dh_lower <= depth[1]]) * soc_dict[f"soc{key}"] * pixel_area) - (len(dh_upper[dh_upper <= depth[1]]) * soc_dict[f"uncert{key}"] * pixel_area)
        
        stats[f"{soc_key}_soc0_3m_content"] = np.nansum([stats[f"{soc_key}_soc{key}_content"] for key in keys])
        stats[f"{soc_key}_soc0_3m_content_uncert"] = np.sqrt(np.nansum([stats[f"{soc_key}_soc{key}_content_uncert"]**2 for key in keys]))   
        stats[f"{soc_key}_soc0_3m"] = np.nansum([stats[f"{soc_key}_soc{key}"] for key in keys])
        stats[f"{soc_key}_soc0_3m_upper"] = np.nansum([stats[f"{soc_key}_soc{key}_upper"] for key in keys])
        stats[f"{soc_key}_soc0_3m_lower"] = np.nansum([stats[f"{soc_key}_soc{key}_lower"] for key in keys if stats[f"{soc_key}_soc{key}_lower"] > 0])
    return stats, soc_data_dict

def create_soc_value_dict(
    polygon_center_x: float,
    polygon_center_y: float,
    temp_txt_file: str,
    data_path: str,
    crs: int = 6933
) -> dict:
    """
    Retrieves SOC values and uncertainties for a given location from Wang et al. 2021 raster files: https://essd.copernicus.org/articles/13/3453/2021/ 

    Parameters:
        polygon_center_x (float): X coordinate of the polygon centroid.
        polygon_center_y (float): Y coordinate of the polygon centroid.
        temp_txt_file (str): Path to temporary text file for fallback coordinate sampling.
        data_path (str): Base path to SOC raster files.
        crs (int, optional): Coordinate reference system. Default is 6933.

    Returns:
        dict: Dictionary containing SOC and uncertainty values for 0-1m, 1-2m, and 2-3m depths.
    """
    base_path = f"{data_path}/path/to/soc/files"
    soc_files = {
        "soc1m": "TP-SOC-100.tif",
        "soc2m": "TP-SOC-200.tif",
        "soc3m": "TP-SOC-300.tif",
        "uncert1m": "TP-UN-SOC-100.tif",
        "uncert2m": "TP-UN-SOC-200.tif",
        "uncert3m": "TP-UN-SOC-300.tif"
    }
    soc_temp = {}
    
    for name, file in soc_files.items():
        soc_file = rasterio.open(os.path.join(base_path, file))
        soc_value = util.sample_data_from_coords(soc_file, [(polygon_center_x, polygon_center_y)], "soc")
        if soc_value is None:
            soc_value = util.try_other_coordinates("soc", soc_file, polygon_center_x, polygon_center_y, crs=6933)
            if soc_value is None:
                soc_value = util.read_temp_text_file("soc", soc_file, temp_txt_file)
                if soc_value is None:
                    raise ValueError("Problem with Wang SOC data.")
        else:
            with open(temp_txt_file, 'w+') as txt_file:
                txt_file.write(str([(polygon_center_x, polygon_center_y)]))
        soc_temp[name] = soc_value[0]

    soc_value_dict = {
        "soc0_1m": soc_temp["soc1m"],
        "soc1_2m": soc_temp["soc2m"] - soc_temp["soc1m"],
        "soc2_3m": soc_temp["soc3m"] - soc_temp["soc2m"],
        "uncert0_1m": soc_temp["uncert1m"],
        "uncert1_2m": np.sqrt(soc_temp["uncert2m"]**2 + soc_temp["uncert1m"]**2),
        "uncert2_3m": np.sqrt(soc_temp["uncert3m"]**2 + soc_temp["uncert2m"]**2),
    }
    return {k: max(v, 0) for k, v in soc_value_dict.items()}


def calculate_deep_soc(
    stats: dict,
    soc_data_dict: dict,
    dh: np.ndarray,
    dh_upper: np.ndarray,
    dh_lower: np.ndarray
) -> dict:
    """
    Calculates deep soil organic carbon (SOC) statistics (3-10m) using an exponential model.

    Parameters:
        stats (dict): Dictionary to store statistics.
        soc_data_dict (dict): Dictionary containing SOC values for each model/source.
        dh (np.ndarray): Height change values.
        dh_upper (np.ndarray): Upper height change values.
        dh_lower (np.ndarray): Lower height change values.

    Returns:
        dict: Updated stats dictionary with deep SOC statistics.
    """
    soc_depths = {
        "soc3_5m": [3.0, 3.5, 4.0, 4.5],
        "soc5_10m": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
    }
    for soc_key, soc_dict in soc_data_dict.items():
        stats = calculate_exponential_soc_model(stats, soc_key, soc_dict, dh, dh_upper, dh_lower, soc_depths)
    return stats

def calculate_exponential_soc_model(
    stats: dict,
    soc_key: str,
    soc_dict: dict,
    dh: np.ndarray,
    dh_upper: np.ndarray,
    dh_lower: np.ndarray,
    soc_depths: dict,
    pixel_area: int = 100
) -> dict:
    """
    Calculates deep SOC statistics (3-10m) using an exponential model fit to 0-3m SOC values.

    Parameters:
        stats (dict): Dictionary to store statistics.
        soc_key (str): Key for SOC model/source.
        soc_dict (dict): SOC values for each depth.
        dh (np.ndarray): Height change values.
        dh_upper (np.ndarray): Upper height change values.
        dh_lower (np.ndarray): Lower height change values.
        soc_depths (dict): Dictionary of depth ranges for SOC calculation.
        pixel_area (int, optional): Area of a pixel in m^2. Default is 100.

    Returns:
        dict: Updated stats dictionary with deep SOC statistics.
    """
    def exp_func(x, a, b):
        return a * np.exp(-b * x)
    if soc_dict is not None and dh is not None:
        try:
            popt, _ = curve_fit(exp_func, [1, 2, 3], [soc_dict["soc0_1m"], soc_dict["soc1_2m"], soc_dict["soc2_3m"]])
        except:
            try:
                popt, _ = curve_fit(exp_func, [1, 2, 3], [soc_dict["soc0_1m"], np.mean([soc_dict["soc0_1m"], soc_dict["soc2_3m"]]), soc_dict["soc2_3m"]])
            except:
                print("No exp model found.")
                popt = None
        uncert_below3m = soc_dict["uncert2_3m"] * pixel_area * 0.5
    else:
        popt = None
        uncert_below3m = 0

    for key, depths in soc_depths.items():
        soc, soc_upper, soc_lower, soc_content = 0, 0, 0, 0
        soc_content_uncert = []
        if soc_dict is not None and popt is not None and dh is not None:
            for d in depths:
                if exp_func(d, *popt) > 0:
                    soc_content += len(dh) * exp_func(d, *popt) * pixel_area * 0.5
                    soc_content_uncert.append((len(dh) * uncert_below3m)**2)
                    soc += len(dh[(dh < -d)]) * exp_func(d, *popt) * pixel_area * 0.5
                    soc_upper += (len(dh_upper[(dh_upper < -d)]) * exp_func(d, *popt) * pixel_area * 0.5) + (len(dh_upper[(dh_upper < -d)]) * uncert_below3m)
                    soc_lower += (len(dh_lower[(dh_lower < -d)]) * exp_func(d, *popt) * pixel_area * 0.5) - (len(dh_upper[(dh_upper < -d)]) * uncert_below3m)
        else:
            soc = soc_upper = soc_lower = soc_content = soc_content_uncert = np.nan

        stats[f"{soc_key}_exp_{key}_content"] = soc_content
        if np.isnan(soc_content_uncert).all():
            stats[f"{soc_key}_exp_{key}_content_uncert"] = np.nan
        else:
            stats[f"{soc_key}_exp_{key}_content_uncert"] = np.sqrt(sum(soc_content_uncert))
        stats[f"{soc_key}_exp_{key}"] = soc
        stats[f"{soc_key}_exp_{key}_upper"] = soc_upper
        stats[f"{soc_key}_exp_{key}_lower"] = soc_lower
    
    stats[f"{soc_key}_exp_soc_under3m_content"] = stats[f"{soc_key}_exp_soc3_5m_content"] + stats[f"{soc_key}_exp_soc5_10m_content"]
    stats[f"{soc_key}_exp_soc_under3m_content_uncert"] = np.sqrt((stats[f"{soc_key}_exp_soc3_5m_content_uncert"])**2 + (stats[f"{soc_key}_exp_soc5_10m_content_uncert"])**2)
    stats[f"{soc_key}_exp_soc_under3m"] = stats[f"{soc_key}_exp_soc3_5m"] + stats[f"{soc_key}_exp_soc5_10m"]
    stats[f"{soc_key}_exp_soc_under3m_upper"] = stats[f"{soc_key}_exp_soc3_5m_upper"] + stats[f"{soc_key}_exp_soc5_10m_upper"] 
    stats[f"{soc_key}_exp_soc_under3m_lower"] = stats[f"{soc_key}_exp_soc3_5m_lower"] + stats[f"{soc_key}_exp_soc5_10m_lower"] 
    return stats 

def calculate_alt(
    polygon: shapely.geometry.base.BaseGeometry,
    stats: dict,
    temp_txt_file_dict: dict,
    data_path: str,
    crs: int = 6933
) -> tuple[dict, float]:
    """
    Calculates the Active Layer Thickness (ALT) for a given polygon and updates the stats dictionary.

    Parameters:
        polygon (shapely.geometry.base.BaseGeometry): Polygon geometry.
        stats (dict): Dictionary to store statistics.
        temp_txt_file_dict (dict): Dictionary of temporary text file paths.
        crs (int, optional): Coordinate reference system. Default is 6933.

    Returns:
        tuple[dict, float]: Updated stats dictionary and the ALT value (rounded).
    """
    polygon_center_x = polygon.centroid.x
    polygon_center_y = polygon.centroid.y
    alt = get_alt_value(polygon_center_x, polygon_center_y, temp_txt_file_dict["alt"], data_path, crs=crs)
    stats["alt"] = alt
    alt_rounded = round(alt)
    stats["alt_rounded"] = alt_rounded
    return stats, alt_rounded

def get_alt_value(
    polygon_center_x: float,
    polygon_center_y: float,
    temp_alt_txt_file: str,
    data_path: str,
    crs: int = 6933
) -> float:
    """
    Retrieves the Active Layer Thickness (ALT) in meters for the Northern Hemisphere from Ran et al. 2022: 

    Parameters:
        polygon_center_x (float): X coordinate of the polygon centroid.
        polygon_center_y (float): Y coordinate of the polygon centroid.
        temp_alt_txt_file (str): Path to temporary text file for fallback coordinate sampling.
        crs (int, optional): Coordinate reference system. Default is 6933.

    Returns:
        float: ALT value in meters.
    """
    alt_file = f"{data_path}/path/to/alt/file"
    alt = rasterio.open(alt_file)
    alt_value = util.sample_data_from_coords(alt, [(polygon_center_x, polygon_center_y)], "alt")
    if alt_value is None:
        alt_value = util.try_other_coordinates(
            datatype="alt",
            file=alt,
            polygon_center_x=polygon_center_x,
            polygon_center_y=polygon_center_y,
            crs=crs
        )
        if alt_value is None:
            alt_value = util.read_temp_text_file("alt", alt, temp_alt_txt_file)
            if alt_value is None:
                alt_value = [200] # Average on the QTP
    else:
        with open(temp_alt_txt_file, 'w+') as file_alt:
            file_alt.write(str([(polygon_center_x, polygon_center_y)]))
    return alt_value[0] / 100.0


def calculate_ground_ice(
    polygon: shapely.geometry.base.BaseGeometry,
    stats: dict,
    temp_txt_file_dict: dict,
    data_path: str,
    dh: np.ndarray,
    dh_upper: np.ndarray,
    dh_lower: np.ndarray,
    alt: float,
    gi: list,
    pixel_area: int = 100,
    crs: int = 6933
) -> tuple[dict, dict]:
    """
    Calculates ground ice statistics for a given polygon and updates the stats dictionary.

    Parameters:
        polygon (shapely.geometry.base.BaseGeometry): Polygon geometry.
        stats (dict): Dictionary to store statistics.
        temp_txt_file_dict (dict): Dictionary of temporary text file paths.
        dh (np.ndarray): Height change values.
        dh_upper (np.ndarray): Upper height change values.
        dh_lower (np.ndarray): Lower height change values.
        alt (float): Active Layer Thickness value.
        gi (list): List of ground ice methods ("data" or others).
        pixel_area (int, optional): Area of a pixel in m^2. Default is 100.
        crs (int, optional): Coordinate reference system. Default is 6933.

    Returns:
        tuple[dict, dict]: Updated stats dictionary and ground ice data dictionary.
    """
    polygon_center_x = polygon.centroid.x
    polygon_center_y = polygon.centroid.y
    depths_dict = {
        "gi2_3m": -alt if alt < 3 else None,
        "gi3_5m": -alt if 3 <= alt < 5 else -3.0 if alt < 3 else None,
        "gi5_10m": -alt if alt >= 5 else -5.0
    }
    if not any(depths_dict.values()):
        raise ValueError("ALT value is not correct.")

    gi_dict = None
    for method in gi:
        gi_dict = get_ground_ice_data(polygon_center_x, polygon_center_y, temp_txt_file_dict["gi"], data_path, crs=crs)
        for key, depth in depths_dict.items():
            if depth is not None and dh is not None:
                perc = gi_dict[key]
                stats[f"{method}_{key}"] = len(dh[dh < depth]) * (perc / 100) * pixel_area
                stats[f"{method}_{key}_lower"] = len(dh_lower[dh_lower < depth]) * (perc / 100) * pixel_area
                stats[f"{method}_{key}_upper"] = len(dh_upper[dh_upper < depth]) * (perc / 100) * pixel_area
            else:
                stats[f"{method}_{key}"] = stats[f"{method}_{key}_lower"] = stats[f"{method}_{key}_upper"] = 0.0
    return stats, gi_dict


def get_ground_ice_data(
    polygon_center_x: float,
    polygon_center_y: float,
    temp_txt_file: str,
    data_path: str,
    crs: int = 6933
) -> dict:
    """
    Retrieves ground ice volumetric water content (VWC) data for a given location from Zuo et al. 2024 raster files.

    Parameters:
        polygon_center_x (float): X coordinate of the polygon centroid.
        polygon_center_y (float): Y coordinate of the polygon centroid.
        temp_txt_file (str): Path to temporary text file for fallback coordinate sampling.
        data_path (str): Base path to ground ice raster files.
        crs (int, optional): Coordinate reference system. Default is 6933.

    Returns:
        dict: Dictionary containing VWC values for 2-3m, 3-5m, and 5-10m depths.
    """
    if crs == 6933:
        base_path = data_path
        gi_files = {
            "gi2_3m": "VWC23.tif",
            "gi3_5m": "VWC35.tif",
            "gi5_10m": "VWC510.tif",
        }
    gi_dict = {}
    for name, path in gi_files.items():
        with rasterio.open(os.path.join(base_path, path)) as file:
            value = util.sample_data_from_coords(file, [(polygon_center_x, polygon_center_y)], "vwc")
            if value is None:
                value = util.try_other_coordinates("vwc", file, polygon_center_x, polygon_center_y, crs) 
                if value is None:
                    value = util.read_temp_text_file("vwc", file, temp_txt_file)
                    if value is None:
                        raise ValueError("Problem with VWC data.")
            else:
                with open(temp_txt_file, 'w+') as txt_file:
                    txt_file.write(str([(polygon_center_x, polygon_center_y)]))
            gi_dict[name] = value[0]
    return gi_dict


def adopt_soc_to_ground_ice(
    stats: dict,
    gi_dict: dict,
    alt: float,
    soc: list,
    gi: list
) -> dict:
    """
    Adjusts SOC statistics based on ground ice content and active layer thickness.

    Parameters:
        stats (dict): Dictionary containing SOC statistics.
        gi_dict (dict): Dictionary with ground ice volumetric water content percentages.
        alt (float): Active Layer Thickness value.
        soc (list): List of SOC data sources.
        gi (list): List of ground ice methods.

    Returns:
        dict: Updated stats dictionary with SOC values adjusted for ground ice.
    """
    gi_factors = {key: 1 - (value / 100) for key, value in gi_dict.items()}
    alt_rounded = round(alt)
    for data in soc:
        for model in ["exp"]:
            soc_values = {key: stats[f"{data}_{key}"] for key in [
                    "soc0_1m", "soc0_1m_upper", "soc0_1m_lower",  "soc0_1m_content", "soc0_1m_content_uncert",
                    "soc1_2m", "soc1_2m_upper", "soc1_2m_lower", "soc1_2m_content", "soc1_2m_content_uncert",
                    "soc2_3m", "soc2_3m_upper", "soc2_3m_lower", "soc2_3m_content", "soc2_3m_content_uncert",
                    f"{model}_soc3_5m", f"{model}_soc3_5m_upper", f"{model}_soc3_5m_lower", f"{model}_soc3_5m_content", f"{model}_soc3_5m_content_uncert",
                    f"{model}_soc5_10m", f"{model}_soc5_10m_upper", f"{model}_soc5_10m_lower", f"{model}_soc5_10m_content", f"{model}_soc5_10m_content_uncert",
            ]}

            depth_ranges = [
                (2, ["soc1_2m", "soc2_3m", f"{model}_soc3_5m", f"{model}_soc5_10m"], "gi2_3m"),
                (3, ["soc2_3m", f"{model}_soc3_5m", f"{model}_soc5_10m"], "gi2_3m"),
                (5, [f"{model}_soc3_5m", f"{model}_soc5_10m"], "gi3_5m"),
                (10, [f"{model}_soc5_10m"], "gi5_10m")
            ]
            for limit, depths, factor in depth_ranges:
                if alt_rounded < limit:
                    for depth in depths:
                        soc_values[f"{depth}"] *= gi_factors[factor]
                        soc_values[f"{depth}_upper"] *= gi_factors[factor]
                        soc_values[f"{depth}_lower"] *= gi_factors[factor]
                        soc_values[f"{depth}_content"] *= gi_factors[factor]
                        soc_values[f"{depth}_content_uncert"] *= gi_factors[factor]
                    break
            
            depths_stats = ["soc0_1m", "soc1_2m", "soc2_3m", f"{model}_soc3_5m", f"{model}_soc5_10m"]
            stats[f"gi_data_{data}_{model}_soc_column"] = sum(soc_values[f"{depth}"] for depth in depths_stats)
            stats[f"gi_data_{data}_{model}_soc_column_upper"] = sum(soc_values[f"{depth}_upper"] for depth in depths_stats)
            stats[f"gi_data_{data}_{model}_soc_column_lower"] = sum(soc_values[f"{depth}_lower"] for depth in depths_stats)
            stats[f"gi_data_{data}_{model}_soc_column_content"] = sum(soc_values[f"{depth}_content"] for depth in depths_stats)
            stats[f"gi_data_{data}_{model}_soc_column_content_uncert"] = np.sqrt(sum((soc_values[f"{depth}_content_uncert"])**2 for depth in depths_stats))
        
    for key, value in gi_dict.items():
        stats[f"gi_data_perc_{key}"] = value   
    return stats