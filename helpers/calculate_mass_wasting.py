import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import helpers.util as util
import helpers.soc_ground_ice_alt as soc_calc
import helpers.calculate_height as height_calc
import helpers.calculate_volume as volume_calc

def calculate_polygon_attributes(
    process_dict: dict,
    gdf: gpd.GeoDataFrame,
    location: str,
    temp_path: str,
    data_path: str,
) -> tuple[gpd.GeoDataFrame, list]:
    """
    Calculates and adds attributes to each polygon in the provided GeoDataFrame.

    Parameters:
        process_dict (dict): Dictionary containing processing parameters.
        gdf (gpd.GeoDataFrame): GeoDataFrame containing polygons to process.
        location (str): Location identifier for temporary files.
        temp_path (str): Path to store temporary files.
        data_path (str): Path to input data files.
        years (list, optional): List of years for additional processing.

    Returns:
        geopandas.GeoDataFrame: Updated GeoDataFrame with attributes.
    """
    tmp_txt_file_dict = util.create_temp_txt_files(temp_path, location, process_dict)
    crs = int(process_dict['crs'])
    polygons_stats = {}
    year_dict = {}
    total = gdf.shape[0]

    for i, (idx, polygon) in enumerate(gdf.geometry.items()):
        print(f"{i} of {total}")
        geometry = f"'{polygon.wkt}', {crs}"  
        updated_diff_dem_list = [] 

        if int(process_dict["year_end"]) - int(process_dict["year_start"]) < 1:
            end_year = int(process_dict["dem_year_end"])
            winter_combo = f"2010-{end_year}"
            time_span = end_year - 2010
        else:
            winter_combo = f"{process_dict['year_start']}-{process_dict['year_end']}"
            time_span = int(process_dict["year_end"]) - int(process_dict["year_start"])
            

        diff_dem_dict = {}
        for filename in os.listdir(data_path):
            if filename.endswith(".tif"):
                pattern = pattern = r'^([a-z_]+)'
                key_temp = re.match(pattern, filename).group(1)[:-1] # type: ignore
                diff_dem_dict[key_temp] = f"{data_path}/{filename}"
        updated_diff_dem_list.append(diff_dem_dict)


        stats, polygon = add_attributes(updated_diff_dem_list, polygon, tmp_txt_file_dict, process_dict, time_span=time_span, crs=crs) 
        if np.isnan(stats["dh_loss"]):
            print("No height calculation possible.")

        stats = postprocess_polygon(stats, polygon, updated_diff_dem_list, process_dict)
        stats = util.postprocess_stats(stats)
        polygons_stats[idx] = stats

        if isinstance(polygon, shapely.geometry.Polygon):
            polygon = shapely.geometry.MultiPolygon([polygon])
        gdf.at[idx, 'geometry'] = polygon


    df = pd.DataFrame.from_dict(polygons_stats, orient='index')
    gdf = gdf.merge(df, left_index=True, right_index=True)

    if "area_y" in gdf.columns:
        gdf.rename(columns={"area_y": "area"}, inplace=True)
        gdf.drop(columns=["area_x"], inplace=True)

    util.remove_temp_txt_files(tmp_txt_file_dict)
    return gdf


def add_attributes(
    diff_dem_objects: list,
    polygon: shapely.geometry.Polygon,
    temp_txt_file_dict: dict,
    process_dict: dict,
    time_span: int = 10,
    crs: int = 3413
) -> tuple[dict, shapely.geometry.Polygon]:
    """
    Add attributes to a single polygon based on the provided difference DEM objects.
    Parameters:
        diff_dem_objects (list): List of difference DEM objects.
        polygon (shapely.geometry.Polygon): The polygon to which attributes will be added.
        temp_txt_file_dict (dict): Dictionary containing paths to temporary text files.
        process_dict (dict): Dictionary containing processing parameters.
        time_span (int): Time span in years for the analysis. Default is 10.
        crs (int): Coordinate reference system EPSG code. Default is 3413.
    Returns:
        dict: A dictionary containing the calculated attributes.
        shapely.geometry.Polygon: The (possibly modified) input polygon.
    """
    stats = {}
    stats["time_span"] = time_span
    stats["area_exact"] = polygon.area

    if process_dict["data_type"] == "optical":
        stats, dh, sigma_dh, dh_upper, dh_lower = add_attributes_optical(diff_dem_objects, polygon, stats)
    else:
        stats, dh, sigma_dh, dh_upper, dh_lower, polygon = add_attributes_dem(diff_dem_objects, polygon, stats, rasterize=(True if process_dict["rasterize"] == "True" else False))

    if dh is None:
        stats["dv_loss"] = stats["dv_loss_upper"] = stats["dv_loss_lower"] = np.nan
        stats["dv_net"] = stats["dv_gain"] = stats["sigma_dv_loss"] = stats["sigma_dv_net"] = stats["sigma_dv_gain"] = np.nan
        stats["dv_0_1m"] = stats["dv_1_2m"] = stats["dv_2_3m"] = stats["dv_under3m"] = stats["dv_3_5m"] = stats["dv_5_10m"] = stats["dv_under10m"] = np.nan
    else:
        stats["dv_loss"], stats["sigma_dv_loss"], stats["dv_net"], stats["sigma_dv_net"], stats["dv_gain"], stats["sigma_dv_gain"] = volume_calc.calculate_volume(dh, sigma_dh)
        stats["dv_loss_upper"], stats["dv_loss_lower"] = volume_calc.calculate_volume_bounds(dh_upper, dh_lower)
        stats = volume_calc.calculate_volume_in_depth(dh, stats)
    return stats, polygon

def add_attributes_optical(
    diff_dem_objects: list,
    polygon: shapely.geometry.Polygon,
    stats: dict
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Add attributes to a polygon using optical RTS delineations based on difference DEMs.
    Parameters:
        diff_dem_objects (list): List of difference DEM objects.
        polygon (shapely.geometry.Polygon): The polygon to which attributes will be added.
        stats (dict): Dictionary to store calculated attributes.
    Returns:
        dict: Updated stats dictionary with calculated attributes.
        shapely.geometry.Polygon: The input polygon.
    """

    dh, sigma_dh, _ = util.read_and_write_dem_data(diff_dem_objects, polygon, all_touched=False)
    dh_big, sigma_dh_big, _ = util.read_and_write_dem_data(diff_dem_objects, polygon, all_touched=True)
    if dh is None:
        stats["dh"] = stats["sigma_dh"] = np.nan
        stats["area"] = stats["area_neg"] = stats["area_pos"] = np.nan
        stats["dh_loss"] = stats["sigma_dh_loss"] = stats["dh_net"] = stats["sigma_dh_net"] = stats["dh_gain"] = stats["sigma_dh_gain"] = np.nan
        stats["dh_loss_mean"] = stats["dh_loss_median"] = stats["dh_gain_mean"] = stats["dh_gain_median"] = stats["dh_net_mean"] = stats["dh_net_median"] = np.nan
        stats["sigma_dh_loss_mean"] = stats["sigma_dh_loss_median"] = stats["sigma_dh_gain_mean"] = stats["sigma_dh_gain_median"] = stats["sigma_dh_net_mean"] = stats["sigma_dh_net_median"] = np.nan
        stats["dh_loss_max"] = stats["sigma_dh_loss_max"] = stats["dh_loss_min"] = stats["sigma_dh_loss_min"] = np.nan
        return stats, None, None, None, None
    stats["area"], stats["area_neg"], stats["area_pos"] = calculate_area(dh)
    stats["area_big"], stats["area_neg_big"], stats["area_pos_big"]= calculate_area(dh_big)
    stats = height_calc.calculate_height(dh, sigma_dh, stats)
    dh_upper, dh_lower = height_calc.calculate_height_bounds(dh, sigma_dh, dh_big, sigma_dh_big)
    stats["dh_upper"], stats["dh_lower"] = dh_upper, dh_lower
    stats = height_calc.calculate_height_averages(dh, sigma_dh, stats)
    return stats, dh, sigma_dh, dh_upper, dh_lower


def add_attributes_dem(
    diff_dem_objects: list,
    polygon: shapely.geometry.Polygon,
    stats: dict,
    rasterize: bool = False
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, shapely.geometry.Polygon]:
    """
    Add attributes to a polygon using DEM-based delineations based on difference DEMs.

    Parameters:
        diff_dem_objects (list): List of difference DEM objects.
        polygon (shapely.geometry.Polygon): The polygon to which attributes will be added.
        stats (dict): Dictionary to store calculated attributes.
        rasterize (bool): Whether to rasterize the polygon before processing. Default is False.

    Returns:
        tuple: (
            dict: Updated stats dictionary with calculated attributes,
            numpy.ndarray: Array of height changes (dh),
            numpy.ndarray: Array of height change uncertainties (sigma_dh),
            numpy.ndarray: Array of upper bounds of height changes (dh_upper),
            numpy.ndarray: Array of lower bounds of height changes (dh_lower),
            shapely.geometry.Polygon: The (possibly modified) input polygon
        )
    """
    dh, sigma_dh, diff_dem = util.read_and_write_dem_data(diff_dem_objects, polygon, all_touched=True)

    if rasterize:
        polygon = util.rasterize_polygon(polygon, diff_dem)

    if dh is None or sigma_dh is None:
        stats["dh"] = stats["sigma_dh"] = np.nan
        stats["area"] = stats["area_neg"] = stats["area_pos"] = np.nan
        stats["dh_loss"] = stats["sigma_dh_loss"] = stats["dh_net"] = stats["sigma_dh_net"] = stats["dh_gain"] = stats["sigma_dh_gain"] = np.nan
        stats["dh_loss_mean"] = stats["dh_loss_median"] = stats["dh_gain_mean"] = stats["dh_gain_median"] = stats["dh_net_mean"] = stats["dh_net_median"] = np.nan
        stats["sigma_dh_loss_mean"] = stats["sigma_dh_loss_median"] = stats["sigma_dh_gain_mean"] = stats["sigma_dh_gain_median"] = stats["sigma_dh_net_mean"] = stats["sigma_dh_net_median"] = np.nan
        stats["dh_loss_max"] = stats["sigma_dh_loss_max"] = stats["dh_loss_min"] = stats["sigma_dh_loss_min"] = np.nan
        return stats, None, None, None, None, polygon

    stats["area"], stats["area_neg"], stats["area_pos"] = calculate_area(dh)
    stats = height_calc.calculate_height(dh, sigma_dh, stats)
    dh_upper, dh_lower = height_calc.calculate_height_bounds(dh, sigma_dh)
    stats["dh_upper"], stats["dh_lower"] = dh_upper, dh_lower
    stats = height_calc.calculate_height_averages(dh, sigma_dh, stats)
    return stats, dh, sigma_dh, dh_upper, dh_lower, polygon

def calculate_area(dh, pixel_area=100):
    """
    Calculate the total, negative, and positive area based on height change array.

    Parameters:
        dh (np.ndarray): Array of height changes.
        pixel_area (float, optional): Area of a single pixel. Default is 100.

    Returns:
        tuple: (total area, negative area, positive area)
    """
    area_total = len(dh[dh != 0.0]) * pixel_area
    area_negative = len(dh[(dh < 0.0)]) * pixel_area
    area_positive = len(dh[(dh > 0.0)]) * pixel_area
    return area_total, area_negative, area_positive

def postprocess_polygon(
    stats: dict,
    polygon: shapely.geometry.Polygon,
    diff_dem_list: list,
    process_dict: dict
) -> dict:
    """
    Quality check of polygon if the DEM falls into areas where the SAR quality is problematic (Coherence < 0.3 or SAR layover / shadow).
    Parameters:
        stats (dict): Dictionary of calculated statistics for the polygon.
        polygon (shapely.geometry.Polygon): The polygon geometry.
        diff_dem_list (list): List of difference DEM objects or masks.
        process_dict (dict): Dictionary containing processing parameters.
    Returns:
        dict: Updated statistics dictionary with quality and containment flags.
    """

    if any(util.is_contained_in_mask(polygon, diff_dem["sar_mask"]) for diff_dem in diff_dem_list):
        stats["quality"] = 0
    else:
        stats["quality"] = 1
    return stats










































def calculate_geology_attributes(process_dict, TEMP, location, gdf):
    crs = int(process_dict['crs'])
    yedoma_vector = gpd.read_file(process_dict['yedoma_file'], crs=f"EPSG:{crs}")
    lgm_vector = gpd.read_file(process_dict['lgm_file'], crs=f"EPSG:{crs}")
    ground_ice_vector = gpd.read_file(process_dict['ground_ice_file'], crs=f"EPSG:{crs}")   
    # CONTENT: l (low), m (medium), h (high), LANDFORM: f (lowlands), r (mountains)
    # low < 10%: fl, rl | medium (10-20%): fm, rh | high > 20%: fh
    polygons_stats = {}
    temp_txt_file_dict = util.create_temp_txt_files(temp_path, location, process_dict)
    
    total = gdf.shape[0]
    for i, (idx, polygon) in enumerate(gdf.geometry.items()):
        print(f"{i} of {total}")
        stats = {}
        geometry = f"'{polygon.wkt}', {crs}"

        stats = soc_calc.retrieve_alt(polygon, stats, temp_txt_file_dict, crs=crs)
        stats = soc_calc.retrieve_ground_ice_content(polygon, stats, temp_txt_file_dict, crs=crs)
        stats = soc_calc.retrieve_soc_content(polygon, stats, temp_txt_file_dict, crs=crs)
        stats = soc_calc.retrieve_yedoma(polygon, stats, yedoma_vector, crs=crs)
        stats = soc_calc.retrieve_lgm_vincinity(polygon, stats, lgm_vector, crs=crs)
        
        stats = util.postprocess_stats(stats)
        polygons_stats[idx] = stats


        if isinstance(polygon, shapely.geometry.Polygon):
            polygon = shapely.geometry.MultiPolygon([polygon])
        gdf.at[idx, 'geometry'] = polygon

    df = pd.DataFrame.from_dict(polygons_stats, orient='index')
    gdf = gdf.merge(df, left_index=True, right_index=True)
    until.remove_temp_txt_files(temp_txt_file_dict)
    return gdf











def add_years_to_gdf(librarian, gdf, years, crs, method_id):
    year_dict = {}
    for idx, polygon in gdf.geometry.items():
        geometry = f"'{polygon.wkt}', {crs}"
        tile_list = librarian.get_tiles_from_textgeom(geometry, crs=crs, geometry_crs=crs, intersect=True)
        years_store = []
        for tile in tile_list:
            tiled_dem_list = librarian.get_tandeminfo(tile, method_id, crs=crs)
            for tiled_dem in tiled_dem_list:
                if tiled_dem.acquisition_date.year in years:
                    if tiled_dem.acquisition_date.year not in years_store:
                        years_store.append(tiled_dem.acquisition_date.year)
        year_dict[idx] = years_store
    gdf["years"] = gdf.index.map(lambda idx: ','.join(map(str, year_dict[idx])))
    return gdf