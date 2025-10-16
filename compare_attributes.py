import os
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from helpers import config_parser as config
import helpers.compare_mass_wasting as mass_wasting

ROOT = "./"  # Change this to your project root
TEMP = "./temp"    # Change this to your temp directory

def main():
    """Main function to compare attributes between reference (DEM) and comparison 
    (optical) RTS polygon datasets for segmentation and mass wasting statistics (validation)."""
    
    
    ini_location = 'ini/compare.ini'
    process_dict = config.get_config_dict(filename=ini_location,
                                            section='general')
    os.makedirs(f"{ROOT}/comparison", exist_ok=True)

    # For QTP, this is fixed to 2011 (based on TanDEM-X DEM availability)
    ref_year_start = 2011 
    ref_year_end = 2020
    time_span = ref_year_end - ref_year_start

    for location in process_dict['sites'].split():
        print(f"Processing QTP: {location}...")

        # Load the reference and comparison GeoDataFrames with attributes added and IDs (manually matched in validation)
        ref_gdf = gpd.read_file(f"{ROOT}/attributes/soc_polygons_dem_{location}_{ref_year_start}_{ref_year_end}.geojson", crs=f"EPSG:{process_dict['crs']}")
        ref_gdf = filter_gdf(ref_gdf, ref_year_start, ref_year_end, "ref_id") 
        comp_gdf = gpd.read_file(f"{ROOT}/attributes/soc_polygons_optical_{location}_{process_dict['year_start']}_{process_dict['year_end']}.geojson", crs=f"EPSG:{process_dict['crs']}")
        comp_gdf = filter_gdf(comp_gdf, process_dict['year_start'], process_dict['year_end'], "comp_id") 
        
        df = compute_segmentation_stats(ref_gdf, comp_gdf)
        df.to_csv(f"{ROOT}/comparison/stats_segmentation_{location}_{process_dict['year_start']}_{process_dict['year_end']}.csv", index=False)
        
        df = compare_mass_wasting_performance(process_dict, ref_gdf, comp_gdf, time_span)
        if df is not None:
            df.to_csv(f"{ROOT}/comparison/stats_mass_wasting_{location}_{process_dict['year_start']}_{process_dict['year_end']}.csv", index=False)
    return


def filter_gdf(gdf, year_start, year_end, string_gdf):
    gdf[string_gdf] = range(1, len(gdf) + 1)
    gdf = gdf[(gdf.year_start.astype(str) == str(year_start)) & (gdf.year_end.astype(str) == str(year_end))]
    return gdf


def compute_segmentation_stats(ref_gdf: gpd.GeoDataFrame, comp_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute statistics comparing reference and comparison GeoDataFrames.

    Parameters:
        ref_gdf (GeoDataFrame): Reference polygons with 'ref_id' column (DEM).
        comp_gdf (GeoDataFrame): Comparison polygons with 'comp_id' column (optical).

    Returns:
        DataFrame: Metrics including true positives, false negatives, false positives, IoU, precision, recall, and F1 scores.
    """
    intersection = gpd.overlay(comp_gdf, ref_gdf, how='intersection', keep_geom_type=False)
    metrics = {
        "num_ref": ref_gdf['id'].nunique(),
        "num_comp": comp_gdf['id'].nunique(),
        "tp": intersection['id_1'].nunique(),
        "fn": ref_gdf['id'].nunique() - intersection['id_1'].nunique(),
        "fp": comp_gdf['id'].nunique() - intersection['id_1'].nunique(),
    }
    metrics["pixel_iou"] = intersection.geometry.area.sum() / (ref_gdf.geometry.area.sum() + comp_gdf.geometry.area.sum() - intersection.geometry.area.sum())
    metrics["pixel_p"] = intersection.geometry.area.sum() / comp_gdf.geometry.area.sum()
    metrics["pixel_r"] = intersection.geometry.area.sum() / ref_gdf.geometry.area.sum()
    metrics["pixel_f1"] = 2 * (metrics["pixel_p"] * metrics["pixel_r"]) / (metrics["pixel_p"] + metrics["pixel_r"])

    metrics["detection_iou"] = metrics["tp"] / metrics["tp"] + metrics["fp"] + metrics["fn"]   
    metrics["detection_iou_mean"] = np.mean([
        row1.geometry.intersection(row2.geometry).area / row1.geometry.union(row2.geometry).area
        for _, row1 in ref_gdf.iterrows()
        for _, row2 in comp_gdf.iterrows()
        if row1.geometry.intersects(row2.geometry)
    ])
    metrics["detection_p"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])
    metrics["detection_r"] = metrics["tp"] / (metrics["tp"] + metrics["fn"])
    metrics["detection_f1"] = 2 * (metrics["detection_p"] * metrics["detection_r"]) / (metrics["detection_p"] + metrics["detection_r"])
    return pd.DataFrame([metrics])


def compare_mass_wasting_performance(
    process_dict: dict,
    gdf_ref: gpd.GeoDataFrame,
    gdf_comp: gpd.GeoDataFrame,
    time_span: int
) -> pd.DataFrame:
    """
    Compare mass wasting performance between reference and comparison GeoDataFrames.

    Parameters:
        process_dict (dict): Dictionary containing process information (e.g., 'gi', 'soc').
        gdf_ref (GeoDataFrame): Reference polygons with relevant attributes (DEM).
        gdf_comp (GeoDataFrame): Comparison polygons with relevant attributes (optical).
        time_span (int, optional): Time span for the comparison.

    Returns:
        DataFrame: Aggregated statistics and ratios for mass wasting analysis.
    """

    dem_dict = {}
    for key, gdf in {"ref": gdf_ref, "comp": gdf_comp}.items():
        key_dict = {}
        key_dict["type"] = key
        key_dict["timespan"] = time_span
        key_dict["num_polygons"] = len(gdf)
        
        # Area, Volume, and Height Change
        sum_list = ["area_exact", "area", "area_neg", "area_pos", "dh_loss", "dh_gain", "dh_net", "dv_loss", "dv_loss_upper", "dv_loss_lower", "dv_gain", "dv_gain_upper", "dv_gain_lower", "dv_net", "dv_0_1m", "dv_1_2m", "dv_2_3m", "dv_under3m", "dv_3_5m", "dv_5_10m", "dv_under10m"]
        error_list = ["sigma_dh_loss", "sigma_dh_gain", "sigma_dh_net", "sigma_dv_loss", "sigma_dv_gain", "sigma_dv_net"]
        for attribute in sum_list:
            if attribute in gdf.columns:
                key_dict[attribute] = np.nansum(gdf[attribute])
        for attribute in error_list:
            if attribute in gdf.columns:
                key_dict[attribute] = np.sqrt(np.nansum(gdf[attribute]**2))

        # SOC, Ground Ice & ALT
        if "alt" in gdf.columns:
            key_dict["alt"] = gdf["alt"][0]
        if "model" in process_dict["gi"].split():
            key_dict["gi_perc"] = np.nanmean(gdf["gi_perc"])
        
        if not process_dict["soc"] == "":
            for soc_data in process_dict["soc"].split():
                for upper_depth in ["0_1m", "1_2m", "2_3m", "0_3m"]:
                    key_dict[f"{soc_data}_soc{upper_depth}"] = np.nansum(gdf[f"{soc_data}_soc{upper_depth}"])
                    key_dict[f"{soc_data}_soc{upper_depth}_upper"] = np.nansum(gdf[f"{soc_data}_soc{upper_depth}_upper"])
                    key_dict[f"{soc_data}_soc{upper_depth}_lower"] = np.nansum(gdf[f"{soc_data}_soc{upper_depth}_lower"][gdf[f"{soc_data}_soc{upper_depth}_lower"] > 0.0])

                for lower_depth in ["3_5m", "5_10m", "_under3m"]:
                    for deep_soc in ["lin", "exp"]:
                        key_dict[f"{soc_data}_{deep_soc}_soc{lower_depth}"] = np.nansum(gdf[f"{soc_data}_{deep_soc}_soc{lower_depth}"])
                        key_dict[f"{soc_data}_{deep_soc}_soc{lower_depth}_upper"] = np.nansum(gdf[f"{soc_data}_{deep_soc}_soc{lower_depth}_upper"])
                        key_dict[f"{soc_data}_{deep_soc}_soc{lower_depth}_lower"] = np.nansum(gdf[f"{soc_data}_{deep_soc}_soc{lower_depth}_lower"][gdf[f"{soc_data}_{deep_soc}_soc{lower_depth}_lower"] > 0.0])

                for gi_type in process_dict["gi"].split():
                    for depth in ["2_3m", "3_5m", "5_10m"]:
                        key_dict[f"{gi_type}_gi{depth}"] = np.nansum(gdf[f"{gi_type}_gi{depth}"])
                        key_dict[f"{gi_type}_gi{depth}_upper"] = np.nansum(gdf[f"{gi_type}_gi{depth}_upper"])
                        key_dict[f"{gi_type}_gi{depth}_lower"] = np.nansum(gdf[f"{gi_type}_gi{depth}_lower"][gdf[f"{gi_type}_gi{depth}_lower"] > 0])
                    key_dict[f"{gi_type}_gi_column"] = key_dict[f"{gi_type}_gi2_3m"] + key_dict[f"{gi_type}_gi3_5m"] + key_dict[f"{gi_type}_gi5_10m"]
                    key_dict[f"{gi_type}_gi_column_upper"] = key_dict[f"{gi_type}_gi2_3m_upper"] + key_dict[f"{gi_type}_gi3_5m_upper"] + key_dict[f"{gi_type}_gi5_10m_upper"]
                    key_dict[f"{gi_type}_gi_column_lower"] = key_dict[f"{gi_type}_gi2_3m_lower"] + key_dict[f"{gi_type}_gi3_5m_lower"] + key_dict[f"{gi_type}_gi5_10m_lower"]
                    for deep_soc in ["lin", "exp"]:
                        key_dict[f"gi_{gi_type}_{soc_data}_{deep_soc}_soc_column"] = key_dict[f"{gi_type}_gi_column"] * key_dict[f"{soc_data}_{deep_soc}_soc_under3m"]
                        key_dict[f"gi_{gi_type}_{soc_data}_{deep_soc}_soc_column_upper"] = key_dict[f"{gi_type}_gi_column_upper"] * key_dict[f"{soc_data}_{deep_soc}_soc_under3m_upper"]
                        key_dict[f"gi_{gi_type}_{soc_data}_{deep_soc}_soc_column_lower"] = key_dict[f"{gi_type}_gi_column_lower"] * key_dict[f"{soc_data}_{deep_soc}_soc_under3m_lower"]

        dem_dict[key] = key_dict

    ratio_dict = {}
    ratio_dict["type"] = "ratio"
    for key in dem_dict["comp"]:
        if key not in ["type", "timespan", "num_polygons", "gi_perc", "alt"]:
            ratio_dict[key] = (dem_dict["comp"][key] - dem_dict["ref"][key]) / dem_dict["ref"][key]
    
    dem_dict["ratio"] = ratio_dict
    return pd.DataFrame.from_dict(dem_dict, orient='index')


  
if __name__ == "__main__":
    main()  