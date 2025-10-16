import numpy as np

def calculate_volume(
    dh: np.ndarray,
    sigma_dh: np.ndarray,
    pixel_area: float = 100
) -> tuple[float, float, float, float, float, float]:
    """
    Calculates volume changes and their uncertainties for a given array of height changes.

    Parameters:
        dh (np.ndarray): Array of height changes.
        sigma_dh (np.ndarray): Array of height change uncertainties.
        pixel_area (float): Area of each pixel in square meters. Default is 100.

    Returns:
        tuple: (dv_loss, sigma_dv_loss, dv_net, sigma_dv_net, dv_gain, sigma_dv_gain)
            dv_loss: Total volume loss (negative height changes).
            sigma_dv_loss: Uncertainty of volume loss.
            dv_net: Net volume change.
            sigma_dv_net: Uncertainty of net volume change.
            dv_gain: Total volume gain (positive height changes).
            sigma_dv_gain: Uncertainty of volume gain.
    """
    dv_net = np.sum(dh * pixel_area)
    dv_gain = np.sum(dh[dh > 0.0] * pixel_area)
    dv_loss = np.sum(dh[dh < 0.0] * pixel_area)
    if sigma_dh is not None:
        sigma_dv_net = np.sqrt(np.nansum((sigma_dh * pixel_area) ** 2))
        sigma_dv_loss = np.sqrt(np.nansum((sigma_dh[dh < 0.0] * pixel_area) ** 2))
        sigma_dv_gain = np.sqrt(np.nansum((sigma_dh[dh > 0.0] * pixel_area) ** 2))
    else:
        sigma_dv_net = sigma_dv_loss = sigma_dv_gain = np.nan
    return dv_loss, sigma_dv_loss, dv_net, sigma_dv_net, dv_gain, sigma_dv_gain
    

def calculate_volume_bounds(
    dh_upper: np.ndarray,
    dh_lower: np.ndarray,
    pixel_area: float = 100
) -> tuple[float, float]:
    """
    Calculates the upper and lower bounds of volume loss for negative height changes.

    Parameters:
        dh_upper (np.ndarray): Array of upper bound height changes.
        dh_lower (np.ndarray): Array of lower bound height changes.
        pixel_area (float): Area of each pixel in square meters. Default is 100.

    Returns:
        tuple: (dv_loss_upper, dv_loss_lower) representing the upper and lower bounds of volume loss.
    """
    dv_loss_upper = np.sum(dh_upper[dh_upper < 0.0] * pixel_area)
    dv_loss_lower = np.sum(dh_lower[dh_lower < 0.0] * pixel_area)
    return dv_loss_upper, dv_loss_lower


def calculate_volume_in_depth(dh: np.ndarray, stats: dict, pixel_area: float = 100) -> dict:
    """
    Calculates volume changes in specific depth intervals and updates the stats dictionary.

    Parameters:
        dh (np.ndarray): Array of height changes.
        stats (dict): Dictionary to store calculated volume attributes.
        pixel_area (float): Area of each pixel in square meters. Default is 100.

    Returns:
        dict: Updated stats dictionary with calculated volume attributes.
    """
    depths = [(0.0, -1.0), (-1.0, -2.0), (-2.0, -3.0), (-3.0, None), (-3.0, -5.0), (-5.0, -10.0), (-10.0, None)]
    for (upper, lower) in depths:
        complete_fit = np.full_like(dh, -1)
        if lower is not None:
            dh_depth = dh[(dh > lower) & (dh < upper)]
            depth_layer = np.full_like(dh_depth, 1) * upper
            partial_fit = dh_depth - depth_layer
            tot_dh_depth = np.nansum(partial_fit) + np.nansum(complete_fit[dh <= lower])
            stats[f"dv_{abs(upper):.0f}_{abs(lower):.0f}m"] = tot_dh_depth * pixel_area

        else:
            dh_depth = dh[dh < upper]
            depth_layer = np.full_like(dh_depth, 1) * upper
            partial_fit = dh_depth - depth_layer
            tot_dh_depth = np.nansum(partial_fit)
            stats[f"dv_under{abs(upper):.0f}m"] = tot_dh_depth * pixel_area
    return stats