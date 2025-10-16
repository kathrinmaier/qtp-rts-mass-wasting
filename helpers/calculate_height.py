import numpy as np


def calculate_height(dh: np.ndarray, sigma_dh: np.ndarray, stats: dict) -> dict:
    """
    Calculates height change statistics and updates the stats dictionary.

    Parameters:
        dh (np.ndarray): Array of height changes.
        sigma_dh (np.ndarray): Array of height change uncertainties.
        stats (dict): Dictionary to store calculated height attributes.

    Returns:
        dict: Updated stats dictionary with calculated height attributes.
    """
    stats["dh"] = dh
    stats["sigma_dh"] = sigma_dh
    stats["dh_net"] = np.sum(dh)
    stats["dh_gain"] = np.sum(dh[dh > 0.0])
    stats["dh_loss"] = np.sum(dh[dh < 0.0])
    if sigma_dh is not None:
        stats["sigma_dh_loss"] = np.sqrt(np.sum(sigma_dh[dh < 0.0] ** 2))
        stats["sigma_dh_net"] = np.sqrt(np.sum(sigma_dh ** 2))
        stats["sigma_dh_gain"] = np.sqrt(np.sum(sigma_dh[dh > 0.0] ** 2))
    return stats

def calculate_height_bounds(dh, sigma_dh, dh_big=None, sigma_dh_big=None):
    """
    Calculates upper and lower bounds for height changes (dh) using uncertainties (sigma_dh).
    If alternative height and uncertainty arrays (dh_big, sigma_dh_big) are provided, use them for upper bound.

    Parameters:
        dh (np.ndarray): Array of height changes.
        sigma_dh (np.ndarray): Array of height change uncertainties.
        dh_big (np.ndarray, optional): Alternative array of height changes for upper bound.
        sigma_dh_big (np.ndarray, optional): Alternative array of uncertainties for upper bound.

    Returns:
        tuple: (dh_upper, dh_lower) arrays with upper and lower bounds for height changes.
    """
    if dh_big is not None and sigma_dh_big is not None:
        dh_upper = np.array(dh_big) - np.array(sigma_dh_big)
    else:
        dh_upper = np.array(dh) - np.array(sigma_dh)
    dh_lower = np.array(dh) + np.array(sigma_dh)
    return dh_upper, dh_lower


def calculate_height_averages(dh, sigma_dh, stats):
    """
    Calculate average height changes and update the stats dictionary.
    Parameters:
        dh (numpy.ndarray): Array of height changes.
        sigma_dh (numpy.ndarray): Array of height change uncertainties.
        stats (dict): Dictionary to store calculated height attributes.
    Returns:
        dict: Updated stats dictionary with calculated height attributes.
    """
    # Loss statistics
    dh_loss = dh[dh < 0.0]
    if dh_loss.size == 0:
        stats.update({
            "dh_loss_max": np.nan,
            "dh_loss_min": np.nan,
            "dh_loss_mean": np.nan,
            "dh_loss_median": np.nan
        })
        if sigma_dh is not None:
            stats.update({
                "sigma_dh_loss_max": np.nan,
                "sigma_dh_loss_min": np.nan,
                "sigma_dh_loss_mean": np.nan,
                "sigma_dh_loss_median": np.nan
            })
    else:
        stats.update({
            "dh_loss_max": np.nanmax(dh_loss),
            "dh_loss_min": np.nanmin(dh_loss),
            "dh_loss_mean": np.nanmean(dh_loss),
            "dh_loss_median": np.nanmedian(dh_loss)
        })
        if sigma_dh is not None:
            sigma_loss = sigma_dh[dh < 0.0]
            stats.update({
                "sigma_dh_loss_max": np.nanmax(sigma_loss),
                "sigma_dh_loss_min": np.nanmin(sigma_loss),
                "sigma_dh_loss_mean": np.nanmean(sigma_loss),
                "sigma_dh_loss_median": np.nanmedian(sigma_loss)
            })

    # Gain statistics
    dh_gain = dh[dh > 0.0]
    if dh_gain.size == 0:
        stats["dh_gain_mean"] = np.nan
        stats["dh_gain_median"] = np.nan
        if sigma_dh is not None:
            stats["sigma_dh_gain_mean"] = np.nan
            stats["sigma_dh_gain_median"] = np.nan
    else:
        stats["dh_gain_mean"] = np.nanmean(dh_gain)
        stats["dh_gain_median"] = np.nanmedian(dh_gain)
        if sigma_dh is not None:
            sigma_gain = sigma_dh[dh > 0.0]
            stats["sigma_dh_gain_mean"] = np.nanmean(sigma_gain)
            stats["sigma_dh_gain_median"] = np.nanmedian(sigma_gain)

    # Net statistics
    if dh.size == 0:
        stats["dh_net_mean"] = np.nan
        stats["dh_net_median"] = np.nan
        if sigma_dh is not None:
            stats["sigma_dh_net_mean"] = np.nan
            stats["sigma_dh_net_median"] = np.nan
    else:
        stats["dh_net_mean"] = np.nanmean(dh)
        stats["dh_net_median"] = np.nanmedian(dh)
        if sigma_dh is not None:
            stats["sigma_dh_net_mean"] = np.nanmean(sigma_dh)
            stats["sigma_dh_net_median"] = np.nanmedian(sigma_dh)
    return stats