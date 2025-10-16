import math
import warnings
import numpy as np
import scipy.stats as stats
from scipy.odr import Model, RealData, ODR
from scipy.special import gamma
from scipy.optimize import curve_fit
import seaborn as sns

def perform_fit(x,y, model='ODR'):    
    """
    Perform a linear regression on the provided data with a selected model.
    
    Parameters:
    x : array-like
        Independent variable data.
    y : array-like
        Dependent variable data.
    model : str
        The model to use for fitting. Currently only 'ODR' (Orthogonal Distance Regression) is supported. 
        'OLS' (Ordinary Least Squares) will be added in the future.
        
    Returns:
    p : list
        Fitted parameters [intercept, slope].
    R : float
        Coefficient of determination (R-squared).
    RMSE : float
        Root Mean Square Error of the fit.
    p_val : float
        p-value for the slope parameter.
    t : float
        t-statistic for the slope parameter.
    s_err : float
        Standard error of the regression.
    n : int
        Number of observations.
    stds : list
        Standard deviations of the fitted parameters.
    t2 : float
        t-value for confidence intervals and prediction intervals.
    adj_R : float
        Adjusted R-squared value.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("At least two data points are required for fitting")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values. Please clean the data before fitting.")
    if model == 'OLS':
        p, stds, t_stat, p_val = OLS_fit(x, y)
    elif model == 'ODR':
        p, stds, t_stat, p_val = ODR_fit(x, y)
    else:
        raise ValueError("Unsupported model type. Use 'ODR' for Orthogonal Distance Regression or 'OLS' for Ordinary Least Squares.")
    
    y_model = powerlaw(x, *p)
    n = x.size # number of observations
    m = len(p) # number of parameters
    dof = n - m # degrees of freedom
    t = stats.t.ppf(0.95, n - m) # used for CI and PI bands
    t2 = stats.t.ppf(0.68, n - m) # used for CI and PI bands

    resid = y - y_model                           
    chi2 = np.sum((resid / y_model)**2) # chi-squared; estimates error in data
    s_err = np.sqrt(np.sum(resid**2) / dof) # standard deviation of the error
    RMSE =  np.sqrt(np.mean((resid)**2))
    R = coefficient_of_determination(y, y_model)
    return p, R, RMSE, p_val, t, s_err, n, stds, t2

def OLS_fit(x, y):
    """
    Perform Ordinary Least Squares (OLS) regression on the provided data.

    Parameters
    ----------
    x : array-like
        Independent variable data.
    y : array-like
        Dependent variable data.

    Returns
    -------
    p : list
        Fitted parameters [slope, intercept].
    stds : list
        Standard deviations of the fitted parameters.
    t_stat : float
        t-statistic for the slope parameter.
    p_val : float
        p-value for the slope parameter.
    """
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    p = result[0]
    residuals = result[1]
    if len(residuals) == 0:
        residuals = np.array([0.])
    stds = np.sqrt(np.diag(np.linalg.inv(np.dot(A.T, A))) * np.sum(residuals) / (len(y) - 2))
    t_stat = p[0] / stds[0]  # t statistic for the slope parameter
    p_val = stats.t.sf(np.abs(t_stat), len(y) - 2) * 2
    # Return as [slope, intercept] for consistency
    return [p[0], p[1]], stds, t_stat, p_val


def ODR_fit(x: np.ndarray, y: np.ndarray):
    """
    Perform Orthogonal Distance Regression (ODR) on the provided data.

    Parameters
    ----------
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.

    Returns
    -------
    p : list
        Fitted parameters [intercept, slope].
    stds : list
        Standard deviations of the fitted parameters.
    t_stat : float
        t-statistic for the slope parameter.
    p_val : float
        p-value for the slope parameter.
    """
    linear_model = Model(linear_func)
    data = RealData(x, y)
    odr = ODR(data, linear_model, beta0=[0., 1.])
    out = odr.run()
    p = [out.beta[0], out.beta[1]]
    stds = [out.sd_beta[0], out.sd_beta[1]] 
    t_stat = out.beta[0] / out.sd_beta[0]  # t statistic for the slope parameter
    p_val = stats.t.sf(np.abs(t_stat), out.iwork[10]) * 2
    return p, stds, t_stat, p_val


def powerlaw(x: np.ndarray, alpha: float, c: float) -> np.ndarray:
    """
    Power law function in log-log space.

    Parameters:
    x : array-like or float
        Independent variable (log10 of area, for example).
    alpha : float
        Slope of the power law in log-log space.
    c : float
        Intercept of the power law in log-log space.

    Returns:
    array-like or float
        Predicted values in log-log space (log10(y)).
    """
    return alpha * x + c


def coefficient_of_determination(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the coefficient of determination (R^2) for the fit.

    Parameters
    ----------
    y_true : np.ndarray
        Actual observed values.
    y_pred : np.ndarray
        Predicted values from the model.

    Returns
    -------
    float
        R-squared value indicating the goodness of fit.
    """
    y_mean = np.mean(y_true)
    squared_error_regr = squared_error(y_true, y_pred)
    squared_error_y_mean = squared_error(y_true, np.full_like(y_true, y_mean))
    return 1 - (squared_error_regr / squared_error_y_mean)

def squared_error(ys_orig: np.ndarray, ys_line: np.ndarray) -> float:
    """
    Calculate the sum of squared errors between original and predicted values.

    Parameters
    ----------
    ys_orig : np.ndarray
        Original observed values.
    ys_line : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Sum of squared errors.
    """
    return np.sum((ys_line - ys_orig) ** 2)


def linear_func(beta, x):
    """
    Linear function for ODR fitting.

    Parameters
    ----------
    beta : array-like
        Model parameters [intercept, slope].
    x : array-like or float
        Independent variable.

    Returns
    -------
    array-like or float
        Predicted values.
    """
    intercept, slope = beta
    return intercept + slope * x


    
