import numpy as np
import pandas as pd
import astropy
import astromet
import math
import seaborn as sns
from numpy.linalg import LinAlgError
from astropy import units as u
#import dev.astromet.astromet as astromet
from tqdm.notebook import tqdm
import scanninglaw.times
from scanninglaw.source import Source
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

                                                            #################### CONSTANTS ####################

G = astropy.constants.G.to(u.AU**3/(u.M_sun*u.year**2)).value
AU = 1
M_sun = 1

dr3Period=34/12 # Calculate the period in years by dividing 34 by 12
mas=astromet.mas # conversion from degrees to milli-arcseconds

                                                            #################### PLOTS ####################

def plot_distributions(df, columns_to_plot, units):
    # Calculate the number of rows and columns needed
    num_cols = len(columns_to_plot)
    grid_size = math.ceil(math.sqrt(num_cols))

    # Create the subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Generate samples from the real distributions
    real_distributions = {
        'pllx': (1/2)*np.random.uniform(0, 1, 10000)**(-1/3),
        'q': 10**np.random.uniform(-3.2, 2.2, 10000),
        'l': np.random.uniform(0, 1, 10000),
        'a': 10*np.random.rand(10000)**2,
        'e': np.random.uniform(0, 1, 10000),
        'vTheta': np.arccos(np.random.uniform(-1, 1, 10000)),
        'vPhi': np.random.uniform(0, 2*np.pi, 10000),
        'vOmega': np.random.uniform(0, 2*np.pi, 10000)
    }

    # Iterate over the specified columns and plot each one
    for i, col in enumerate(columns_to_plot):
        sns.kdeplot(df[col], ax=axs[i], shade=True, label='KDE')
        sns.kdeplot(real_distributions[col], ax=axs[i], shade=False, color='r', label='"Real" Distribution')
        axs[i].set_title(f"{col}{units[col]}")
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Density')
        axs[i].legend()

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Adjust the layout of the subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

from matplotlib.ticker import MultipleLocator, FuncFormatter

def plot_eccentricity_distributions(df, eccentricity_ranges, color_codes):
    # Define the eccentricity ranges and corresponding dataframes
    dataframes = [
        df[(df['e'] >= e_min) & (df['e'] <= e_max)]
        for e_min, e_max in eccentricity_ranges
    ]

    # Create a figure with 20 subplots
    fig, axs = plt.subplots(4, 5, figsize=(16, 12), constrained_layout=True)

    # Function to format x-axis ticks in terms of 2π
    def format_func_x(value, tick_number):
        N = int(value / (2 * np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$2\pi$"
        else:
            return r"${0}2\pi$".format(N)

    # Function to format y-axis ticks in terms of π
    def format_func_y(value, tick_number):
        N = int(value / np.pi)
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi$"
        else:
            return r"${0}\pi$".format(N)

    # Loop over the eccentricity ranges and color codes to create the subplots
    for row, (color_code, color_label, color_func) in enumerate(color_codes):
        for col, (e_min, e_max) in enumerate(eccentricity_ranges):
            df_filtered = dataframes[col]
            scatter = axs[row, col].scatter(df_filtered['vPhi'], df_filtered['vTheta'], c=color_func(df_filtered), cmap='Spectral_r', s=10)
            axs[row, col].set_title(f'e = {e_min} and {e_max}')
            axs[row, col].set_xlabel('φ')
            axs[row, col].set_ylabel('θ')
            
            # Set the x-axis and y-axis ticks and labels
            axs[row, col].xaxis.set_major_locator(MultipleLocator(base=2 * np.pi))
            axs[row, col].xaxis.set_major_formatter(FuncFormatter(format_func_x))
            axs[row, col].yaxis.set_major_locator(MultipleLocator(base=np.pi))
            axs[row, col].yaxis.set_major_formatter(FuncFormatter(format_func_y))
        
        # Add a color bar for the entire row
        cbar = fig.colorbar(scatter, ax=axs[row, :], orientation='vertical')
        cbar.set_label(color_label)

    plt.show()

def plot_scatter_with_lines(df_filter, x_col, y_col, color_col_num, color_col_den, colorbar_label):
    # figure size
    plt.figure(figsize=(10, 8))

    # colormap normalization for the colorbar scale
    norm = LogNorm(vmin=0.1, vmax=10)

    # plot on the x axis x_col and on the y axis y_col and colormap the points by color_col
    plt.scatter(df_filter[x_col], df_filter[y_col], c=df_filter[color_col_num]/df_filter[color_col_den], cmap='Spectral_r', s=20, alpha=1, norm=norm)
    l_points = np.linspace(0, 1, 1000)
    q_points = np.linspace(0, 1, 1000)
    # draw a thick dotted line for l = q
    plt.plot(q_points, l_points, 'k--', linewidth=2)
    # draw a thick grey dotted line for l = q**3.5
    plt.plot(q_points, q_points**3.5, 'k:', linewidth=5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xscale('log')
    plt.xlim(10**-2, 10**1)
    plt.colorbar(label=colorbar_label)
    plt.show()

def plot_single_cluster_with_center(df1, df2, col_x, col_y, label1='SB', label2='DB', x_lim=None, y_lim=None, log_x=False, log_y=False, ax=None):
    """
    Plots clusters with their centers for two datasets on a single plot.

    Parameters:
    - df1: First DataFrame
    - df2: Second DataFrame
    - col_x: Column name for x-axis
    - col_y: Column name for y-axis
    - label1: Label for the first dataset
    - label2: Label for the second dataset
    - x_lim: Tuple specifying the x-axis limits (min, max)
    - y_lim: Tuple specifying the y-axis limits (min, max)
    - log_x: Boolean to set x-axis to log scale
    - log_y: Boolean to set y-axis to log scale
    - ax: Matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot for the first dataset with small faint points
    ax.scatter(df1[col_x], df1[col_y], color='blue', alpha=0.1, s=10, label=f'{label1} Points')

    # Scatter plot for the second dataset with small faint points
    ax.scatter(df2[col_x], df2[col_y], color='red', alpha=0.1, s=10, label=f'{label2} Points')

    # Kernel density estimate plot for the first dataset
    try:
        sns.kdeplot(data=df1, x=col_x, y=col_y, color='blue', fill=False, alpha=0.5, label=label1, levels=10, ax=ax)
    except LinAlgError:
        print(f"LinAlgError encountered for KDE plot of {label1} with columns {col_x} and {col_y}")

    # Kernel density estimate plot for the second dataset
    try:
        sns.kdeplot(data=df2, x=col_x, y=col_y, color='red', fill=False, alpha=0.3, label=label2, levels=10, ax=ax)
    except LinAlgError:
        print(f"LinAlgError encountered for KDE plot of {label2} with columns {col_x} and {col_y}")

    # Calculate the mean of col_x and col_y for the first dataset
    df1_center_x = df1[col_x].mean()
    df1_center_y = df1[col_y].mean()

    # Calculate the mean of col_x and col_y for the second dataset
    df2_center_x = df2[col_x].mean()
    df2_center_y = df2[col_y].mean()

    # Scatter plot for the first dataset center point
    ax.scatter(df1_center_x, df1_center_y, color='blue', alpha=1.0, s=200, edgecolor='black', linewidth=2, label=f'{label1} Center')

    # Scatter plot for the second dataset center point
    ax.scatter(df2_center_x, df2_center_y, color='red', alpha=1.0, s=200, edgecolor='black', linewidth=2, label=f'{label2} Center')

    # Set labels and title
    ax.set_xlabel(col_x, fontsize=14)
    ax.set_ylabel(col_y, fontsize=14)
    ax.set_title(f"{col_x} vs {col_y} Cluster Centers", fontsize=16)

    # Optimize x-axis limits if not provided
    if x_lim is None:
        min_x = min(df1[col_x].min(), df2[col_x].min())
        max_x = max(df1[col_x].max(), df2[col_x].max())
        ax.set_xlim(min_x, max_x)
    else:
        ax.set_xlim(x_lim)

    # Optimize y-axis limits if not provided
    if y_lim is None:
        min_y = min(df1[col_y].min(), df2[col_y].min())
        max_y = max(df1[col_y].max(), df2[col_y].max())
        ax.set_ylim(min_y, max_y)
    else:
        ax.set_ylim(y_lim)

    # Set log scale if specified
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # Add grid lines
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    # Add legend
    ax.legend(fontsize=12)

    plt.tight_layout()
    if ax is None:
        plt.show()

def plot_3d_clusters(sb, db, x_col, y_col, z_col, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
    """
    Plots a 3D scatter plot of specified columns for two datasets.
    
    Parameters:
    sb (DataFrame): DataFrame containing the first dataset.
    db (DataFrame): DataFrame containing the second dataset.
    x_col (str): Column name for the x-axis.
    y_col (str): Column name for the y-axis.
    z_col (str): Column name for the z-axis.
    """
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the first dataset with small faint points
    ax.scatter(sb[x_col], sb[y_col], sb[z_col], color='blue', alpha=0.1, s=10, label='SB Points')

    # Scatter plot for the second dataset with small faint points
    ax.scatter(db[x_col], db[y_col], db[z_col], color='red', alpha=0.1, s=10, label='DB Points')

    # Plot the center of the clusters
    sb_center = sb[[x_col, y_col, z_col]].mean()
    db_center = db[[x_col, y_col, z_col]].mean()

    ax.scatter(sb_center[x_col], sb_center[y_col], sb_center[z_col], color='blue', alpha=1.0, s=200, edgecolor='black', linewidth=2, label='SB Center')
    ax.scatter(db_center[x_col], db_center[y_col], db_center[z_col], color='red', alpha=1.0, s=200, edgecolor='black', linewidth=2, label='DB Center')

    # Plot the x,y,z coordinates of the cluster centers
    ax.text(sb_center[x_col], sb_center[y_col], sb_center[z_col], f'({sb_center[x_col]:.2f}, {sb_center[y_col]:.2f}, {sb_center[z_col]:.2f})', color='blue')
    ax.text(db_center[x_col], db_center[y_col], db_center[z_col], f'({db_center[x_col]:.2f}, {db_center[y_col]:.2f}, {db_center[z_col]:.2f})', color='red')

    # Set labels and title
    ax.set_xlabel(x_col, fontsize=14)
    ax.set_ylabel(y_col, fontsize=14)
    ax.set_zlabel(z_col, fontsize=14)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_title(f'{x_col} vs {y_col} vs {z_col} Cluster Centers', fontsize=16)

    # Add legend
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

                                                            #################### FUNCTIONS ####################

# write a function that calculates the values of zeta_0 for a given set of parameters e, vtheta, vphi
# in terms of e, k_ss = sin(vtheta)sin(vphi), k_sc = sin(vtheta)cos(vphi) and epsilon = sqrt(1-e^2)
def zeta_0(ecc, vTheta, vPhi):
    k_ss = np.sin(vTheta) * np.sin(vPhi)
    k_sc = np.sin(vTheta) * np.cos(vPhi)
    epsilon = np.sqrt(1 - (ecc ** 2))

    term_1 = (1 / ecc)
    term_2 = (k_ss ** 2) * epsilon * (1 - epsilon)
    term_3 = (k_sc ** 2) * (((ecc ** 4) + (ecc ** 2) * (3 * epsilon - 5) + (4 * (1 - epsilon))) / (((1 - epsilon) ** 2) * epsilon))

    sqrt_argument = term_2 - term_3
    # print sqrt_argument where sqrt_argument < 0 and return the indeces of the rows
    indeces = sqrt_argument[sqrt_argument < 0].index

    zeta = term_1 * np.sqrt(sqrt_argument)
    return zeta, indeces


def beta_0(ecc, vTheta, vPhi):

    term_1 = (np.sin(vTheta)**2)/2
    term_2 = (ecc**2)*(3+(np.sin(vTheta)**2)*((np.cos(vPhi)**2)-2))/4

    beta = np.sqrt(1 - term_1 - term_2)
    return beta

def sigma_spectroscopic_error(q, a, P, zeta_0):
    
    term_1 = q/(1+q)
    term_2 = (2*np.pi*a)/(P)
    term_3 = zeta_0

    sigma = term_1*term_2*term_3
    
    return sigma

def sigma_astrometric_error(q, l, pllx, a, A, beta_0):
    
    term_1 = np.abs((q - l) / ((1 + q) * (1 + l)))
    term_2 = pllx
    term_3 = a/A
    term_4 = beta_0

    sigma = term_1*term_2*term_3*term_4
    return sigma

def inferred_P(A, sigma_spectroscopic_error, sigma_astrometric_error, parallax, zeta_0, beta_0):
    term_1 = (2 * np.pi * A) / (sigma_spectroscopic_error)
    term_2 = sigma_astrometric_error / parallax
    term_3 = zeta_0 / beta_0
    return term_1 * term_2 * term_3

def solve_q(A, G, m_1, sigma_spectroscopic_error, sigma_astrometric_error, parallax, zeta_0, beta_0):

    ###### Calculate alpha ######
    alpha_term_1 = (((A) * (sigma_spectroscopic_error**2)) / (G * m_1))
    # print(alpha_term_1)
    alpha_term_2 = (sigma_astrometric_error / parallax)
    # print(alpha_term_2)
    alpha_term_3 = (1 / (beta_0 * (zeta_0**2)))
    # print(alpha_term_3)
    alpha =  alpha_term_1 * alpha_term_2 * alpha_term_3

    ###### Calculate mu and lam ######
    mu = - (((6 + alpha) / 3) * alpha)
    lam = ((27 + (18 * alpha) + (2 * alpha**2)) / 27) * alpha

    ###### Calculate q ######
    q = ((alpha / 3) + ((lam / 2)**(1/3)) * (
        (1 + np.sqrt(1 + (4 * mu**3) / (27 * lam**2)))**(1/3) + 
        (1 - np.sqrt(1 + (4 * mu**3) / (27 * lam**2)))**(1/3)
    ))

    return q

def simulate_gaia(nTest=10000, alError=1):
    dataNames = ('RA', 'Dec', 'pmRA', 'pmDec', 'pllx',
                 'M_tot', 'q', 'l', 'a', 'e', 'P', 'tPeri',
                 'vTheta', 'vPhi', 'vOmega',
                 'predict_dTheta', 'simple_dTheta',
                 'N_obs', 'sigma_al', 'sigma_ac',
                 'fit_ra', 'fit_dec', 'fit_pmrac', 'fit_pmdec', 'fit_pllx',
                 'sigma_rac', 'sigma_dec', 'sigma_pmrac', 'sigma_pmdec', 'sigma_pllx',
                 'N_vis', 'frac_good', 'AEN', 'UWE'
                )
    
    allData = astropy.table.Table(names=dataNames)

    for i in tqdm(range(nTest)):
        allData.add_row()
        thisRow = allData[i]
        
        params = astromet.params()
        params.ra = 360 * np.random.rand()
        params.dec = np.arcsin(-1 + 2 * np.random.rand()) * 180 / np.pi
        
        c = Source(params.ra, params.dec, unit='deg')
        sl = dr3_sl(c, return_times=True, return_angles=True)
        ts = 2010 + np.squeeze(np.hstack(sl['times'])) / 365.25
        sort = np.argsort(ts)
        ts = np.double(ts[sort])
        
        phis = np.squeeze(np.hstack(sl['angles']))[sort]
        
        params.parallax = 10 * np.power(np.random.rand(), -1/3)  # all within 100 pc
        params.pmrac = params.parallax * np.random.randn()
        params.pmdec = params.parallax * np.random.randn()
        params.period = 10 ** (-1.5 + 3 * np.random.rand())  # periods between 0.03 and 30 years
        params.l = np.random.rand()  # uniform light ratio
        params.q = 4 * np.random.rand() ** 2  # mass ratios between 0 and 4 (half less than 1)
        params.a = 10 * np.random.rand() ** 2
        params.e = np.random.rand()
        params.vtheta = np.arccos(-1 + 2 * np.random.rand())
        params.vphi = 2 * np.pi * np.random.rand()
        params.vomega = 2 * np.pi * np.random.rand()
        orbitalPhase = np.random.rand()  # fraction of an orbit completed at t=0
        params.tperi = params.period * orbitalPhase
        
        thisRow['RA'] = params.ra
        thisRow['Dec'] = params.dec
        thisRow['pmRA'] = params.pmrac
        thisRow['pmDec'] = params.pmdec
        thisRow['pllx'] = params.parallax
        thisRow['M_tot'] = 4 * (np.pi ** 2) * astromet.Galt / ((params.period ** 2) * (params.a ** 3))
        thisRow['q'] = params.q
        thisRow['l'] = params.l
        thisRow['a'] = params.a
        thisRow['e'] = params.e
        thisRow['P'] = params.period
        thisRow['tPeri'] = params.tperi
        thisRow['vTheta'] = params.vtheta
        thisRow['vPhi'] = params.vphi
        thisRow['vOmega'] = params.vomega
        thisRow['sigma_al'] = alError
        # thisRow['sigma_ac'] = acError

        trueRacs, trueDecs = astromet.track(ts, params)

        # added .astype(float) to avoid astromet error
        phis = phis.astype(float)
        
        t_obs, x_obs, phi_obs, rac_obs, dec_obs = astromet.mock_obs(ts, phis, trueRacs, trueDecs, err=alError)
        
        fitresults = astromet.fit(t_obs, x_obs, phi_obs, alError, params.ra, params.dec)
        results = astromet.gaia_results(fitresults)
        
        # print('ra, dec, pllx, pmrac, pmdec ',params.ra,params.dec,params.parallax,params.pmrac,params.pmdec)
        # print(results)
        
        # bug somewhere in these
        # thisRow['simple_dTheta'] = astromet.dtheta_simple(params)
        # thisRow['predict_dTheta'] = astromet.dtheta_full(params, np.min(ts), np.max(ts))  
        
        thisRow['fit_ra'] = results['ra']
        thisRow['fit_dec'] = results['dec']
        thisRow['fit_pmrac'] = results['pmra']
        thisRow['fit_pmdec'] = results['pmdec']
        thisRow['fit_pllx'] = results['parallax']

        thisRow['sigma_rac'] = results['ra_error']
        thisRow['sigma_dec'] = results['dec_error']
        thisRow['sigma_pmrac'] = results['pmra_error']
        thisRow['sigma_pmdec'] = results['pmdec_error']
        thisRow['sigma_pllx'] = results['parallax_error']

        # results['UWE'] --> results['uwe']
        thisRow['UWE'] = results['uwe']

        thisRow['N_obs'] = results['astrometric_n_obs_al']
        # thisRow['frac_good'] = results['astrometric_n_good_obs_al'] / results['astrometric_n_obs_al']
        thisRow['N_vis'] = results['visibility_periods_used']
        # thisRow['AEN'] = results['astrometric_excess_noise']

    return allData

                                                    #################### Missing Data Inference ####################

def fill_missing_values_XGBOOST(df, target_feature, model=None, imputation_strategy='mean'):
    # Separate rows with and without missing target_feature
    df_known = df.dropna(subset=[target_feature])
    df_missing = df[df[target_feature].isnull()]

    # Identify categorical columns
    categorical_cols = df_known.select_dtypes(include=['object']).columns

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_known = encoder.fit_transform(df_known[categorical_cols])
    encoded_missing = encoder.transform(df_missing[categorical_cols])

    # Create DataFrames from the encoded data
    encoded_known_df = pd.DataFrame(encoded_known, columns=encoder.get_feature_names_out(categorical_cols))
    encoded_missing_df = pd.DataFrame(encoded_missing, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate encoded columns
    X_known = df_known.drop(columns=[target_feature] + list(categorical_cols))
    X_known = pd.concat([X_known.reset_index(drop=True), encoded_known_df.reset_index(drop=True)], axis=1)

    X_missing = df_missing.drop(columns=[target_feature] + list(categorical_cols))
    X_missing = pd.concat([X_missing.reset_index(drop=True), encoded_missing_df.reset_index(drop=True)], axis=1)

    # Fill missing values in features
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_known = imputer.fit_transform(X_known)
    X_missing = imputer.transform(X_missing)

    y_known = df_known[target_feature]

    # Split the known data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

    # Train the model with progress bar
    if model is None:
        model = XGBRegressor(random_state=42)
    
    for _ in tqdm(range(1), desc="Training model"):
        model.fit(X_train, y_train)

    # Predict the missing target_feature values with progress bar
    y_pred = None
    for _ in tqdm(range(1), desc="Predicting test set"):
        y_pred = model.predict(X_test)
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R² Score: {r2}')
    print(f'Root Mean Squared Error: {rmse}')

    # Feature importance
    feature_importances = model.feature_importances_
    features = list(df_known.drop(columns=[target_feature] + list(categorical_cols)).columns) + list(encoder.get_feature_names_out(categorical_cols))
    
    # Ensure the lengths match
    min_length = min(len(features), len(feature_importances))
    features = features[:min_length]
    feature_importances = feature_importances[:min_length]
    
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(feature_importance_df)

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X_known, y_known, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training error')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation error')
    plt.xlabel('Training size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.show()

    # Predict the missing target_feature values with progress bar
    predicted_values = None
    for _ in tqdm(range(1), desc="Predicting missing values"):
        predicted_values = model.predict(X_missing)

    # Fill the missing values in the original dataframe
    df.loc[df[target_feature].isnull(), target_feature] = predicted_values

    # Return the full dataframe with the updated column
    return df

def fill_missing_values_RF(df, target_feature, model=None, imputation_strategy='mean'):
    # Separate rows with and without missing target_feature
    df_known = df.dropna(subset=[target_feature])
    df_missing = df[df[target_feature].isnull()]

    # Identify categorical columns
    categorical_cols = df_known.select_dtypes(include=['object']).columns

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_known = encoder.fit_transform(df_known[categorical_cols])
    encoded_missing = encoder.transform(df_missing[categorical_cols])

    # Create DataFrames from the encoded data
    encoded_known_df = pd.DataFrame(encoded_known, columns=encoder.get_feature_names_out(categorical_cols))
    encoded_missing_df = pd.DataFrame(encoded_missing, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and concatenate encoded columns
    X_known = df_known.drop(columns=[target_feature] + list(categorical_cols))
    X_known = pd.concat([X_known.reset_index(drop=True), encoded_known_df.reset_index(drop=True)], axis=1)

    X_missing = df_missing.drop(columns=[target_feature] + list(categorical_cols))
    X_missing = pd.concat([X_missing.reset_index(drop=True), encoded_missing_df.reset_index(drop=True)], axis=1)

    # Fill missing values in features
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_known = imputer.fit_transform(X_known)
    X_missing = imputer.transform(X_missing)

    y_known = df_known[target_feature]

    # Split the known data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42)

    # Print the shapes of the splits
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Train the model with progress bar
    if model is None:
        model = RandomForestRegressor(random_state=42)
    
    for _ in tqdm(range(1), desc="Training model"):
        model.fit(X_train, y_train)

    # Predict the missing target_feature values with progress bar
    y_pred = None
    for _ in tqdm(range(1), desc="Predicting test set"):
        y_pred = model.predict(X_test)
    
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(np.mean(np.abs(y_test - y_pred)))
    
    # Predict the missing target_feature values with progress bar
    predicted_values = None
    for _ in tqdm(range(1), desc="Predicting missing values"):
        predicted_values = model.predict(X_missing)

    # Fill the missing values in the original dataframe
    df.loc[df[target_feature].isnull(), target_feature] = predicted_values

    # Return the full dataframe with the updated column
    return df

                                                    #################### Triage ####################

def AMRF(a, M_1, parallax, P):
    term_1 = a/parallax
    term_2 = (M_1/M_sun)**(-(1/3))
    term_3 = (P)**(-(2/3))

    return term_1 * term_2 * term_3

def AMRF_q(q, S):

    term_1 = q/((1+q)**(2/3))
    term_2 = (1-((S*(1+q))/(q*(1+S))))
    return term_1 * term_2