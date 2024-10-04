import numpy as np
import pandas as pd
import astropy
import astropy.table
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
from matplotlib.colors import Normalize
from sevnpy.sevn import SEVNmanager, sevnwrap
from tqdm import tqdm
from IC4popsyn.ic4popsyn import populations as pop
from scipy.spatial import cKDTree
from IPython.display import Image, display


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

            # Set the colormap and normalization depending on the row
            if row == 0 or row == 1:
                cmap = 'twilight_shifted'
                norm = Normalize(0, 1)
            elif row == 2:
                cmap = 'Spectral_r'
                norm = None
            else:
                cmap = 'Spectral_r'
                norm = None

            # Scatter plot with the correct color mapping and normalization
            scatter = axs[row, col].scatter(df_filtered['vPhi'], df_filtered['vTheta'], c=color_func(df_filtered), cmap=cmap, s=10, norm=norm)
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
    plt.figure(figsize=(6, 5))

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

def HR_plot(cross_df_filtered):
    # Normalize ruwe values from 0.1 to 10
    norm = Normalize(vmin=0.1, vmax=10)

    # Plot the Hertzsprung-Russell diagram color coded by ruwe
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(cross_df_filtered['bp_rp_corrected'], cross_df_filtered['M_G'], s=0.1, c=cross_df_filtered['ruwe'], cmap='Spectral_r', norm=norm, alpha=1)

    # Invert y-axis to have bright stars at the top
    plt.gca().invert_yaxis()

    # Set plot labels and title
    plt.xlabel('(BP - RP) Corrected Color')
    plt.ylabel('M_G (Absolute G Magnitude)')
    plt.title('Hertzsprung-Russell Diagram')

    # Add color bar
    plt.colorbar(sc, label='ruwe')

    # Show the plot
    plt.show()

def plot_scatter_with_colorbar(df, x_col, y_col, color_col, remnant_type, figsize=(10, 6), cmap='Spectral_r', s=50, legend_loc='lower right'):
    """
    Plot a scatter plot with a colorbar and legend.

    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for the x-axis
    - y_col: Column name for the y-axis
    - color_col: Column name for the color coding
    - remnant_type: Dictionary mapping remnant type values to labels
    - figsize: Tuple specifying the figure size (default: (10, 6))
    - cmap: Colormap to use for the scatter plot (default: 'Spectral_r')
    - s: Size of the scatter plot markers (default: 50)
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap=cmap, s=s)
    cbar = fig.colorbar(sc)
    cbar.set_label('RemnantType')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Create a legend
    handles = []
    for rem_type, label in remnant_type.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=sc.cmap(sc.norm(rem_type)), markersize=10, label=label))
    ax.legend(handles=handles, title='RemnantType', loc=legend_loc, fontsize='small')

    # Show the plot
    plt.show()

def plot_scatter_with_colorbar_combined(df, x_col, y_col, color_col, color_dict, ax, legend_loc='best'):
    """
    Plot a scatter plot with a colorbar.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    x_col (str): Column name for the x-axis.
    y_col (str): Column name for the y-axis.
    color_col (str): Column name for the color coding.
    color_dict (dict): Dictionary for color coding.
    ax (matplotlib.axes.Axes): Axes object to plot on.
    legend_loc (str): Location of the legend.
    """
    scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap='Spectral_r', s=1)
    cbar = plt.colorbar(scatter, ax=ax, ticks=list(color_dict.keys()))
    cbar.ax.set_yticklabels(list(color_dict.values()))
    cbar.set_label(color_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}')
    ax.legend(loc=legend_loc)
    ax.set_xscale('log')
    ax.set_yscale('log')

def plot_hr_diagrams(df, cross_df_filtered):
    """
    Plot Hertzsprung-Russell diagrams with various color codings.

    Parameters:
    df (pd.DataFrame): DataFrame containing the original data.
    cross_df_filtered (pd.DataFrame): DataFrame containing the filtered cross-matched data.
    """
    # Change period from days to years
    df['P'] = df['P'] / 365.25

    # Define the remnant type dictionary
    remnant_type = {0: "NotARemnant - 0", 1: "HeWD - 1", 2: "COWD - 2", 3: "ONeWD - 3", 4: "NS_ECSN - 4",
                5: "NS_CCSN - 5", 6: "BH - 6", -1: "Empty - -1"}

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))

    # Plot 1: Color-coded by RemnantType_1
    axs[0, 0].scatter(cross_df_filtered['bp_rp_corrected'], cross_df_filtered['M_G'], s=1, c='k', alpha=0.5)
    sc1 = axs[0, 0].scatter(df['bp_rp_corrected'], df['M_G'], c=df['RemnantType_1'], cmap='Spectral_r', s=1)
    cbar = fig.colorbar(sc1, ax=axs[0, 0], ticks=list(remnant_type.keys()))
    cbar.ax.set_yticklabels(list(remnant_type.values()))
    cbar.set_label('Remnant Type')
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_xlabel('(BP - RP) Corrected Color')
    axs[0, 0].set_ylabel('M_G (Absolute G Magnitude)')
    axs[0, 0].set_title('Hertzsprung-Russell Diagram (RemnantType_1)')

    # Plot 2: Color-coded by Mass Primary
    axs[0, 1].scatter(cross_df_filtered['bp_rp_corrected'], cross_df_filtered['M_G'], s=1, c='k', alpha=0.5)
    sc2 = axs[0, 1].scatter(df['bp_rp_corrected'], df['M_G'], c=df['M_1'], cmap='Spectral_r', s=1)
    cbar = fig.colorbar(sc2, ax=axs[0, 1])
    cbar.set_label('Mass Primary (M_sun)')
    axs[0, 1].invert_yaxis()
    axs[0, 1].set_xlabel('(BP - RP) Corrected Color')
    axs[0, 1].set_ylabel('M_G (Absolute G Magnitude)')
    axs[0, 1].set_title('Hertzsprung-Russell Diagram (Mass Primary)')

    # Plot 3: Color-coded by Mass Secondary
    axs[0, 2].scatter(cross_df_filtered['bp_rp_corrected'], cross_df_filtered['M_G'], s=1, c='k', alpha=0.5)
    sc3 = axs[0, 2].scatter(df['bp_rp_corrected'], df['M_G'], c=df['M_2'], cmap='Spectral_r', s=1)
    cbar = fig.colorbar(sc3, ax=axs[0, 2])
    cbar.set_label('Mass Secondary (M_sun)')
    axs[0, 2].invert_yaxis()
    axs[0, 2].set_xlabel('(BP - RP) Corrected Color')
    axs[0, 2].set_ylabel('M_G (Absolute G Magnitude)')
    axs[0, 2].set_title('Hertzsprung-Russell Diagram (Mass Secondary)')

    # Plot 4: Color-coded by ruwe with normalization from 0 to 100
    norm_ruwe = Normalize(vmin=0, vmax=100)
    axs[1, 0].scatter(cross_df_filtered['bp_rp_corrected'], cross_df_filtered['M_G'], s=1, c='k', alpha=0.5)
    sc4 = axs[1, 0].scatter(df['bp_rp_corrected'], df['M_G'], c=df['ruwe'], cmap='Spectral_r', s=1, norm=norm_ruwe)
    cbar = fig.colorbar(sc4, ax=axs[1, 0])
    cbar.set_label('ruwe')
    axs[1, 0].invert_yaxis()
    axs[1, 0].set_xlabel('(BP - RP) Corrected Color')
    axs[1, 0].set_ylabel('M_G (Absolute G Magnitude)')
    axs[1, 0].set_title('Hertzsprung-Russell Diagram (ruwe)')

    # Plot 5: Color-coded by Mass Ratio with normalization from 0 to 5
    norm_mass_ratio = Normalize(vmin=0, vmax=5)
    axs[1, 1].scatter(cross_df_filtered['bp_rp_corrected'], cross_df_filtered['M_G'], s=1, c='k', alpha=0.5)
    sc5 = axs[1, 1].scatter(df['bp_rp_corrected'], df['M_G'], c=df['q'], cmap='Spectral_r', s=1, norm=norm_mass_ratio)
    cbar = fig.colorbar(sc5, ax=axs[1, 1])
    cbar.set_label('mass ratio')
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_xlabel('(BP - RP) Corrected Color')
    axs[1, 1].set_ylabel('M_G (Absolute G Magnitude)')
    axs[1, 1].set_title('Hertzsprung-Russell Diagram (mass ratio)')

    # Plot 6: Color-coded by Period with a logarithmic color scale normalized from 10^-1 to 10^1
    norm_period = LogNorm(vmin=10**-1, vmax=10**1)
    axs[1, 2].scatter(cross_df_filtered['bp_rp_corrected'], cross_df_filtered['M_G'], s=1, c='k', alpha=0.5)
    sc6 = axs[1, 2].scatter(df['bp_rp_corrected'], df['M_G'], c=df['P'], cmap='Spectral_r', s=1, norm=norm_period)
    cbar = fig.colorbar(sc6, ax=axs[1, 2])
    cbar.set_label('Period (log scale) (Myrs)')
    axs[1, 2].invert_yaxis()
    axs[1, 2].set_xlabel('(BP - RP) Corrected Color')
    axs[1, 2].set_ylabel('M_G (Absolute G Magnitude)')
    axs[1, 2].set_title('Hertzsprung-Russell Diagram (Period)')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
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

def inferred_P(A, sigma_spectroscopic_error, sigma_astrometric_error, parallax, zeta_0, beta_0, q, l):
    term_1 = (2 * np.pi * A) / (sigma_spectroscopic_error)
    term_2 = sigma_astrometric_error / parallax
    term_3 = zeta_0 / beta_0
    term_4 = (q*(1+l))/(np.abs(q-l))
    return term_1 * term_2 * term_3 * term_4

def P(A, sigma_spectroscopic_error, sigma_astrometric_error, parallax, zeta_0, beta_0):
    term_1 = (2 * np.pi * A) / (sigma_spectroscopic_error)
    term_2 = sigma_astrometric_error / parallax
    term_3 = zeta_0 / beta_0
    return term_1 * term_2 * term_3

def infer_q(q, l):

    ###### Calculate alpha ######
    alpha_term_1 = (q**2*np.abs(q-l))
    alpha_term_2 = ((l+1)*(1+q)**2)
    alpha =  alpha_term_1 / alpha_term_2

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
        # uwe=np.linalg.norm(pos-np.matmul(design,fitparams))/(errs*np.sqrt(2*len(ts)-5))
        thisRow['UWE'] = results['uwe']

        thisRow['N_obs'] = results['astrometric_n_obs_al']
        # thisRow['frac_good'] = results['astrometric_n_good_obs_al'] / results['astrometric_n_obs_al']
        thisRow['N_vis'] = results['visibility_periods_used']
        # thisRow['AEN'] = results['astrometric_excess_noise']

    return allData

def gaia_observation(nTest, evolved_binaries, astromet, Source, dr3_sl, alError=1):
    dataNames = ('RA', 'Dec', 'pmRA', 'pmDec', 'pllx', 'M_1', 'M_2',
                 'M_tot', 'q', 'l', 'a', 'e', 'P', 'tPeri',
                 'Luminosity_0','Luminosity_1', 'Temperature_0', 'Temperature_1',
                 'vTheta', 'vPhi', 'vOmega',
                 'predict_dTheta', 'simple_dTheta',
                 'N_obs', 'sigma_al', 'sigma_ac',
                 'fit_ra', 'fit_dec', 'fit_pmrac', 'fit_pmdec', 'fit_pllx',
                 'sigma_rac', 'sigma_dec', 'sigma_pmrac', 'sigma_pmdec', 'sigma_pllx',
                 'N_vis', 'frac_good', 'AEN', 'UWE', 'ast_errors', 'rv_errors',
                 'astrometric_chi2_al', 'astrometric_n_good_obs_al', 'astrometric_params_solved',
                 'ruwe', 'RemnantType_0', 'RemnantType_1'
                )
    
    allData = astropy.table.Table(names=dataNames)

    for i in tqdm(range(nTest)):
        allData.add_row()
        thisRow = allData[i]
        
        params = astromet.params()
        params.ra = 360 * np.random.rand(1)[0]
        params.dec = 180 / np.pi * np.arcsin(np.random.uniform(low=-1, high=1))
        
        c = Source(params.ra, params.dec, unit='deg')
        sl = dr3_sl(c, return_times=True, return_angles=True)
        ts = 2010 + np.squeeze(np.hstack(sl['times'])) / 365.25
        sort = np.argsort(ts)
        ts = np.double(ts[sort])
        
        phis = np.squeeze(np.hstack(sl['angles']))[sort]
        
        params.parallax = 10*np.power(np.random.rand(),-1/3)  # parallax in mas
        params.pmrac = params.parallax * (1) * np.random.randn()
        params.pmdec = params.parallax * (1) * np.random.randn()
        params.mass_1 = evolved_binaries['Mass_0'][i]  # primary mass in Msun
        params.mass_2 = evolved_binaries['Mass_1'][i]  # secondary mass in Msun
        params.period = evolved_binaries['Period'][i]  # periods between 0.03 and 30 years
        params.l = evolved_binaries['l'][i]  # uniform light ratio
        params.q = evolved_binaries['q'][i]  # uniform mass ratio
        params.a = evolved_binaries['Semimajor'][i]  # semi-major axis in AU  
        params.e = evolved_binaries['Eccentricity'][i]  # eccentricity
        params.L_0 = evolved_binaries['Luminosity_0'][i]  # primary luminosity in Lsun
        params.L_1 = evolved_binaries['Luminosity_1'][i]  # secondary luminosity in Lsun
        params.T_0 = evolved_binaries['Temperature_0'][i]  # primary temperature in K
        params.T_1 = evolved_binaries['Temperature_1'][i]  # secondary temperature in K
        params.vtheta = np.arccos(-1 + 2 * np.random.rand())
        params.vphi = 2 * np.pi * np.random.rand()
        params.vomega = 2 * np.pi * np.random.rand()
        orbitalPhase = np.random.rand()  # fraction of an orbit completed at t=0
        params.tperi = params.period * orbitalPhase
        params.ast_errors = 10 ** np.random.uniform(-2, 4)
        params.rv_errors = 10 ** 3 * 10 ** np.random.uniform(-3, 2)
        params.remnant_type_0 = evolved_binaries['RemnantType_0'][i]
        params.remnant_type_1 = evolved_binaries['RemnantType_1'][i]
        
        thisRow['RA'] = float(params.ra)
        thisRow['Dec'] = float(params.dec)
        thisRow['pmRA'] = float(params.pmrac)
        thisRow['pmDec'] = float(params.pmdec)
        thisRow['pllx'] = float(params.parallax)
        thisRow['M_1'] = float(params.mass_1)
        thisRow['M_2'] = float(params.mass_2)
        thisRow['M_tot'] = float(params.mass_1 + params.mass_2)
        thisRow['q'] = float(params.q)
        thisRow['l'] = float(params.l)
        thisRow['a'] = float(params.a)
        thisRow['e'] = float(params.e)
        thisRow['P'] = float(params.period)
        thisRow['Luminosity_0'] = float(params.L_0)
        thisRow['Luminosity_1'] = float(params.L_1)
        thisRow['Temperature_0'] = float(params.T_0)
        thisRow['Temperature_1'] = float(params.T_1)
        thisRow['tPeri'] = float(params.tperi)
        thisRow['vTheta'] = float(params.vtheta)
        thisRow['vPhi'] = float(params.vphi)
        thisRow['vOmega'] = float(params.vomega)
        thisRow['sigma_al'] = float(alError)
        #thisRow['sigma_ac'] = float(acError)
        thisRow['ast_errors'] = float(params.ast_errors)
        thisRow['rv_errors'] = float(params.rv_errors)
        thisRow['RemnantType_0'] = float(params.remnant_type_0)
        thisRow['RemnantType_1'] = float(params.remnant_type_1)

        trueRacs, trueDecs = astromet.track(ts, params)

        # added .astype(float) to avoid astromet error
        phis = phis.astype(float)
        
        t_obs, x_obs, phi_obs, rac_obs, dec_obs = astromet.mock_obs(ts, phis, trueRacs, trueDecs, err=alError)
        
        fitresults = astromet.fit(t_obs, x_obs, phi_obs, alError, params.ra, params.dec)
        results = astromet.gaia_results(fitresults)
        
        # print('ra, dec, pllx, pmrac, pmdec ', params.ra, params.dec, params.parallax, params.pmrac, params.pmdec)
        # print(results)
        
        # bug somewhere in these
        #thisRow['simple_dTheta'] = astromet.dtheta_simple(params)
        #thisRow['predict_dTheta'] = astromet.dtheta_full(params, np.min(ts), np.max(ts))  
        
        thisRow['fit_ra'] = float(results['ra'])
        thisRow['fit_dec'] = float(results['dec'])
        thisRow['fit_pmrac'] = float(results['pmra'])
        thisRow['fit_pmdec'] = float(results['pmdec'])
        thisRow['fit_pllx'] = float(results['parallax'])

        thisRow['sigma_rac'] = float(results['ra_error'])
        thisRow['sigma_dec'] = float(results['dec_error'])
        thisRow['sigma_pmrac'] = float(results['pmra_error'])
        thisRow['sigma_pmdec'] = float(results['pmdec_error'])
        thisRow['sigma_pllx'] = float(results['parallax_error'])
        thisRow['astrometric_chi2_al'] = float(results['astrometric_chi2_al'])
        thisRow['astrometric_n_good_obs_al'] = float(results['astrometric_n_good_obs_al'])
        thisRow['astrometric_params_solved'] = float(results['astrometric_params_solved'])

        # results['UWE'] --> results['uwe']
        # uwe = np.linalg.norm(pos - np.matmul(design, fitparams)) / (errs * np.sqrt(2 * len(ts) - 5))
        thisRow['UWE'] = float(results['uwe'])

        thisRow['N_obs'] = float(results['astrometric_n_obs_al'])
        # thisRow['frac_good'] = results['astrometric_n_good_obs_al'] / results['astrometric_n_obs_al']
        thisRow['N_vis'] = float(results['visibility_periods_used'])
        # thisRow['AEN'] = results['astrometric_excess_noise']

        # calculate ruwe
        thisRow['ruwe'] = np.sqrt(results['astrometric_chi2_al'] / (results['astrometric_n_good_obs_al'] - results['astrometric_params_solved']))

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


                                                        #################### HR Diagram ####################

# Function to calculate distance from parallax
def calculate_distance(parallax):
    """Convert parallax in milliarcseconds to distance in parsecs."""
    # Ensure non-negative parallax
    with np.errstate(divide='ignore'):
        distance = 1000 / parallax
    # Handle zero and negative parallax by setting distance to NaN
    distance[parallax <= 0] = np.nan
    return distance

# Function to calculate absolute magnitude
def calculate_absolute_magnitude(phot_g_mean_mag, distance):
    """Calculate absolute magnitude from apparent magnitude and distance."""
    return phot_g_mean_mag - 5 * np.log10(distance) + 5

                                                #################### SEVN - Simulation of Binaries ####################

# Constants
G = 6.674e-11  # Gravitational constant in m^3/kg/s^2
M_sun = 1.989e30  # Mass of the sun in kg
AU = 1.496e11  # Astronomical unit in meters
day_to_sec = 86400  # Conversion from days to seconds

def create_binary_population(Nbin=100001, backup=1, z=0.02, mass_ranges=[2.3, 100], alphas=[-2.3], q_max=4.0, mass_min=2.3, model='sana12', period_units='day'):
    """
    Create a population of binary stars and save the data to a PETAR file.

    Parameters:
    - Nbin: Number of binary systems (default: 100001)
    - backup: Number of backup systems (default: 1)
    - z: Metallicity (default: 0.02)
    - mass_ranges: List of mass ranges for the initial mass function (default: [0.1,0.5,150])
    - alphas: List of power-law slopes for the initial mass function (default: [-2.3])
    - q_max: Maximum mass ratio (default: 4.0)
    - mass_min: Minimum mass (default: 2.3)
    - model: Model to use for the binary population (default: 'sana12')
    - period_units: Units for the period ('Myr', 'yr', 'day') (default: 'day')
    """
    # Create a population of binaries
    binSana = pop.Binaries(Nbin, model=model, mass_ranges=mass_ranges, alphas=alphas, q_max=q_max, mass_min=mass_min)
    
    # Save the population as input for MOBSE
    binSana.save_mobse_input('mobse', z, 13600, backup)

    type1 = [1] * Nbin
    type2 = [1] * Nbin
    tini = [0.0] * Nbin

    # Extract masses and periods from the population
    m1 = binSana.population['m1']  # mass in solar masses
    m2 = binSana.population['m2']  # mass in solar masses
    e = binSana.population['ecc']  # eccentricity

    # Convert periods to the desired units
    if period_units == 'Myr':
        p = binSana.population['p'] / (365.25 * 1e6)  # Convert days to Myr
    elif period_units == 'yr':
        p = binSana.population['p'] / 365.25  # Convert days to years
    elif period_units == 'day':
        p = binSana.population['p'] 
    else:
        raise ValueError("Invalid period_units. Choose from 'Myr', 'yr', or 'day'.")
    
    a = binSana.population['a']

    # Calculate semi-major axis (in AU)
    Z = [z] * Nbin

    # Save to PETAR file with semi-major axis included
    np.savetxt("petar_" + str(z) + ".in", 
        np.c_[m1, m2, type1, type2, p, e, a, Z, tini],
        fmt="%4.4f %4.4f %i %i %15.9f %1.4f %s %1.4f %1.2f",
        header=str(Nbin - backup), comments='')

def convert_in_to_csv(input_file, output_file):
    """
    Convert a .in file to a .csv file.

    Parameters:
    input_file (str): Path to the input .in file.
    output_file (str): Path to the output .csv file.
    """
    try:
        # Load the data from the .in file, skipping the first row
        df = pd.read_csv(input_file, delim_whitespace=True, header=None, skiprows=1)

        # Specify column names based on the expected format
        # check if name of the file starts with 'petar'
        if input_file.startswith('petar'):
            column_names = ['m1', 'm2', 'type1', 'type2', 'period', 'eccentricity', 'a', 'Z', 'tini']
        else:
            column_names = ["name", "m1", "m2", "period", "eccentricity", "Z", "tmax"]
        df.columns = column_names[:df.shape[1]]  # Only set columns that exist

        # Save the DataFrame to a .csv file
        df.to_csv(output_file, index=False)

        print(f"Conversion successful! Saved as {output_file}")

    except Exception as e:
        print(f"Error reading the file: {e}")

def run_sevn_simulations(IC_df, num_rows=10, t_end=1000, snmodel="delayed", rseed=0):
    """
    Run SEVN simulations for a binary star population.

    Parameters:
    - csv_file: Path to the CSV file containing the binary star population data (default: "petar_0.2.csv")
    - num_rows: Number of rows to read from the CSV file (default: 10)
    - t_end: End time for the SEVN evolution (default: 1000)
    - snmodel: Supernova model to use (default: "delayed")
    - rseed: Random seed for reproducibility (default: 0)
    """
    SEVNmanager.init()
    
    # Read the CSV file
    df_petar = IC_df.head(num_rows)
    binary_ids = df_petar['binary_id']

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame()

    # Loop over the arrays with tqdm progress bar
    for i in tqdm(range(0,num_rows), desc="Running SEVN simulations"):
        output, log = sevnwrap.evolve_binary(Semimajor=df_petar['a'][i],
                                            Eccentricity=df_petar['eccentricity'][i],
                                            Mzams_0=df_petar['m1'][i],
                                            Z_0=df_petar['Z'][i],
                                            Mzams_1=df_petar['m2'][i],
                                            Z_1=df_petar['Z'][i],
                                            tend=int(t_end),
                                            snmodel=snmodel,  # SN model to use, see the SEVN userguide
                                            rseed=rseed  # Random seed for reproducibility, if 0 or not included a random value will be generated
        ) 

        # Convert output to DataFrame and append binary_id
        output_df = pd.DataFrame(output)
        output_df['binary_id'] = binary_ids[i]
        results_df = pd.concat([results_df, output_df], ignore_index=True)

    # Close SEVN manager
    SEVNmanager.close()

    return results_df

def process_gaia_data(file_path):
    """
    Load and process Gaia data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing Gaia data.

    Returns:
    pd.DataFrame: Processed DataFrame with corrected magnitudes and distances.
    """
    # Load data in a dataframe
    cross_df = pd.read_csv(file_path)

    # Correct for extinction and reddening
    cross_df['G_corrected'] = cross_df['phot_g_mean_mag'] - cross_df['ag_gspphot']
    cross_df['bp_rp_corrected'] = cross_df['bp_rp'] - cross_df['ebpminrp_gspphot']

    # Calculate distance in parsecs from parallax (parallax is in milliarcseconds)
    cross_df['distance_pc'] = calculate_distance(cross_df['parallax'])

    # Calculate absolute magnitude M_G
    cross_df['M_G'] = calculate_absolute_magnitude(cross_df['G_corrected'], cross_df['distance_pc'])

    # Filter out rows with invalid distances (e.g., NaN values)
    cross_df_filtered = cross_df.dropna(subset=['M_G', 'bp_rp_corrected'])

    return cross_df_filtered

def match_and_update(df, cross_df_filtered):
    """
    Match rows from df with cross_df_filtered using nearest neighbor search and update df with matched values.

    Parameters:
    df (pd.DataFrame): DataFrame containing the original data.
    cross_df_filtered (pd.DataFrame): DataFrame containing the filtered cross-matched data.

    Returns:
    pd.DataFrame: Updated DataFrame with matched M_G and bp_rp_corrected values.
    """
    # Select the columns to match on from both dataframes
    df_coords = df[['P', 'pllx', 'ruwe']].to_numpy()
    cross_df_coords = cross_df_filtered[['period', 'parallax', 'ruwe']].to_numpy()

    # Remove rows with NaN or infinite values from both dataframes
    df_clean = df[['P', 'pllx', 'ruwe']].replace([np.inf, -np.inf], np.nan).dropna()
    cross_df_clean = cross_df_filtered[['period', 'parallax', 'ruwe']].replace([np.inf, -np.inf], np.nan).dropna()

    # Create a KDTree for fast nearest neighbor search
    tree = cKDTree(cross_df_clean.to_numpy())

    # Find the nearest match in cross_df_filtered for each row in df_clean
    distances, indices = tree.query(df_clean.to_numpy())

    # Add the matched M_G and bp_rp_corrected values from cross_df_filtered back to df
    df.loc[df_clean.index, 'M_G'] = cross_df_filtered.iloc[indices]['M_G'].values
    df.loc[df_clean.index, 'bp_rp_corrected'] = cross_df_filtered.iloc[indices]['bp_rp_corrected'].values

    return df