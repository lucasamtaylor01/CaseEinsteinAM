import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

def clean_data(df):

    # Standardize string columns
    df.columns = df.columns.str.upper()
    df.columns = df.columns.str.replace(' ', '_')
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    df[string_cols] = df[string_cols].apply(lambda x: x.astype(str).str.strip().str.upper())
    df = df.rename(columns={'SUB-CATEGORY': 'SUB_CATEGORY'})

    # Remove duplicate orders
    df = df.drop_duplicates(subset='ORDER_ID')

    # Drop irrelevant columns
    df = df.drop(columns=['CUSTOMER_NAME', 'POSTAL_CODE', 'ORDER_ID'])

    # Parse date columns
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'])
    df['SHIP_DATE'] = pd.to_datetime(df['SHIP_DATE'])

    # Remove inconsistent records where ship date precedes order date
    df = df[df['SHIP_DATE'] >= df['ORDER_DATE']]


    state_map = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT',
        'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
        'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL',
        'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME',
        'MARYLAND': 'MD', 'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI',
        'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
        'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM',
        'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND',
        'OHIO': 'OH', 'OKLAHOMA': 'OK', 'OREGON': 'OR',
        'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI',
        'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD',
        'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA',
        'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY',
        'DISTRICT OF COLUMBIA': 'DC'
    }

    df['STATE'] = df['STATE'].map(state_map)
    df['COUNTRY'] = df['COUNTRY'].replace({'UNITED STATES': 'USA'})

    # Compute net sales
    df['NET_SALES'] = df['SALES'] * df['QUANTITY'] * (1-df['DISCOUNT'])
    df = df.drop(columns=['SALES', 'QUANTITY', 'DISCOUNT'])

    return df


def treat_outliers(df):
    # Log-transform NET_SALES for outlier treatment
    df['NET_SALES_LOG'] = np.log1p(df['NET_SALES'])

    # Quantile bounds for outlier removal
    q_low = 0.15
    q_high = 0.85

    # Filter outliers in PROFIT and NET_SALES_LOG
    q_profit_low = df['PROFIT'].quantile(q_low)
    q_profit_high = df['PROFIT'].quantile(q_high)

    q_net_sales_low = df['NET_SALES_LOG'].quantile(q_low)
    q_net_sales_high = df['NET_SALES_LOG'].quantile(q_high)

    df = df[
        (df['PROFIT'].between(q_profit_low, q_profit_high)) &
        (df['NET_SALES_LOG'].between(q_net_sales_low, q_net_sales_high))
    ]

    df = df.reset_index(drop=True)
    df = df.drop(columns=['NET_SALES_LOG'])

    return df

def scale_data(df, cols):
    # Standardize PROFIT and NET_SALES
    scaler = StandardScaler()
    df[[col + '_SCALED' for col in cols]] = scaler.fit_transform(df[cols])
    return df, scaler

def preprocess(df):
    # Preprocessing pipeline for modeling
    df = scale_data(df, ['PROFIT', 'NET_SALES'])[0]
    df = treat_outliers(df)
    return df

def data_clustering(df):
    # Select relevant columns for clustering and aggregate by customer
    df_clustering = df[['ORDER_DATE',
                        'SHIP_MODE',
                        'CUSTOMER_ID',
                        'SEGMENT',
                        'CATEGORY',
                        'REGION',
                        'NET_SALES',
                        'PROFIT',
                        'NET_SALES_SCALED',
                        'PROFIT_SCALED']].copy()

    # Group by customer and aggregate relevant columns
    df_clustering = df_clustering.groupby('CUSTOMER_ID', as_index=False).agg({
        'ORDER_DATE': 'first',
        'SHIP_MODE': 'first',
        'CATEGORY': 'first',
        'REGION': 'first',
        'SEGMENT': 'first',
        'NET_SALES': 'sum',
        'PROFIT': 'sum',
        'NET_SALES_SCALED': 'sum',
        'PROFIT_SCALED': 'sum'
    })

    # Prepare data for clustering with one-hot encoding
    X_scaled = df_clustering.drop(columns=['CUSTOMER_ID', 'NET_SALES', 'PROFIT'])
    X_scaled['ORDER_DATE'] = pd.to_datetime(df_clustering['ORDER_DATE'], errors='coerce').dt.year
    cols_one_hot = ['SHIP_MODE', 'SEGMENT', 'REGION', 'CATEGORY', 'ORDER_DATE']

    X_scaled = pd.get_dummies(
        X_scaled,
        columns=cols_one_hot,
    )

    return X_scaled, df_clustering

def build_temporal_dict(df_clustering):

    # Prepare data for temporal analysis by cluster
    df_clustering['ORDER_DATE'] = pd.to_datetime(df_clustering['ORDER_DATE'])

    # Group by month and cluster, summing total profit
    df_temporal = df_clustering.groupby(
        [df_clustering['ORDER_DATE'].dt.to_period('M'), 'CLUSTER']
    )['PROFIT'].sum().reset_index()

    # Convert ORDER_DATE to datetime and sort
    df_temporal['ORDER_DATE'] = df_temporal['ORDER_DATE'].dt.to_timestamp()
    df_temporal = df_temporal.sort_values('ORDER_DATE').reset_index(drop=True)

    # Split data by cluster
    df_temporal_cluster_0 = df_temporal[df_temporal['CLUSTER'] == 0].copy()
    df_temporal_cluster_1 = df_temporal[df_temporal['CLUSTER'] == 1].copy()
    df_temporal_cluster_2 = df_temporal[df_temporal['CLUSTER'] == 2].copy()

    # Build dictionary of temporal data per cluster
    df_temporal_dict = {
        0: df_temporal_cluster_0,
        1: df_temporal_cluster_1,
        2: df_temporal_cluster_2
    }

    # Filter to data from July 2020 onward
    for i in df_temporal_dict:
        df_temporal_dict[i] = df_temporal_dict[i][
            df_temporal_dict[i]['ORDER_DATE'] >= '2020-07-01'
        ]

    # Test stationarity and difference each cluster's series as needed
    for i, data in df_temporal_dict.items():
        stationary = test_stationarity(data['PROFIT'])
        df_temporal_dict[i] = df_temporal_dict[i].loc[stationary.index].copy()
        df_temporal_dict[i]['PROFIT'] = stationary

    return df_temporal_dict


def test_stationarity(timeseries, diff_order=0):

    # Dickey-Fuller stationarity test; recursively differences until stationary
    result = adfuller(timeseries)
    if result[1] >= 0.05:
        return test_stationarity(timeseries.diff().dropna(), diff_order + 1)

    return timeseries
