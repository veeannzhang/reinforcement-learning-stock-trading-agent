import warnings
import pandas as pd


def _print_statistics(
    df: pd.DataFrame, 
    step: str, 
    warn: bool=False
) -> None:
    """Prints simple summary statistics about the input DataFrame.
    Arguments
    ----------
        df : pd.DataFrame
        step : str
            Name of the preprocessing step that produced `df`.
        warn : bool
            Toggles warning for missing values.
    """
    nas = df.isna()
    na_count = nas.sum().values[0]
    na_perc = round(nas.mean().values[0] * 100, 2)
    print(
        f'\tStep: {step}',
        f'Rows: {df.shape[0]}',
        f'NA count: {na_count}', 
        f'NA %: {na_perc}',
        sep=' | '
    )
    if warn and na_count > 0:
        warnings.warn('There are still missing values!')
        


def format_raw_series(
    series_id: str,
    df: pd.DataFrame,
    print_std: bool=True
) -> pd.DataFrame:
    """Sets index and column names to the raw data returned by
    the FRED API .get_series() method.
    Arguments
    ----------
        series_id : str
            FRED series identifier.
        df : pd.DataFrame
        print_std : bool
            Toggles printing of data stats.
    """
    dt_idx = pd.DatetimeIndex(df['date'])
    df = df.set_index(dt_idx)
    df = df.rename({'value': series_id}, axis=1)
    df = df[[series_id]]
    
    if print_std:
        _print_statistics(df, step='format')

    return df


def expand_full_dates(
    df : pd.DataFrame,
    freq : str,
    print_std: bool=True
) -> pd.DataFrame:
    """Maps the data to the full range of dates between
    the min. and max. dates, at the given frequency.
    May generate NAs if a date did not have a value.
    Arguments
    ----------
        df : pd.DataFrame
        freq : str
            the frequency of the series data
        print_std : bool
            Toggles printing of data stats.
    """
    # get min/max dt from data
    min_dt = df.index.min()
    max_dt = df.index.max()

    # generate full datetime range
    full_dt_range = pd.date_range(min_dt, max_dt, freq=freq)

    # map data to full range
    df = df.reindex(full_dt_range)
    
    if print_std:
        _print_statistics(df, step='expand dates')

    return df


def impute_missing_data(
    col : str,
    df : pd.DataFrame,
    print_std: bool=True
) -> pd.DataFrame:
    """Imputes missing data using the rolling mean method,
    with a window size of 7.
    Arguments
    ----------
        col : str
        df : pd.DataFrame
        print_std : bool
            Toggles printing of data stats.
    """
    roll_mean = df[col].rolling(window=7, min_periods=1).mean()
    df[col] = df[col].fillna(roll_mean)
    
    if print_std:
        _print_statistics(df, step='impute', warn=True)

    return df