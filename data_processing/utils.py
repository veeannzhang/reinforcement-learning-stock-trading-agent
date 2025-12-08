import pandas as pd


def get_indices_of_first_datapoint(na_df: pd.DataFrame) -> pd.DataFrame:
    """Returns the index identifying the first non-null data point
    of each column in a DataFrame.

    Arguments:
    ----------
        na_df : pd.DataFrame
            result of df.isna() - True for NA value, False otherwise.
    """
    # get iloc of first datapoint for each column
    first_datapoint_iloc = (
        na_df
        .cumprod(axis=0) # in each column, values after first non-null set to 0
        .sum(axis=0) # summing up a column gives the iloc of first non-null
    )

    # get loc of those positions
    first_datapoint_loc = na_df.index[first_datapoint_iloc]

    # sort column by loc of first data (descending)
    first_datapoint_df = pd.DataFrame({
        'first datapoint': first_datapoint_loc,
        'column': na_df.columns
    }).sort_values(['first datapoint', 'column'], ascending=False)

    return first_datapoint_df