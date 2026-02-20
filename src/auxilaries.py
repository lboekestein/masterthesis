"""
Auxiliary functions for the thesis project.
"""

import pandas as pd
import requests
import calendar


def retrieve_data_from_api(run: str, 
                           loa: str, 
                           verbose=False,
                           date_start=None,
                           date_end=None
                           ) -> pd.DataFrame:
    """ 
    Retrieve data from the API and return as a pandas DataFrame.

    Args:
        run (str): run identifier (e.g. fatalities001_2022_06_t01)
        loa (str): level of analysis; cm or pgm
        verbose (bool): whether to print progress messages
        start_date (str): start date for filtering data (YYYY-MM-DD), default is None
        end_date (str): end date for filtering data (YYYY-MM-DD), default is None

    Returns:
        pd.DataFrame: DataFrame containing the api data
    """

    # set up api url
    if date_start and date_end:
        api_url = f'https://api.viewsforecasting.org/{run}/{loa}?date_start={date_start}&date_end={date_end}'
    else:
        api_url = f'https://api.viewsforecasting.org/{run}/{loa}'

    # get response
    response = requests.get(api_url)

    # check response status
    page_data=response.json()

    master_list=[]
    master_list+=page_data['data']

    # loop through pages
    i = 1
    while page_data['next_page'] != '':

        # if verbose, print progress
        if verbose:
            print(f"Retrieving page {i}/{page_data['page_count']-1} at {loa} level...         ", end='\r', flush=True)

        r=requests.get(page_data['next_page'])
        page_data=r.json()

        master_list+=page_data['data']
        i += 1

    # convert to dataframe
    forecasts=pd.DataFrame(master_list)

    return forecasts

def date_to_month_id(year: int, month: int) -> int:
    """ 
    Convert year and month to month_id.

    Args:
        year (int): year (e.g. 2022)
        month (int): month (1-12)
    
    Returns:
        int: month_id
    """

    return (year - 1980) * 12 + month


def month_id_to_ym(month_id: int) -> str:
    """
    Converts month_id to month name and year

    Args:
        month_id: integer representing the month id
    
    Returns:
        String consisting of month name and year
    """

    offset = month_id - 1
    year = 1980 + offset // 12
    month_num = offset % 12 + 1
    month_name = calendar.month_name[month_num]

    return f"{month_name} {year}"