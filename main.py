"""
coding challenge for french car insurance
"""

import pandas as pd
import numpy as np
import os
from pydiap.transformations import Input, Output, transform
from pydiap import sparktools
from pyspark.sql import functions as F
import re
from counts import counts

def main():
    """"
    main function to load data
    """
    claims_filepath = r"C:\Users\LILEL\Desktop\Citrault-Challenge\raw_data\freMTPL2freq.csv"
    histogram_plotpath = r"C:\Users\LILEL\Desktop\Citrault-Challenge\plots\histogram"
    damage_filepath = r"C:\Users\LILEL\Desktop\Citrault-Challenge\raw_data\freMTPL2sev.csv"
    df_claims = pd.read_csv(claims_filepath)
    df_damage = pd.read_csv(damage_filepath)
    df_claims = df_claims.replace('\'', '')
    string_columns = ['Area', 'VehBrand', 'VehGas', 'Region']
    for str_col in string_columns:
        df_claims[str_col] = df_claims[str_col].apply(lambda x: x.replace("'", ""))
    counts(df_claims, histogram_plotpath)

    print(df_damage)

if __name__ == "__main__":
    main()