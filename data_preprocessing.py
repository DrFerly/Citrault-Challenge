import numpy as np


def join_and_clean_data(df_claims, df_damage, string_columns):
    df_claims = df_claims.replace('\'', '')
    for str_col in string_columns:
        df_claims[str_col] = df_claims[str_col].apply(lambda x: x.replace("'", ""))
    df_claims["IDpol"] = df_claims["IDpol"].astype(int)
    df_claims["IDpol"] = df_claims["IDpol"].astype('string')
    df_damage["IDpol"] = df_damage["IDpol"].astype(str)
    # print(df_claims['ClaimNb'].value_counts())
    df_claims = df_claims[df_claims["ClaimNb"]<=3]
    df_claims = df_claims[df_claims["Exposure"]<=1]
    df = df_claims.merge(df_damage, on=["IDpol"], how='left')
    df_total_amount = df[["IDpol", "ClaimAmount"]].groupby(['IDpol']).sum()
    df = df_claims.merge(df_total_amount, on=["IDpol"], how='left')
    df["ClaimAmount"] = df["ClaimAmount"].replace(np.nan, 0)
    # df["Total_ClaimAmount"] = df["ClaimNb"] * df["ClaimAmount"]
    # df = df[~df["ClaimAmount"].isnull()]
    return df


def encode_string_columns(colname, df):
    """
    doc str
    """
    # df_encoded = pd.get_dummies(df[colname])
    # df = df.drop(columns=colname)
    # df = df.join(df_encoded)
    # return df
    region_array = df[colname].unique()
    int_array = []
    dict_region_mapping={}
    for i in range(0, len(region_array)):
        int_array.append(i)
    for region, i in zip(region_array, int_array):
        dict_region_mapping.update({region:i})
    df[colname] = df[colname].map(dict_region_mapping)
    return df