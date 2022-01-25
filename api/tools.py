# importing necessary packages

import pandas as pd
import datetime as dt

pd.options.mode.chained_assignment = None


def history_accounts(df, history=180):

    """
    Returns accounts' indexes which have more than 180 
    days of history and less than 180 days of history  
    """
    kept_accounts = []
    rejected_accounts = []
    for index in df.id.unique():
        old = df[df["id"] == index].reset_index().date.iloc[0]
        new = df[df["id"] == index].reset_index().date.iloc[-1]
        duration = (new - old).days
        if duration >= history:
            kept_accounts.append(index)
        else:
            rejected_accounts.append(index)
    return (kept_accounts, rejected_accounts)


def positive_transactions(x):
    """
    Returns only positive transactions, 
    and set 0 to negative ones. 
    
    """
    if x >= 0:
        return x
    else:
        return 0


def negative_transactions(x):

    """
    Returns only negative transactions, 
    and set 0 to positive ones.
    """
    if x <= 0:
        return x
    else:
        return 0


def preprocess(test_account, test_transactions, nb_month=6):

    """
    This function returns preprocessed data that will be used as input 
    to the ML models
    """

    # Convert dictionnaries to dataframes
    transactions = pd.DataFrame(map(dict, test_transactions))
    accounts = pd.DataFrame(map(dict, [test_account]))

    print(accounts)
    print(transactions)

    # Convert date columns to date types
    accounts["update_date"] = pd.to_datetime(accounts["update_date"])
    transactions["date"] = pd.to_datetime(transactions["date"])

    # Merge transactions and accounts dataframes
    df = pd.merge(accounts, transactions, how="inner", left_on="id", right_on="id")

    df = df.set_index("date")

    # Keep only accounts that have more than 180 days of history
    kept_accounts, rejected_accounts = history_accounts(df, history=180)
    df_kept_accounts = df[df["id"].isin(kept_accounts)]
    accounts_kept = accounts[accounts["id"].isin(kept_accounts)]

    # Create new features
    list_of_df = []

    for id in set(accounts_kept["id"]):

        # Transactions datframe df_id related to id
        df_id = df_kept_accounts[df_kept_accounts["id"] == id]

        # Create new columns "positive_transactions" and "negative_transactions"
        df_id["positive_transactions"] = df_id["amount"].apply(
            lambda x: positive_transactions(x)
        )
        df_id["negative_transactions"] = df_id["amount"].apply(
            lambda x: negative_transactions(x)
        )

        # Create "balance_per_day" column of balances for each date
        df_id = df_id.groupby("date").sum()
        df_id.sort_values(by="date", ascending=False, inplace=True)
        df_id["balance_per_day"] = -df_id["amount"]
        df_id["balance_per_day"].iloc[0] += accounts_kept[
            accounts_kept["id"] == id
        ].balance[0]
        df_id["balance_per_day"] = df_id["balance_per_day"].cumsum()

        df_id = df_id.drop("balance", axis=1)

        df_id.sort_values(by="date", ascending=True, inplace=True)

        # Normalize data so that I solve sparcity of dates (add missing dates in df_id)
        for i, date in enumerate(df_id.index[:-1]):

            days = (df_id.index[i + 1] - df_id.index[i]).days
            if days == 1:
                continue

            else:
                for j in range(1, days):
                    df_id.loc[dt.timedelta(days=j) + date] = df_id.loc[date]
                    df_id.loc[
                        dt.timedelta(days=j) + date,
                        ["amount", "positive_transactions", "negative_transactions"],
                    ] = [0, 0, 0]

        df_id.sort_values(by="date", ascending=True, inplace=True)

        df_id["sum_outgoing_next_30_days"] = None

        # For each day of df_id compute sum of outgoings and incomes for next month
        for i, date in enumerate(df_id.index[:-30]):
            df_id.loc[date, "sum_outgoing_next_30_days"] = df_id.iloc[i + 1 : i + 31][
                "negative_transactions"
            ].sum()
            df_id.loc[date, "sum_income_next_30_days"] = df_id.iloc[i + 1 : i + 31][
                "positive_transactions"
            ].sum()

        for i in range(nb_month):
            df_id["negative_transaction_last_" + str(i + 1) + "_month"] = df_id[
                "sum_outgoing_next_30_days"
            ].shift((i + 1) * 30)
            df_id["positive_transaction_last_" + str(i + 1) + "_month"] = df_id[
                "sum_income_next_30_days"
            ].shift((i + 1) * 30)

        length = len(df_id)
        df_id = df_id.reset_index()

        # Monthly resampling of df_id
        df_id = df_id.drop([i for i in range(length) if i % 30 != 0])

        list_of_df.append(df_id)

    df_final = pd.DataFrame()
    for i, d in enumerate(list_of_df):
        df_final = pd.concat([df_final, d], axis=0)

    return df_final


def prediction_model(df, model):

    """
    Returns prediction of df using model 
    
    """
    df_pred = pd.DataFrame(df.iloc[-1]).T

    X = df_pred.drop(
        [
            "id",
            "sum_outgoing_next_30_days",
            "date",
            "amount",
            "positive_transactions",
            "negative_transactions",
            "sum_income_next_30_days",
        ],
        axis=1,
    )
    X = X.astype("float32")

    prediction = model.predict(X)

    return prediction[0]
