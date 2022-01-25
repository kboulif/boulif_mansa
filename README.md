# Exercice hiring MANSA


# Predict next month outgoing

The aim of this project is to predict next month outgoing given past 6 months of 
bank accounts' transactions, and serve the model using FastAPI.


## Author

- [Kaoutar Boulif](https://github.com/kboulif)


## Project Structure

```
boulif_mansa
│
│   README.md
│   requirements.txt    
│
└───notebooks
│      Data_preprocessing_and_visualizations.ipynb
│      Final_dataset_construction.ipynb
│      Machine_learning_models.ipynb
│   
└───data
│      accounts.csv
│      transactions.csv
│      accounts_transactions.csv
│      no_duplicates_data.csv
│      no_duplicates_accounts.csv
│      no_duplicates_transactions.csv
│      df_final_monthly.csv    
│          
└───api
│      main.py
│      model.py
│      main.py
│      model.pkl
│        
└───tests
│      test_main.py
│ 
└────────────

```
## Python version

- I used python version 3.9.7

## Notebooks

1 . Data_preprocessing_and_visualizations.ipynb :

In this notebook, I did the following tasks:

- Remove ouliers from "accounts" and "transactions" dataframes, which are datas that are not in the interval
[mean - c * std , mean + c * std ] where mean and std are the mean and standard deviation of 
the concerned column respectively. 

- Merge transactions and accounts dataframes and save the merged data as 
"accounts_transactions.csv". 

- Perform some preprocessing on "accounts_transactions.csv" like checking missing values,
check if we have at least one transaction, check if the update_date is consistent
with transaction dates, filter data by history days number and keep only accounts ids 
which have more than 180 days of history and remove duplicate transactions.

- Visualize some accounts' transactions and other plots. 



2 . Final_dataset_construction.ipynb :

The aim of this notebook is to build new dataframe which will be the input of
our training models:

- The data is composed of daily transactions for each account. We add new lines of transactions
with values of amount equal to 0, in order to solve sparcity of transactions. 
The columns are date of transactions, amount of transaction, positive and negative transactions, 
balance of account, negative transactions for last one to six months , 
positive transactions for last one to six months, sum of outgoing for next month
and sum of income for next month. The rows are then resampled by moths to have df_final_monthly
transactions.







3 . Machine_learning_models.ipynb :

In this notebook, we do the following tasks:

- Remove again outliers using the IQR rule, because as we created new data, 
we had some monthly transactions which are outliers compared to other transactions 
in the data.

- Define metrics that we will use to evaluate and select the best model. The 
metrics used are MAE and MAPE.

- Split data to train and test (80% train against 20% test).

- Show correlations between target variable and independant variables.

- Modeling section


## Install necessary packages


```bash
pip install -r requirements.txt
``` 
   
## FastAPI

0 . Run the main file inside "api" folder by calling :

```bash
uvicorn main:app 
```

This should start the local server and you should be able to see 

the automatically generated API docs at 

```
bash http://127.0.0.1:8000/docs

```

1 . Once you run the API, you can test it by opening another terminal 

window in the "tests" folder and calling:

```bash
python -m test_main
```







