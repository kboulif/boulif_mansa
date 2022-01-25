from datetime import date
from typing import List
from pydantic import BaseModel
import joblib
from tools import history_accounts,positive_transactions,negative_transactions,preprocess,prediction_model

class Account(BaseModel):
    id : int
    update_date: date
    balance: float


class Transaction(BaseModel):
    id : int
    amount: float
    date: date


class ResponsePredict(BaseModel):
    id : int
    predicted_amount: float


MODEL:str="model.pkl"


class Model:

    def __init__(self,model_path:str = MODEL):

        self.model = joblib.load(model_path)
            
    def predict(
        self, transactions : List[Transaction], account: Account
    ):

        df_preprocessed = preprocess(account,transactions)
        
        prediction = prediction_model(df_preprocessed, self.model)

        return ResponsePredict(
        id=account.id,
        predicted_amount=min(prediction, 0)
        )
        
        
        
        

    
    
    
