import json
from datetime import date
#from bson import json_util
import requests





# class DateTimeEncoder(json.JSONEncoder):
#     def default(self, z):
#         if isinstance(z, date):
#             return (str(z))
#         else:
#             return super().default(z)

# You can use this function to test your api
# Make sure the uvicorn server is running locally on `http://127.0.0.1:8000/`
# or change the hardcoded localhost below
def test_predict():
    """
    Test the predict route with test data
    """
    test_account = {
			"id":1,
			"balance": 10000,
		        "update_date": str(date(2020, 11, 3))
		    }


    test_transactions = [
        {"id":1,
	"date": str(date(2020, i, j)), "amount": -100}
        for i in range(1, 10)
        for j in [5,17,26]
    ]

    test_data = {
        "account": test_account,
        "transactions": test_transactions,
    }

    print("Calling API with test data:")
    print(json.dumps(test_data))

    response = requests.post(
        "http://127.0.0.1:8000/predict", data=json.dumps(test_data)
    )

    print("Response: ")
  
    print(response.json())

    assert response.status_code == 200


if __name__ == "__main__":
    test_predict()
