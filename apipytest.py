import requests

url = "http://127.0.0.1:5000/predict"
payload = {"uid": '48800000000000'}  # Replace with a valid UID from your dataset

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
