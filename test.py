import requests

url = "https://crack-classification-backend.onrender.com/"
files = {"file": open("test_img2.jpg", "rb")}

response = requests.post(url, files=files)

# Check the status code first
print(f"Status Code: {response.status_code}")

# If the status code is 200, then try to parse the response
if response.status_code == 200:
    print(response.json())
else:
    print("Error: Unable to get a valid response.")

