import requests 
import subprocess


# token = subprocess.getoutput('gcloud auth print-identity-token')
# headers = {'Authorization': f'Bearer {token}'}
resp = requests.post("https://prediction-ppgfuuhtkq-ts.a.run.app", files={"file": "data/car_description.wav"})

print("Status Code:", resp.status_code)
print("Response Content:", resp.text)

# Parse JSON only if response is successful
if resp.status_code == 200:
    print(resp.json())
else:
    print("Failed to get a valid response")
