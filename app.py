import requests
import json
import joblib

app = Flask(__name__)
joblib.dump(model, 'multi_linear_regression_model.joblib')
# Example data for prediction
# This should match the features used for training, including 'ocean_proximity'
# 'total_bedrooms' can be optionally missing, as the API handles it.
example_data = {
    "longitude": -122.22,
    "latitude": 37.86,
    "housing_median_age": 21.0,
    "total_rooms": 7099.0,
    "total_bedrooms": None, # Example of missing value
    "population": 2401.0,
    "households": 1138.0,
    "median_income": 8.3014,
    "ocean_proximity": "NEAR BAY"
}

# Define the API endpoint URL (assuming Flask app is running locally on port 5000)
url = 'http://127.0.0.1:5000/predict' # Update with actual deployed URL if not local

headers = {'Content-Type': 'application/json'}

try:
    # Send the POST request
    response = requests.post(url, data=json.dumps(example_data), headers=headers)
    response.raise_for_status() # Raise an exception for bad status codes

    # Print the JSON response
    print("API Response:", response.json())
except requests.exceptions.ConnectionError:
    print(f"Connection Error: Could not connect to the API at {url}. Make sure the Flask app is running.")
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}. Response: {response.text}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("\nTo run the Flask API, execute the following command in a separate terminal or Colab cell:")
print("!python -c 'from your_script_name import app; app.run(host=\'0.0.0.0\', port=5000, debug=False)'")
print("If running in Colab, consider using ngrok to expose the server for external access.")

if __name__** "__main__":
  app.run(debug=True)
