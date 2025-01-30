from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Read feature names from feature_names.txt
with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

@app.route('/')
def home():
    # Pass feature names to the template
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect all form inputs dynamically
    form_inputs = pd.DataFrame(request.form.to_dict(), index=[0])
    
    # Ensure all inputs are numeric
    features = form_inputs.astype(float)
    
    # Make the prediction
    prediction = model.predict(features)
    
    # Return the prediction result
    return f'The predicted class is: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)
