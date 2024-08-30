from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Define the neural network architecture
class ImprovedNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize global variables for scaler and model
scaler = None
model = None

def train_model(data_file):
    global scaler, model

    try:
        # Load the dataset from the uploaded file
        data = pd.read_csv(data_file)

        # Define features and target variables
        features = [
            'chemical_composition', 'casting_temp', 'cooling_water_temp',
            'casting_speed', 'entry_temp', 'emulsion_temp', 'emulsion_pressure',
            'emulsion_concentration', 'quench_pressure'
        ]
        targets = ['UTS', 'elongation', 'conductivity']

        # Extract features and targets
        X = data[features]
        y = data[targets]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        # Initialize the model
        model = ImprovedNN(input_dim=len(features), output_dim=len(targets))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        loss_fn = nn.MSELoss()

        # Train the model
        epochs = 200
        losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = loss_fn(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if (epoch+1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        return True
    except Exception as e:
        print(f'Error in training model: {e}')
        return False

def assess_quality(predictions, targets):
    quality_labels = []
    ranges = {
        'UTS': (100, 200),
        'elongation': (10, 30),
        'conductivity': (30, 60)
    }

    for pred in predictions:
        pred_quality = 'good'
        pred_results = {}
        for i, target in enumerate(targets):
            min_val, max_val = ranges[target]
            pred_val = pred[i]
            pred_results[target] = pred_val
            if pred_val < min_val:
                pred_quality = 'bad'
            elif pred_val > max_val and pred_quality != 'bad':
                pred_quality = 'better'

        quality_labels.append((pred_results, pred_quality))

    return quality_labels

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global scaler, model

    if request.method == 'POST':
        try:
            # Handle file upload and input data
            file = request.files['file_upload']
            file_path = 'uploaded_data.csv'
            file.save(file_path)
            if not train_model(file_path):
                flash('Model training failed. Please check the data file and try again.')
                return redirect(url_for('home'))

            # Extract and validate form data
            form_data = {
                'chemical_composition': float(request.form['chemical_composition']),
                'casting_temp': float(request.form['casting_temp']),
                'cooling_water_temp': float(request.form['cooling_water_temp']),
                'casting_speed': float(request.form['casting_speed']),
                'entry_temp': float(request.form['entry_temp']),
                'emulsion_temp': float(request.form['emulsion_temp']),
                'emulsion_pressure': float(request.form['emulsion_pressure']),
                'emulsion_concentration': float(request.form['emulsion_concentration']),
                'quench_pressure': float(request.form['quench_pressure'])
            }

            # Prepare data for prediction
            input_data = [list(form_data.values())]
            input_scaled = scaler.transform(input_data)

            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                prediction = model(input_tensor).numpy()[0]

            UTS, elongation, conductivity = prediction

            # Assess quality of the prediction
            quality = assess_quality([prediction], ['UTS', 'elongation', 'conductivity'])[0][1]

            # Redirect to prediction page with results
            return redirect(url_for('prediction', UTS=UTS, elongation=elongation, conductivity=conductivity, quality=quality))

        except ValueError as e:
            flash('Invalid input data. Please make sure all fields are filled in correctly.')
        except Exception as e:
            flash(f'An error occurred: {e}')

    return render_template('upload.html')

@app.route('/prediction')
def prediction():
    # Get the prediction results from the query parameters
    UTS = request.args.get('UTS')
    elongation = request.args.get('elongation')
    conductivity = request.args.get('conductivity')
    quality = request.args.get('quality')
    return render_template('prediction.html', UTS=UTS, elongation=elongation, conductivity=conductivity, quality=quality)

@app.route('/data')
def data():
    # Load the dataset to display
    try:
        data = pd.read_csv('uploaded_data.csv')
        # Convert the DataFrame to HTML for rendering
        data_html = data.to_html(classes='table table-striped', index=False)
    except FileNotFoundError:
        flash('No data file found. Please upload a file first.')
        return redirect(url_for('home'))
    except Exception as e:
        flash(f'Error loading data: {e}')
        return redirect(url_for('home'))

    return render_template('data.html', table=data_html)

if __name__ == '__main__':
    app.run(debug=True)
