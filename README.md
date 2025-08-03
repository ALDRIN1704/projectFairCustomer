# projectFairCustomer

`fairCustomer` is a customer risk classification and visualization dashboard built with Python. It includes a machine learning pipeline for training a risk classifier, clustering customer data, and displaying insights through a web interface.

## Features

- Risk classifier model using customer data.
- K-means clustering of customer risk profiles.
- Interactive web dashboard to view customer risk and distribution plots.
- Authentication system for user login and signup.

## Project Structure

- `app.py`: Main Flask application.
- `cluster_customer_risk.py`: Clusters customers based on risk.
- `train_risk_classifier.py`: Trains the risk prediction model.
- `database.db`: SQLite database for storing user data.
- `data/`: Contains input CSV data (`Accounts-Receivable.csv`).
- `models/`: Contains trained model and label encoder files.
- `templates/`: HTML files for web interface.
- `static/`: Static resources like CSS and generated plots.

## Setup Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python fairCustomer/train_risk_classifier.py
   ```

3. **Cluster customer risk**
   ```bash
   python fairCustomer/cluster_customer_risk.py
   ```

4. **Run the app**
   ```bash
   python fairCustomer/app.py
   ```

5. **Open your browser**
   Navigate to `http://127.0.0.1:5000` to access the dashboard.

## Screenshot
<img width="1920" height="1080" alt="Screenshot 2025-08-03 235000" src="https://github.com/user-attachments/assets/d002f899-54d0-41e9-aad8-ae21a580aa4d" />


## Requirements

- Python 3.7+
- Flask
- Scikit-learn
- Pandas
- Matplotlib

## License

MIT License

