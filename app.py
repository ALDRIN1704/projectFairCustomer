from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import joblib
import sqlite3
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'

# File paths
db_path = "database.db"
data_path = "data/Accounts-Receivable.csv"
clustered_data_path = "outputs/customer_risk_clusters.csv"

# Load model and encoded data
df = pd.read_csv(clustered_data_path)
classifier = joblib.load("models/risk_classifier.pkl")
label_encoder = joblib.load("models/risk_label_encoder.pkl")

# Initialize SQLite DB for users
def init_user_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_user_db()

# ----------------- Signup -----------------
@app.route('/signup', methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            error = "Passwords do not match."
        else:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                               (username, email, password))
                conn.commit()
                conn.close()
                return redirect("/login")
            except sqlite3.IntegrityError:
                error = "Email already exists."
            except Exception as e:
                error = str(e)
    return render_template("auth/signup.html", error=error)




@app.route('/')

def welcome():
    return render_template("welcome.html")


# ----------------- Login -----------------
@app.route('/login', methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE (username = ? OR email = ?) AND password = ?",
                       (username, username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['username'] = user[1]  # store username in session
            return redirect(url_for('dashboard'))  # redirect to dashboard
        else:
            error = "Invalid credentials."
    return render_template("auth/login.html", error=error)
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ----------------- Dashboard -----------------
@app.route('/dashboard')
def dashboard():
    risk_filter = request.args.get('risk')
    filtered_df = df[df['risk_level'] == risk_filter] if risk_filter else df

    return render_template("dashboard.html",
                           tables=[filtered_df.to_html(classes='data', index=False)],
                           titles=filtered_df.columns.values,
                           risk_levels=df['risk_level'].unique(),
                           selected_risk=risk_filter)

# ----------------- Prediction by Customer ID -----------------
@app.route('/predict', methods=["GET", "POST"])
def predict():
    prediction = None
    customer_id_input = None
    chart_path = None

    if request.method == "POST":
        customer_id_input = request.form['customer_id'].strip()

        try:
            customer_row = df[df['customerID'] == customer_id_input]

            if customer_row.empty:
                prediction = f"❌ Customer ID '{customer_id_input}' not found."
            else:
                features = customer_row[[
                    'total_invoices', 'avg_days_late', 'max_days_late',
                    'total_days_late', 'avg_invoice_amount', 'total_disputes'
                ]].values

                pred = classifier.predict(features)[0]
                prediction = label_encoder.inverse_transform([pred])[0]

                # Plotting historical behavior from original dataset
                raw_df = pd.read_csv(data_path)
                history = raw_df[raw_df['customerID'] == customer_id_input]
                history_sorted = history.sort_values(by='InvoiceDate')

                plt.figure(figsize=(8, 4))
                plt.plot(history_sorted['InvoiceDate'], history_sorted['DaysLate'], marker='o')
                plt.xticks(rotation=45)
                plt.title(f"Days Late Over Time: {customer_id_input}")
                plt.xlabel("Invoice Date")
                plt.ylabel("Days Late")
                plt.tight_layout()

                os.makedirs("static/customer_charts", exist_ok=True)
                chart_path = f"static/customer_charts/{customer_id_input}.png"
                plt.savefig(chart_path)
                plt.close()

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("predict.html", prediction=prediction,
                           customer_id=customer_id_input, chart_path=chart_path)

# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(debug=True)
