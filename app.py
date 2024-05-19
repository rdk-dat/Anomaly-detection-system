import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.utils import get_file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    path = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
    data = pd.read_csv(path, header=None)
    columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type"
    ]
    data.columns = columns
    data = data.dropna()
    data = data.drop_duplicates()

    categorical_columns = ["protocol_type", "service", "flag", "attack_type"]
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
    
    X = data.drop(["attack_type"], axis=1)
    y = data["attack_type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    for col in data.columns:
        if data[col].dtype == object and col != "attack_type":
            try:
                data[col] = data[col].astype(float)
            except ValueError:
                continue
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    random_forest_model = RandomForestClassifier(n_estimators=1000, random_state=42)
    random_forest_model.fit(X_train_scaled, y_train)
    rf_predictions = random_forest_model.predict(X_test_scaled)

    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_precision = precision_score(y_test, rf_predictions, average='macro', zero_division=1)
    rf_recall = recall_score(y_test, rf_predictions, average='macro')
    rf_f1 = f1_score(y_test, rf_predictions, average='macro')

    
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_predictions = svm_model.predict(X_test_scaled)

    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_precision = precision_score(y_test, svm_predictions, average='macro', zero_division=1)
    svm_recall = recall_score(y_test, svm_predictions, average='macro')
    svm_f1 = f1_score(y_test, svm_predictions, average='macro')



    metrics_data = {
        'Model': ['Random Forest', 'SVM'],
        'Accuracy': [rf_accuracy, svm_accuracy],
        'Precision': [rf_precision, svm_precision],
        'Recall': [rf_recall, svm_recall],
        'F1-Score': [rf_f1, svm_f1]
    }
    metrics_df = pd.DataFrame(metrics_data)

    metrics_html = metrics_df.to_html(classes='table table-striped')
    return render_template('index.html', tables=[metrics_html], titles=['na', 'Prediction Results'])

if __name__ == '__main__':
    app.run(debug=True)
