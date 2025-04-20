import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Cyber Attack Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
local_css("style.css")

# Load or generate sample data
@st.cache_data
def load_data():
    # In a real application, you would load your actual dataset here
    # For demonstration, we'll create synthetic data
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic features
    data = {
        'ip_address': [f"192.168.{np.random.randint(0,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
        'request_count': np.random.poisson(50, n_samples),
        'request_size': np.random.exponential(1000, n_samples),
        'response_time': np.random.normal(200, 50, n_samples),
        'error_rate': np.random.uniform(0, 0.3, n_samples),
        'traffic_src': np.random.choice(['internal', 'external', 'partner'], n_samples),
        'protocol': np.random.choice(['HTTP', 'HTTPS', 'FTP', 'SSH'], n_samples),
        'time_since_last_request': np.random.exponential(60, n_samples),
        'auth_attempts': np.random.randint(0, 10, n_samples),
        'is_attack': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Add some patterns to make attacks somewhat predictable
    df.loc[(df['request_count'] > 80) & (df['error_rate'] > 0.2), 'is_attack'] = 1
    df.loc[(df['auth_attempts'] > 5) & (df['time_since_last_request'] < 10), 'is_attack'] = 1
    df.loc[(df['protocol'] == 'SSH') & (df['auth_attempts'] > 3), 'is_attack'] = 1
    
    return df

# Preprocess data
def preprocess_data(df):
    # Copy dataframe
    df_processed = df.copy()
    
    # Encode categorical features
    le = LabelEncoder()
    df_processed['traffic_src'] = le.fit_transform(df_processed['traffic_src'])
    df_processed['protocol'] = le.fit_transform(df_processed['protocol'])
    
    # Drop IP address (not useful for ML)
    df_processed = df_processed.drop('ip_address', axis=1)
    
    return df_processed

# Train model
def train_model(df, model_name='Random Forest'):
    X = df.drop('is_attack', axis=1)
    y = df['is_attack']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    if model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        model = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=sum(y==0)/sum(y==1))
    elif model_name == 'LightGBM':
        model = LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(iterations=100, random_state=42, silent=True)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, scaler, accuracy, roc_auc, report, cm, X_test_scaled, y_test, y_pred_proba

# Plot ROC curve
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    return fig

# Plot confusion matrix
def plot_confusion_matrix(cm):
    fig = px.imshow(cm, 
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Normal', 'Attack'],
                   y=['Normal', 'Attack'],
                   text_auto=True,
                   aspect="auto")
    fig.update_layout(title='Confusion Matrix')
    return fig

# Feature importance plot
def plot_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    feat_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feat_importance, x='Importance', y='Feature', orientation='h', 
                 title='Feature Importance')
    return fig

# Main app function
def main():
    # Load data
    df = load_data()
    df_processed = preprocess_data(df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Page", ["Dashboard", "Data Exploration", "Model Training", "Real-time Prediction"])
    
    # Dashboard
    if app_mode == "Dashboard":
        st.title("🛡️ AI Cyber Attack Prediction System")
        st.markdown("""
        This system uses machine learning to predict potential cyber attacks based on network traffic patterns.
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Attack Percentage", f"{df['is_attack'].mean()*100:.2f}%")
        col3.metric("Normal Traffic", f"{(1-df['is_attack'].mean())*100:.2f}%")
        
        # Attack distribution over time (simulated)
        st.subheader("Attack Pattern Distribution")
        attack_df = df[df['is_attack'] == 1].sample(frac=0.5, random_state=42)
        fig = px.scatter(attack_df, x='request_count', y='error_rate', 
                         color='protocol', size='auth_attempts',
                         title="Attack Patterns by Protocol",
                         hover_data=['traffic_src'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Protocol distribution
        st.subheader("Traffic Protocol Distribution")
        fig = px.pie(df, names='protocol', title='Traffic by Protocol')
        st.plotly_chart(fig, use_container_width=True)
        
    # Data Exploration
    elif app_mode == "Data Exploration":
        st.title("Data Exploration")
        
        # Show raw data
        if st.checkbox("Show Raw Data"):
            st.subheader("Raw Data")
            st.write(df)
        
        # Show processed data
        if st.checkbox("Show Processed Data"):
            st.subheader("Processed Data (for ML)")
            st.write(df_processed.head())
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select Feature to Visualize", df_processed.columns)
        
        if df_processed[feature].nunique() < 10:
            fig = px.histogram(df, x=feature, color='is_attack', 
                             title=f"Distribution of {feature} by Attack Status",
                             barmode='group')
        else:
            fig = px.box(df, x='is_attack', y=feature, 
                        title=f"Distribution of {feature} by Attack Status",
                        color='is_attack')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlation Matrix")
        corr = df_processed.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Training
    elif app_mode == "Model Training":
        st.title("Model Training")
        
        # Model selection
        model_name = st.selectbox("Select Model", 
                                ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost'])
        
        if st.button("Train Model"):
            with st.spinner(f'Training {model_name} Model...'):
                model, scaler, accuracy, roc_auc, report, cm, X_test, y_test, y_pred_proba = train_model(df_processed, model_name)
                
                # Save model and scaler
                joblib.dump(model, 'cyber_attack_model.pkl')
                joblib.dump(scaler, 'scaler.pkl')
                
                # Display results
                st.success("Model trained successfully!")
                
                # Metrics
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("ROC AUC Score", f"{roc_auc:.4f}")
                
                # Classification report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Blues'))
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)
                
                # ROC curve
                st.subheader("ROC Curve")
                st.plotly_chart(plot_roc_curve(y_test, y_pred_proba), use_container_width=True)
                
                # Feature importance
                st.subheader("Feature Importance")
                feature_importance_plot = plot_feature_importance(model, df_processed.drop('is_attack', axis=1).columns)
                if feature_importance_plot:
                    st.plotly_chart(feature_importance_plot, use_container_width=True)
                else:
                    st.warning("Feature importance not available for this model.")
    
    # Real-time Prediction
    elif app_mode == "Real-time Prediction":
        st.title("Real-time Attack Prediction")
        
        # Load model if exists
        try:
            model = joblib.load('cyber_attack_model.pkl')
            scaler = joblib.load('scaler.pkl')
            model_loaded = True
        except:
            st.warning("Please train a model first from the 'Model Training' page.")
            model_loaded = False
        
        if model_loaded:
            # Input form
            with st.form("prediction_form"):
                st.subheader("Enter Traffic Features")
                
                col1, col2 = st.columns(2)
                request_count = col1.number_input("Request Count", min_value=0, value=50)
                request_size = col2.number_input("Request Size (bytes)", min_value=0, value=1000)
                
                col1, col2 = st.columns(2)
                response_time = col1.number_input("Response Time (ms)", min_value=0, value=200)
                error_rate = col2.slider("Error Rate", 0.0, 1.0, 0.1)
                
                col1, col2 = st.columns(2)
                traffic_src = col1.selectbox("Traffic Source", ['internal', 'external', 'partner'])
                protocol = col2.selectbox("Protocol", ['HTTP', 'HTTPS', 'FTP', 'SSH'])
                
                col1, col2 = st.columns(2)
                time_since_last = col1.number_input("Time Since Last Request (s)", min_value=0, value=60)
                auth_attempts = col2.number_input("Authentication Attempts", min_value=0, value=1)
                
                submitted = st.form_submit_button("Predict")
                
                if submitted:
                    # Create input dataframe
                    input_data = pd.DataFrame({
                        'request_count': [request_count],
                        'request_size': [request_size],
                        'response_time': [response_time],
                        'error_rate': [error_rate],
                        'traffic_src': [traffic_src],
                        'protocol': [protocol],
                        'time_since_last_request': [time_since_last],
                        'auth_attempts': [auth_attempts]
                    })
                    
                    # Preprocess input
                    input_processed = preprocess_data(input_data)
                    input_scaled = scaler.transform(input_processed)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    prediction_proba = model.predict_proba(input_scaled)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    if prediction[0] == 1:
                        st.error(f"🚨 Attack Detected (Probability: {prediction_proba[0][1]:.2%})")
                        st.markdown("""
                        **Recommended Actions:**
                        - Isolate the affected system
                        - Investigate traffic logs
                        - Check for unusual patterns
                        - Notify security team
                        """)
                    else:
                        st.success(f"✅ Normal Traffic (Probability: {prediction_proba[0][0]:.2%})")
                    
                    # Show probabilities
                    st.subheader("Prediction Probabilities")
                    proba_df = pd.DataFrame({
                        'Class': ['Normal', 'Attack'],
                        'Probability': prediction_proba[0]
                    })
                    fig = px.bar(proba_df, x='Class', y='Probability', 
                                 color='Class', text='Probability',
                                 title="Attack Probability Distribution")
                    st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()