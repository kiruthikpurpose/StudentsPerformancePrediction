import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

@st.cache_data
def load_data(file_path='StudentsPerformanceDataset.xlsx'):
    if not os.path.exists(file_path):
        return None, None, None, None, None, None
    df = pd.read_excel(file_path)
    categorical_features = ['Gender','Parental_Education','Internet_Access','Tutoring_Classes','Sports_Activity','Extra_Curricular','School_Type','Teacher_Feedback']
    df_processed = df.copy()
    encoders = {}
    for col in categorical_features:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f'{col}_Encoded'] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = le
    exclude_cols = ['Student_ID','Final_Score'] + [c for c in categorical_features if c in df_processed.columns]
    feature_cols = [col for col in df_processed.columns if col not in exclude_cols and col != 'Final_Score']
    X = df_processed[feature_cols]
    y = df_processed['Final_Score']
    return df, df_processed, X, y, feature_cols, encoders

@st.cache_resource
def train_model(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    selector = SelectKBest(score_func=f_regression, k=min(15, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    selected_features = X.columns[selector.get_support()]
    model = LinearRegression().fit(X_train_sel, y_train)
    y_pred_test = model.predict(X_test_sel)
    metrics_df = pd.DataFrame({
        'R² Score':[r2_score(y_test, y_pred_test)],
        'RMSE':[np.sqrt(mean_squared_error(y_test, y_pred_test))],
        'MAE':[mean_absolute_error(y_test, y_pred_test)],
        'MAPE':[np.mean(np.abs((y_test - y_pred_test) / y_test))*100 if (y_test>0).any() else np.nan]
    }, index=['Test'])
    feature_importance = pd.DataFrame({'Feature':selected_features,'Coefficient':model.coef_}).set_index('Feature')
    return model, scaler, selector, metrics_df, feature_importance, selected_features, y_train.mean(), X_train.mean()

df_raw, df_processed, X, y, feature_cols, encoders = load_data()

if df_raw is None:
    st.error("Dataset not found. Place 'StudentsPerformanceDataset.xlsx' next to this app.")
else:
    model, scaler, selector, metrics_df, feature_importance, selected_features, y_train_mean, X_train_mean = train_model(X, y, feature_cols)
    page = st.sidebar.radio("Navigation", ("Predict","Data Analysis","Model Performance"))

    if page == "Predict":
        st.title("Live Predictor")
        defaults = X_train_mean.to_dict()
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                Previous_Sem_Score = st.slider("Previous Semester Score", 0.0, 100.0, float(defaults.get('Previous_Sem_Score', 65.0)))
                Study_Hours_per_Week = st.slider("Study Hours per Week", 0.0, 80.0, float(defaults.get('Study_Hours_per_Week', 20.0)))
                Attendance_Percentage = st.slider("Attendance Percentage", 0.0, 100.0, float(defaults.get('Attendance_Percentage', 80.0)))
                Library_Usage_per_Week = st.slider("Library Usage (hrs/week)", 0.0, 40.0, float(defaults.get('Library_Usage_per_Week', 4.0)))
            with col2:
                Family_Income = st.number_input("Family Income", value=float(defaults.get('Family_Income', 50000.0)))
                Sleep_Hours = st.slider("Sleep Hours", 0.0, 12.0, float(defaults.get('Sleep_Hours', 7.0)))
                Travel_Time = st.slider("Travel Time (hrs)", 0.0, 10.0, float(defaults.get('Travel_Time', 1.5)))
                Peer_Influence = st.slider("Peer Influence (1-10)", 1.0, 10.0, float(defaults.get('Peer_Influence', 5.0)))
            Gender_val = None
            if 'Gender' in encoders:
                Gender_val = st.selectbox("Gender", options=list(encoders['Gender'].classes_))
            Parental_val = None
            if 'Parental_Education' in encoders:
                Parental_val = st.selectbox("Parental Education", options=list(encoders['Parental_Education'].classes_))
            Internet_val = None
            if 'Internet_Access' in encoders:
                Internet_val = st.selectbox("Internet Access?", options=list(encoders['Internet_Access'].classes_))
            Tutoring_val = None
            if 'Tutoring_Classes' in encoders:
                Tutoring_val = st.selectbox("Tutoring Classes?", options=list(encoders['Tutoring_Classes'].classes_))
            Sports_val = None
            if 'Sports_Activity' in encoders:
                Sports_val = st.selectbox("Sports Activity?", options=list(encoders['Sports_Activity'].classes_))
            Extra_val = None
            if 'Extra_Curricular' in encoders:
                Extra_val = st.selectbox("Extra-Curricular?", options=list(encoders['Extra_Curricular'].classes_))
            School_val = None
            if 'School_Type' in encoders:
                School_val = st.selectbox("School Type", options=list(encoders['School_Type'].classes_))
            Teacher_val = None
            if 'Teacher_Feedback' in encoders:
                Teacher_val = st.selectbox("Teacher Feedback", options=list(encoders['Teacher_Feedback'].classes_))
            submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = defaults.copy()
            input_data.update({
                'Previous_Sem_Score': Previous_Sem_Score,
                'Study_Hours_per_Week': Study_Hours_per_Week,
                'Attendance_Percentage': Attendance_Percentage,
                'Library_Usage_per_Week': Library_Usage_per_Week,
                'Family_Income': Family_Income,
                'Sleep_Hours': Sleep_Hours,
                'Travel_Time': Travel_Time,
                'Peer_Influence': Peer_Influence
            })
            if Gender_val is not None:
                try:
                    input_data['Gender_Encoded'] = int(encoders['Gender'].transform([Gender_val])[0])
                except Exception:
                    input_data['Gender_Encoded'] = int(defaults.get('Gender_Encoded', 0))
            if Parental_val is not None:
                try:
                    input_data['Parental_Education_Encoded'] = int(encoders['Parental_Education'].transform([Parental_val])[0])
                except Exception:
                    input_data['Parental_Education_Encoded'] = int(defaults.get('Parental_Education_Encoded', 0))
            if Internet_val is not None:
                try:
                    input_data['Internet_Access_Encoded'] = int(encoders['Internet_Access'].transform([Internet_val])[0])
                except Exception:
                    input_data['Internet_Access_Encoded'] = int(defaults.get('Internet_Access_Encoded', 0))
            if Tutoring_val is not None:
                try:
                    input_data['Tutoring_Classes_Encoded'] = int(encoders['Tutoring_Classes'].transform([Tutoring_val])[0])
                except Exception:
                    input_data['Tutoring_Classes_Encoded'] = int(defaults.get('Tutoring_Classes_Encoded', 0))
            if Sports_val is not None:
                try:
                    input_data['Sports_Activity_Encoded'] = int(encoders['Sports_Activity'].transform([Sports_val])[0])
                except Exception:
                    input_data['Sports_Activity_Encoded'] = int(defaults.get('Sports_Activity_Encoded', 0))
            if Extra_val is not None:
                try:
                    input_data['Extra_Curricular_Encoded'] = int(encoders['Extra_Curricular'].transform([Extra_val])[0])
                except Exception:
                    input_data['Extra_Curricular_Encoded'] = int(defaults.get('Extra_Curricular_Encoded', 0))
            if School_val is not None:
                try:
                    input_data['School_Type_Encoded'] = int(encoders['School_Type'].transform([School_val])[0])
                except Exception:
                    input_data['School_Type_Encoded'] = int(defaults.get('School_Type_Encoded', 0))
            if Teacher_val is not None:
                try:
                    input_data['Teacher_Feedback_Encoded'] = int(encoders['Teacher_Feedback'].transform([Teacher_val])[0])
                except Exception:
                    input_data['Teacher_Feedback_Encoded'] = int(defaults.get('Teacher_Feedback_Encoded', 0))

            ordered_input = {col: input_data.get(col, defaults.get(col, 0.0)) for col in feature_cols}
            input_df = pd.DataFrame([ordered_input])
            scaled_input = scaler.transform(input_df)
            selected_input = selector.transform(scaled_input)
            prediction = float(model.predict(selected_input)[0])
            diff = prediction - y_train_mean
            st.metric("Predicted Final Score", f"{prediction:.2f} / 100")
            st.metric("Difference from Avg", f"{diff:+.2f} pts")

    elif page == "Data Analysis":
        st.title("Data Analysis")
        corr = df_processed[feature_cols + ['Final_Score']].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, cmap="viridis", ax=ax)
        st.pyplot(fig)

    elif page == "Model Performance":
        st.title("Model Performance")
        st.dataframe(metrics_df)
        fi_plot = feature_importance.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=fi_plot['Coefficient'], y=fi_plot.index, orientation='h'))
        st.plotly_chart(fig, use_container_width=True)
