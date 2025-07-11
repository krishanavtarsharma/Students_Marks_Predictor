import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import base64
import datetime

# -----------------------
# 🎯 Page Setup
# -----------------------
st.set_page_config(page_title="💡 AI-Powered Student Predictor", layout="wide")
st.title("📚 AI-Powered Student Performance Dashboard")
st.markdown("Predict, Analyze, and Improve Student Outcomes with Smart Insights")

# -----------------------
# 🔧 Utility Functions
# -----------------------
def add_download_button(df, filename):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="{filename}">📅 Download CSV</a>', unsafe_allow_html=True)

def suggest_college(score):
    if score >= 90:
        return "🎓 Tier 1 College (IITs, NITs, IIITs)"
    elif score >= 75:
        return "🏫 Tier 2 College (State/Private Universities)"
    elif score >= 60:
        return "🏢 Tier 3 College (General Admission Colleges)"
    else:
        return "🔁 Suggest Improvement & Reattempt"

def score_to_gpa(score):
    if score >= 90: return 10
    elif score >= 80: return 9
    elif score >= 70: return 8
    elif score >= 60: return 7
    elif score >= 50: return 6
    elif score >= 40: return 5
    else: return 0

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv("students_data.csv")
    except:
        sample_csv = """
StudentName,StudyHours,SleepHours,Attendance,AssignmentsCompleted,InternetUsage,FinalScore
John Doe,6,7,85,90,3,82
Jane Smith,8,6,92,100,2,95
Rahul Kumar,4,5,70,60,6,55
Anjali Mehra,5.5,8,78,80,3.5,75
Arjun Rathi,3,6.5,60,50,7,45
"""
        from io import StringIO
        return pd.read_csv(StringIO(sample_csv))

# -----------------------
# 📁 Data Handling
# -----------------------
st.sidebar.header("📂 Dataset Options")
use_sample = st.sidebar.checkbox("Use Sample Dataset", True)

if use_sample:
    df = load_sample_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV with StudentName, StudyHours, SleepHours, Attendance, AssignmentsCompleted, InternetUsage (hrs)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# -----------------------
# 🔄 Dataset Editor
# -----------------------
with st.sidebar.expander("📝 Edit Dataset"):
    df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
    add_download_button(df, "modified_dataset.csv")

# ➕ Add New Student Entry
with st.sidebar.expander("➕ Add New Student"):
    with st.form("add_form"):
        name = st.text_input("Student Name")
        study = st.slider("Study Hours", 1.0, 12.0, 6.0)
        sleep = st.slider("Sleep Hours", 3.0, 10.0, 7.0)
        att = st.slider("Attendance (%)", 50, 100, 80)
        assign = st.slider("Assignments Completed (%)", 0, 100, 80)
        net = st.slider("Internet Usage (hrs/day)", 0.0, 10.0, 3.0)
        final = st.slider("Final Score (%)", 0, 100, 75)
        submitted = st.form_submit_button("Add Student")
        if submitted:
            new_row = pd.DataFrame([[name, study, sleep, att, assign, net, final]],
                                   columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            st.success(f"✅ Added {name} to dataset")

# 🔍 Search Student
with st.sidebar.expander("🔍 Search Student"):
    query = st.text_input("Enter Student Name").strip().lower()
    if query:
        result_df = df[df["StudentName"].str.lower().str.contains(query)]
        st.write("🔎 Results:")
        st.dataframe(result_df)

# 📊 GPA Calculation
df['GPA'] = df['FinalScore'].apply(score_to_gpa)

# 📊 Summary
with st.sidebar.expander("📊 Dataset Summary"):
    st.metric("Total Students", len(df))
    st.metric("Average Score", f"{df['FinalScore'].mean():.2f}")
    st.metric("Score Range", f"{df['FinalScore'].min()} - {df['FinalScore'].max()}")

# -----------------------
# 🧐 Model Training
# -----------------------
features = ['StudyHours', 'SleepHours', 'Attendance', 'AssignmentsCompleted', 'InternetUsage']
model = LinearRegression()
X = df[features]
y = df['FinalScore'] if 'FinalScore' in df else None

if y is not None:
    model.fit(X, y)
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, 100)  # ✅ Force predictions between 0 and 100
    metrics = {
        'R² Score': r2_score(y, y_pred),
        'MAE': mean_absolute_error(y, y_pred),
        'MSE': mean_squared_error(y, y_pred)
    }

# -----------------------
# 🚀 Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & Metrics", "🎯 Single Prediction", "📁 Batch Prediction", "📈 Visualize", "🧪 Performance Report", "🎓 College Suggestions"
])

# 📊 Tab 1: Data & Metrics
with tab1:
    st.subheader("📋 Current Dataset")
    st.dataframe(df, use_container_width=True)
    if y is not None:
        st.markdown("### 📉 Model Metrics")
        st.metric("R² Score", f"{metrics['R² Score']:.2f}")
        st.metric("MAE", f"{metrics['MAE']:.2f}")
        st.metric("MSE", f"{metrics['MSE']:.2f}")

# 🎯 Tab 2: Single Prediction
with tab2:
    st.subheader("🎯 Predict Individual Student Performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        study = st.slider("📚 Study Hours", 1.0, 12.0, 6.0)
        assignments = st.slider("🗑️ Assignments Completed (%)", 0, 100, 80)
    with c2:
        sleep = st.slider("💤 Sleep Hours", 3.0, 10.0, 6.5)
        internet = st.slider("🌐 Internet Usage (hrs/day)", 0.0, 10.0, 3.0)
    with c3:
        attendance = st.slider("🏫 Attendance (%)", 50, 100, 80)
        student_name = st.text_input("👤 Student Name", value="John Doe")

    input_array = np.array([[study, sleep, attendance, assignments, internet]])
    raw_pred = model.predict(input_array)[0]
    pred = max(0, min(100, raw_pred))
    st.success(f"📚 Predicted Final Score for {student_name}: {pred:.2f} / 100")
    st.info(f"🎓 Suggested College: {suggest_college(pred)}")
    st.info(f"📊 Estimated GPA: {score_to_gpa(pred)}")
    if raw_pred > 100:
        st.warning("⚠️ Predicted score capped to 100%")

# 📁 Tab 3: Batch Prediction
with tab3:
    st.subheader("📁 Predict Scores from File")
    batch_file = st.file_uploader("Upload file with StudentName, StudyHours, SleepHours, Attendance, AssignmentsCompleted, InternetUsage", type=["csv"], key="batch")
    if batch_file:
        batch_df = pd.read_csv(batch_file)
        batch_pred = model.predict(batch_df[features])
        batch_df['PredictedScore'] = np.clip(batch_pred, 0, 100)
        batch_df['SuggestedCollege'] = batch_df['PredictedScore'].apply(suggest_college)
        batch_df['GPA'] = batch_df['PredictedScore'].apply(score_to_gpa)
        st.dataframe(batch_df)
        add_download_button(batch_df, "batch_predictions.csv")

# 📈 Tab 4: Visualization
with tab4:
    st.subheader("📈 Data Visualization")
    st.markdown("Customize filters to explore performance patterns:")
    att_min = st.slider("Minimum Attendance", 50, 100, 70)
    study_min = st.slider("Minimum Study Hours", 1, 10, 4)
    filtered = df[(df['Attendance'] >= att_min) & (df['StudyHours'] >= study_min)]

    if not filtered.empty:
        fig = px.scatter(filtered, x='StudyHours', y='FinalScore', color='Attendance',
                         size='SleepHours', title="Study Hours vs Score (Filtered)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data matching filter criteria")

# 🧪 Tab 5: Performance Report Generator
with tab5:
    st.subheader("🧪 Generate Custom Performance Report")
    if y is not None:
        low_score_df = df[df['FinalScore'] < 60]
        high_score_df = df[df['FinalScore'] >= 90]

        st.markdown("#### ⚠️ Students Below 60%")
        st.dataframe(low_score_df, use_container_width=True)
        st.markdown("#### 🏆 Top Performers (Above 90%)")
        st.dataframe(high_score_df, use_container_width=True)

        report_df = pd.concat([
            low_score_df.assign(Flag="Low Performer"),
            high_score_df.assign(Flag="Top Performer")
        ])
        add_download_button(report_df, "performance_report.csv")

# 🎓 Tab 6: College Suggestion Summary
with tab6:
    st.subheader("🎓 College Suggestion Summary")
    if 'FinalScore' in df:
        df['SuggestedCollege'] = df['FinalScore'].apply(suggest_college)
        college_counts = df['SuggestedCollege'].value_counts()
        st.bar_chart(college_counts)
        st.dataframe(df[['StudentName', 'StudyHours', 'SleepHours', 'Attendance',
                         'AssignmentsCompleted', 'InternetUsage', 'FinalScore', 'GPA', 'SuggestedCollege']])
        add_download_button(df, "college_suggestions.csv")

# 🖚 Footer
st.markdown("---")
st.markdown(f"App generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("Crafted with ❤️ by AI using Streamlit & Scikit-Learn")
