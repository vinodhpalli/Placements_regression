import streamlit as st
import pandas as pd
import joblib

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Interview Success Predictor",
    page_icon="🎯",
    layout="wide"
)

# --------------------------
# LOAD MODEL
# --------------------------
model = joblib.load("model.pkl")

# --------------------------
# CUSTOM CSS
# --------------------------
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.title {
    text-align:center;
    color:#4F46E5;
    font-size:45px;
    font-weight:700;
}

.subtitle {
    text-align:center;
    color:#6b7280;
    font-size:20px;
    margin-bottom:20px;
}

.stButton > button {
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    background:linear-gradient(135deg,#4F46E5,#7C3AED);
    color:white;
    font-size:20px;
    font-weight:bold;
}

.result-card{
    background:linear-gradient(135deg,#4F46E5,#7C3AED);
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 10px 25px rgba(0,0,0,0.2);
}

.result-title{
    font-size:28px;
    font-weight:600;
}

.result-score{
    font-size:60px;
    font-weight:800;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER
# --------------------------
st.markdown(
    """
    <div class='title'>
        🎯 Interview Success Predictor
    </div>
    <div class='subtitle'>
        AI Powered Placement Readiness Assessment
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------
# INPUT SECTION
# --------------------------
col1, col2 = st.columns(2)

with col1:

    st.subheader("📚 Academic Information")

    cgpa = st.slider(
        "CGPA",
        0.0,
        10.0,
        7.5,
        0.1
    )

    aptitude = st.slider(
        "Aptitude Test Score",
        0,
        100,
        70
    )

    ssc_marks = st.slider(
        "SSC Marks (%)",
        0,
        100,
        75
    )

    hsc_marks = st.slider(
        "HSC Marks (%)",
        0,
        100,
        75
    )

    soft_skills = st.slider(
        "Soft Skills Rating",
        0.0,
        10.0,
        7.0,
        0.1
    )

with col2:

    st.subheader("💼 Experience & Activities")

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=20,
        value=2
    )

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=20,
        value=3
    )

    workshops = st.number_input(
        "Workshops / Certifications",
        min_value=0,
        max_value=20,
        value=2
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["No", "Yes"]
    )

    placement_training = st.selectbox(
        "Placement Training",
        ["No", "Yes"]
    )

st.divider()

# --------------------------
# PREDICTION
# --------------------------
if st.button("🚀 Predict Success Probability"):

    input_data = pd.DataFrame({
        "CGPA":[cgpa],
        "Internships":[internships],
        "Projects":[projects],
        "Workshops/Certifications":[workshops],
        "AptitudeTestScore":[aptitude],
        "SoftSkillsRating":[soft_skills],
        "SSC_Marks":[ssc_marks],
        "HSC_Marks":[hsc_marks],
        "ExtracurricularActivities_Yes":[1 if extracurricular=="Yes" else 0],
        "PlacementTraining_Yes":[1 if placement_training=="Yes" else 0]
    })

    try:

        prediction = model.predict(input_data)[0]

        prediction = max(0, min(100, prediction))

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    🎯 Interview Success Probability
                </div>
                <div class="result-score">
                    {prediction:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Placement Readiness")

        st.progress(int(prediction))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Success Probability",
                f"{prediction:.2f}%"
            )

        with col2:
            st.metric(
                "CGPA",
                cgpa
            )

        with col3:
            st.metric(
                "Projects",
                projects
            )

        if prediction >= 80:
            st.success(
                "Excellent profile. Very high probability of interview success."
            )

        elif prediction >= 60:
            st.info(
                "Good profile. Improve aptitude and soft skills for better opportunities."
            )

        else:
            st.warning(
                "Focus on projects, internships, certifications and aptitude preparation."
            )

    except Exception as e:

        st.error("Feature mismatch with trained model.")

        st.write("Error:")
        st.write(e)

        try:
            st.write("Expected Features:")
            st.write(model.feature_names_in_)
        except:
            pass