import streamlit as st
import requests
import json

st.set_page_config(page_title="XAI Threat Intelligence", layout="wide")

st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #38bdf8;
        text-align: center;
    }
    .card {
        padding: 15px;
        border-radius: 10px;
        background-color: #0f172a;
        color: white;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">XAI Threat Intelligence System</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Original Image", use_column_width=True)

    if st.button("Analyze"):
        try:
            response = requests.post(
                "http://api:8000/predict-xai",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }
            )

            data = response.json()

            if "prediction" not in data:
                st.error(f"API Error: {data}")
                st.stop()

            label = data["prediction"]["label"]
            confidence = data["prediction"]["confidence"]
            threat = data["threat_level"]
            score = data["threat_score"]
            trust = data["trust_level"]
            consistency = data["consistency"]

            st.markdown(f"## 🔍 {label}")

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Confidence", f"{confidence:.4f}")
            col2.metric("Threat", threat)
            col3.metric("Threat Score", f"{score:.4f}")
            col4.metric("Trust", trust)
            col5.metric("Consistency", f"{consistency:.4f}")

            st.markdown("### AI Explanation")
            st.info(data["explanation"])

            st.markdown("### Explainability Maps")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    "data:image/jpeg;base64," + data["xai"]["gradcam_pp"],
                    caption="GradCAM++"
                )

            with col2:
                st.image(
                    "data:image/jpeg;base64," + data["xai"]["scorecam"],
                    caption="ScoreCAM"
                )

            with col3:
                st.image(
                    "data:image/jpeg;base64," + data["xai"]["integrated_gradients"],
                    caption="Integrated Gradients"
                )

            st.markdown("### Attention Analysis")

            f1, f2, f3 = st.columns(3)

            f1.metric("GradCAM++ Focus", f"{data['focus_scores']['gradcam_pp']:.4f}")
            f2.metric("ScoreCAM Focus", f"{data['focus_scores']['scorecam']:.4f}")
            f3.metric("IG Focus", f"{data['focus_scores']['integrated_gradients']:.4f}")

            report = {
                "label": label,
                "confidence": confidence,
                "threat_score": score,
                "threat_level": threat,
                "trust": trust,
                "consistency": consistency,
                "uncertainty": data.get("uncertainty", "N/A"),
                "explanation": data["explanation"]
            }

            st.markdown("### 📥 Download Report")

            st.download_button(
                label="Download JSON Report",
                data=json.dumps(report, indent=4),
                file_name="xai_report.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
