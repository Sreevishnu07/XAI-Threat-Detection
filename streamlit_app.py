import streamlit as st
import requests

st.title("XAI Threat Intelligence System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Original", use_column_width=True)

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

            st.subheader(data["prediction"]["label"])
            st.write("Confidence:", data["prediction"]["confidence"])
            st.write("Threat Level:", data["threat_level"])
            st.write("Threat Score:", data["threat_score"])
            st.write("Trust:", data["trust_level"])
            st.write("Consistency:", data["consistency"])

            st.write("Explanation:")
            st.info(data["explanation"])

            st.image(
                "data:image/jpeg;base64," + data["xai"]["gradcam_pp"],
                caption="GradCAM++"
            )

            st.image(
                "data:image/jpeg;base64," + data["xai"]["scorecam"],
                caption="ScoreCAM"
            )

            st.image(
                "data:image/jpeg;base64," + data["xai"]["integrated_gradients"],
                caption="Integrated Gradients"
            )

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
