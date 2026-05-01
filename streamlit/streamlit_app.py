import streamlit as st
import requests
import json
import base64
import time
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

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
    .arrow {
        font-size: 40px;
        text-align: center;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {opacity: 0.3;}
        50% {opacity: 1;}
        100% {opacity: 0.3;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">XAI Threat Intelligence System</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def create_pdf(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("XAI Threat Intelligence Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Label: {data['label']}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {data['confidence']}", styles["Normal"]))
    elements.append(Paragraph(f"Threat Level: {data['threat_level']}", styles["Normal"]))
    elements.append(Paragraph(f"Threat Score: {data['threat_score']}", styles["Normal"]))
    elements.append(Paragraph(f"Trust: {data['trust']}", styles["Normal"]))
    elements.append(Paragraph(f"Consistency: {data['consistency']}", styles["Normal"]))
    elements.append(Paragraph(f"Uncertainty: {data['uncertainty']}", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Explanation:", styles["Heading2"]))
    elements.append(Paragraph(data["explanation"], styles["Normal"]))
    elements.append(Spacer(1, 12))

    def decode_image(b64):
        return BytesIO(base64.b64decode(b64))

    elements.append(Paragraph("GradCAM++", styles["Heading3"]))
    elements.append(RLImage(decode_image(data["gradcam_pp"]), width=200, height=200))

    elements.append(Paragraph("ScoreCAM", styles["Heading3"]))
    elements.append(RLImage(decode_image(data["scorecam"]), width=200, height=200))

    elements.append(Paragraph("Integrated Gradients", styles["Heading3"]))
    elements.append(RLImage(decode_image(data["ig"]), width=200, height=200))

    doc.build(elements)
    buffer.seek(0)
    return buffer

if uploaded_file:
    st.image(uploaded_file, caption="Original Image", use_column_width=True)

    if st.button("Analyze"):
        try:
            response = requests.post(
                "http://xai-api:8000/predict-xai",
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
            uncertainty = data.get("uncertainty", "N/A")

            cache_hit = data.get("cache", False)

            st.markdown(f"## {label}")

            if cache_hit:
                st.success("⚡ Served from cache (fast response)")
            else:
                st.info("Computed fresh inference")

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Confidence", f"{confidence:.4f}")
            col2.metric("Threat", threat)
            col3.metric("Threat Score", f"{score:.4f}")
            col4.metric("Trust", trust)
            col5.metric("Consistency", f"{consistency:.4f}")

            st.markdown("### AI Explanation")
            st.info(data["explanation"])

            st.markdown("### Explainability Flow")

            col1, col_arrow1, col2, col_arrow2, col3 = st.columns([3,1,3,1,3])

            with col1:
                st.image(
                    "data:image/jpeg;base64," + data["xai"]["gradcam_pp"],
                    caption="GradCAM++"
                )

            with col_arrow1:
                st.markdown("<div class='arrow'>➡️</div>", unsafe_allow_html=True)

            time.sleep(0.5)

            with col2:
                st.image(
                    "data:image/jpeg;base64," + data["xai"]["scorecam"],
                    caption="ScoreCAM"
                )

            with col_arrow2:
                st.markdown("<div class='arrow'>➡️</div>", unsafe_allow_html=True)

            time.sleep(0.5)

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

            st.markdown("### Download Report")

            json_report = {
                "label": label,
                "confidence": confidence,
                "threat_score": score,
                "threat_level": threat,
                "trust": trust,
                "consistency": consistency,
                "uncertainty": uncertainty,
                "explanation": data["explanation"]
            }

            st.download_button(
                label="Download JSON Report",
                data=json.dumps(json_report, indent=4),
                file_name="xai_report.json",
                mime="application/json"
            )

            pdf_data = {
                "label": label,
                "confidence": confidence,
                "threat_score": score,
                "threat_level": threat,
                "trust": trust,
                "consistency": consistency,
                "uncertainty": uncertainty,
                "explanation": data["explanation"],
                "gradcam_pp": data["xai"]["gradcam_pp"],
                "scorecam": data["xai"]["scorecam"],
                "ig": data["xai"]["integrated_gradients"]
            }

            pdf_buffer = create_pdf(pdf_data)

            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="xai_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
