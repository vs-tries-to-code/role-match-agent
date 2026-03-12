import streamlit as st
import pymupdf4llm
import tempfile
from google import genai 
import json
import os
from dotenv import load_dotenv

# Configuration
load_dotenv()
api_key = os.getenv("API_KEY")

# Initializing with the NEW Client approach (instead of genai.configure())
client = genai.Client(api_key=api_key)

st.set_page_config(page_title="Career Role Discovery", layout="wide")

# UI Header
st.title("AI Career Discovery Agent")
st.write("Upload your resume to see matching roles and project-based feedback.")

# File uploader
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file is not None:
    # A spinner keeps the UI active while the AI "thinks"
    with st.spinner("Agent is analyzing your skills and projects..."):
        
        # Save to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # Added 'pymupdf4llm.use_layout(False)' to SKIP the ONNX engine entirely.
            # Using force_text=True  alone is not enough
            pymupdf4llm.use_layout(False)
            md_text = pymupdf4llm.to_markdown(tmp_path, force_text=True)
            
            # The prompt fed to the model
            prompt = f"""
            Identify 3 job roles for this candidate based on their skills.
            Return ONLY a JSON object with this exact structure:
            {{
                "roles": [
                    {{
                        "title": "Role Name",
                        "skill_match": 85,
                        "resume_score": 70,
                        "reason": "Why this fits based on their projects.",
                        "improvement": "One tip to improve."
                    }}
                ]
            }}
            RESUME: {md_text}
            """

            # Modern LLM Call
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            
            #Parsing and UI Display
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw_text)

            st.success("Analysis Complete!")
            st.divider()

            for item in data['roles']:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.subheader(f"{item['title']}")
                        st.write(f"**Justification:** {item['reason']}")
                        st.info(f"**Tip:** {item['improvement']}")
                    with col2:
                        st.write(f"**Skill Match: {item['skill_match']}%**")
                        st.progress(item['skill_match'] / 100)
                        st.write(f"**Resume Score: {item['resume_score']}%**")
                        st.progress(item['resume_score'] / 100)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
        finally:
            # Cleanup: deleting the temp file to save space
            if os.path.exists(tmp_path):
                os.remove(tmp_path)