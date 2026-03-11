import streamlit as st
import pymupdf4llm
import tempfile

# file uploader ui
st.title("Role Match")
st.write("Upload your resume to see matching roles and project-based feedback.")
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file is not None:
    # 1. Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # 2. Pass the path to PyMuPDF4LLM
    md_text = pymupdf4llm.to_markdown(tmp_path)
    
    st.success("Resume parsed successfully!")
    # Now you can send 'md_text' to Gemini

prompt = f"""
        Act as a professional career consultant. 
        Analyze the following resume and identify 3 to 5 potential job roles.
        For each role, calculate a Skill Match % and a Resume Quality % based on the content.
        
        Return the result ONLY as a JSON object with this exact structure:
        {{
            "roles": [
                {{
                    "title": "Role Name",
                    "skill_match": 85,
                    "resume_score": 70,
                    "reason": "Why this role fits based on their projects.",
                    "improvement": "One specific tip to improve the resume for this role."
                }}
            ]
        }}
        
        RESUME CONTENT:
        {md_text}
        """


