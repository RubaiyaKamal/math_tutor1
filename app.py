import streamlit as st
import google.generativeai as genai
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import os

# ---------- Setup ----------

pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract\tesseract.exe"

genai.configure(api_key="AIzaSyCeVJTQondc1QP1rOXCGXLeRQa5mlhLkRI")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-2.0-flash")

# ---------- Utility Functions ----------

def remove_duplicates(text: str) -> str:
    sentences = re.split(r'[.?!]', text)
    seen = set()
    result = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            result.append(s_clean)
            seen.add(s_clean)
    return ". ".join(result)

# ---------- OCR + AI Functions ----------

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
    edges = cv2.Canny(denoised, 30, 200)
    enhanced = cv2.addWeighted(denoised, 0.8, edges, 0.2, 0)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    scale_percent = 300
    width = int(dilated.shape[1] * scale_percent / 100)
    height = int(dilated.shape[0] * scale_percent / 100)
    resized = cv2.resize(dilated, (width, height), interpolation=cv2.INTER_CUBIC)
    processed = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite("processed.png", processed)
    return processed

def clean_extracted_text(text: str) -> str:
    text = re.sub(r'[@0©w]+ *\)', lambda m: f"{chr(97 + (len(m.group(0).replace(' ', '')) - 1) % 26)})", text)
    text = re.sub(r'==|\+=', '=', text)
    text = re.sub(r'[lL]\b|°\s*', '°', text)
    text = re.sub(r'\bra\b|\|', '', text)
    text = re.sub(r'\b(\d+)\s*degrees\b|\b(\d+)\s*deg\b', r'\1°', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    return text.strip()

def extract_text_from_image(image: Image.Image) -> str:
    processed_img = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/=°()^ '
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    if not text.strip():
        custom_config = r'--oem 3 --psm 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/=°()^ '
        text = pytesseract.image_to_string(processed_img, config=custom_config)
    text = clean_extracted_text(text)
    text = remove_duplicates(text)
    return text

def is_math_question(line: str) -> bool:
    return bool(re.search(r'\d.*[+\-×x*/=^()°]|[xyz]', line))

def parse_questions(text: str) -> list:
    questions = []
    current_question = ""
    label_index = 0
    parts = re.split(r'(\w\))|\||\bQuestion\b|\.', text, flags=re.IGNORECASE)
    angle_pattern = r'(\b[xyz]\b|\d{1,3}°)'
    for part in parts:
        if part and re.match(r'[a-z]\)', part):
            if current_question:
                angles = re.findall(angle_pattern, current_question)
                angles = [a for a in angles if not a.startswith("180")]
                if angles and "triangle" in current_question.lower():
                    current_question += f" Angles: {', '.join(angles)}."
                if is_math_question(current_question):
                    questions.append(f"{chr(97 + label_index)}) {current_question.strip()}")
                    label_index += 1
            current_question = ""
        elif part:
            current_question += part + " "
    if current_question:
        angles = re.findall(angle_pattern, current_question)
        angles = [a for a in angles if not a.startswith("180")]
        if angles and "triangle" in current_question.lower():
            current_question += f" Angles: {', '.join(angles)}."
        if is_math_question(current_question):
            questions.append(f"{chr(97 + label_index)}) {current_question.strip()}")
    return questions

def solve_question_with_gemini(question_text: str) -> str:
    prompt = f"""
You are a helpful AI math tutor specialized in GCSE-level (AQA/Edexcel) exams, covering algebra and geometry.

Rules:
- Fix OCR errors like 'L' as '°', '=' as '2', or '@', '0)' as labels.
- Focus on solving for x if it's a triangle question (e.g., x + 2x + 63 = 180).
- Ignore invalid triangle angles like 180° inside the angle list.
- If angle expressions are unclear, assume a common GCSE pattern (x, 2x, 63°) and explain your assumption.

Solve the following question step by step:

Question: {question_text}
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error from Gemini API: {str(e)}"

# ---------- Streamlit UI ----------

st.set_page_config(page_title="MathMind – AI GCSE Solver", page_icon="📘")
st.title("📘 MathMind (Edexcel & AQA)")
st.markdown("**📖 Instantly solve GCSE math questions (algebra & geometry) using AI. Enter text or upload a photo!**")

input_method = st.radio("Choose input type", ("Text Input", "Image Upload"))

# ---------- Text Input Mode ----------

if input_method == "Text Input":
    question = st.text_area("✍️ Enter your math question below (e.g., 2x + 3 = 9 or triangle angles x, 2x, 63°):")
    if st.button("💡 Solve"):
        if question.strip():
            with st.spinner("Solving your question using Gemini..."):
                solution = solve_question_with_gemini(question)
            st.success("✅ Solution:")
            st.markdown(solution)
        else:
            st.warning("⚠️ Please enter a math question.")

# ---------- Image Upload Mode ----------

else:
    uploaded_file = st.file_uploader("📷 Upload an image with math questions", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        if st.button("🔍 Extract & Solve"):
            with st.spinner("Extracting text using OCR..."):
                extracted_text = extract_text_from_image(image)

            if not extracted_text:
                st.warning("⚠️ No text detected. Try a high-contrast image, avoid handwriting, or crop to the question area.")
            else:
                st.subheader("📝 Extracted Text")
                st.code(extracted_text)

                questions = parse_questions(extracted_text)

                if questions:
                    st.success(f"✅ Found {len(questions)} question(s).")
                    st.subheader("📘 AI-Powered Solutions")
                    for q in questions:
                        label = q.split(')')[0] + ')'
                        content = q.split(')')[1].strip()
                        with st.expander(f"Question {label}: {content}"):
                            solution = solve_question_with_gemini(q)
                            st.markdown(solution)
                else:
                    st.warning("⚠️ No math questions found. Try a clearer or more math-focused image.")

