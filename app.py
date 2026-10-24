import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from PIL import Image
import io
from io import BytesIO
import pymupdf as fitz

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

SYSTEM_PROMPT = """
You are an AI Medical Report Reader.
- User will upload a medical report image or text.
- Explain it in simple Hindi + English mix.
- Structure: 1. Report kya kehta hai 2. Abnormal values kaunsi hai 3. Kya karna chahiye
- Last me disclaimer: I am AI, not a doctor, please consult doctor.
- Be friendly and clear.
"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_text = request.form.get("message", "")
        file = request.files.get("file")

        contents = [SYSTEM_PROMPT]

        if file and file.filename!= "":
            image_bytes = file.read()
            filename  = file.filename.lower()
            if filename.endswith(".pdf"):
                try:
                    doc = fitz.open(stream=image_bytes,filetype="pdf")
                    page = doc[0]
                    pix = page.get_pixmap(dpi=200)
                    pil_image = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
                    contents.append(pil_image)
                    pdf_text = ""
                    for p in doc:
                        pdf_text += p.get_text()
                    if pdf_text.strip():
                        contents.append(f"PDF se nikla hua text:{pdf_text[:4000]}")
                except Exception as e:
                    return jsonify({"reply":f"PDF read karne mein error:{e}"})
            else:
                try:
                    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    contents.append(pil_image)
                except Exception as e:
                    return jsonify({"reply":f"Image read nahi ho payi: {e}. Please JPG/PNG upload karo."})
            if user_text:
                contents.append(user_text)
            else:
                contents.append("Is medical report ko simple Hindi me samjhao")
        else:
            contents.append(user_text)

        # Jo model available ho usse try karega
        models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
        last_error = ""
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                return jsonify({"reply": response.text})
            except Exception as e:
                last_error = str(e)
                continue

        return jsonify({"reply": f"Server busy hai: {last_error}"})

    except Exception as e:
        return jsonify({"reply": f"Error: {e}"})

if __name__ == "__main__":
    app.run(debug=True)
