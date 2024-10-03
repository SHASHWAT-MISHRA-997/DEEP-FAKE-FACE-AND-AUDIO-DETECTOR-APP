import streamlit as st    

import numpy as np
import cv2
from helpers import detect_fake_face, analyze_audio_features
import sqlite3
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import time

# Configure Streamlit app
st.set_page_config(page_title="Deepfake Face & Audio Detector", layout="wide")

# Add CSS for custom styles
st.markdown(f"""
    <style>
        body {{
            font-family: Arial, sans-serif;
        }}
        .banner {{
            display: flex;
            align-items: center;
            height: 300px;
            border-bottom: 5px solid #7f00ff;
            position: relative;
            box-shadow: 0 0 19px rgba(0, 0, 0, 0.5);
            background: linear-gradient(90deg, rgba(255, 0, 0, 0.6), rgba(0, 255, 0, 0.6), rgba(0, 0, 255, 0.6));
            background-size: 300%;
            animation: rainbow 5s linear infinite;
        }}
        @keyframes rainbow {{
            0% {{ background-position: 0%; }}
            50% {{ background-position: 100%; }}
            100% {{ background-position: 0%; }}
        }}
        .banner-text {{
            color: white;
            font-size: 2.5em;
            text-align: left;
            z-index: 2;
            padding: 20px;
            line-height: 1.5;
            background: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
            transition: transform 0.3s, text-shadow 0.3s;
        }}
        .banner-text:hover {{
            transform: scale(1.05);
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.8);
        }}
        h1, h4 {{
            margin: 0;
            text-align: center;
            color: #333;
        }}
        .footer {{
            text-align: center;
            color: white;
            background-color: #3f00ff;
            padding: 20px;
            font-size: 4.0em;
            font-weight: bold;
            border-radius: 400px;
            margin-top: 41px;
            box-shadow: 0 5px 9px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s;
        }}
        .footer:hover {{
            background-color: #1100cc;
            transition: 0.9s;
        }}
    </style>
""", unsafe_allow_html=True)

# Add the banner section with image and title side by side
col1, col2 = st.columns([2, 3])  # Adjust column widths as needed

with col1:
    st.image("E:\\deep fake\\DP2.jpeg", use_column_width=True)  # Load the image

with col2:
    st.markdown("""<div class='banner-text'>🔍 Deepfake Face & Audio Detector 🔍</div>""", unsafe_allow_html=True) 
    st.markdown("<h4>Analyze images, videos, and audio for authenticity using AI</h4>", unsafe_allow_html=True)

    # Instructions section
    st.markdown("""<div class="instruction-box">
        <h2 class="instruction-title">How to Use the App:</h2>
        <ul>
            <li>👁️ <strong>Image Analysis:</strong> Upload an image to detect if it has been altered or deepfaked.</li>
            <li>🎥 <strong>Video Analysis:</strong> Upload a video file to check for deepfake content in the visuals.</li>
            <li>🎙️ <strong>Audio Analysis:</strong> Upload an audio file to verify its authenticity and detect synthetic voices.</li>
            <li>💾 Ensure that files are in accepted formats (e.g., .jpg, .png, .mp4, .wav).</li>
        </ul>
    </div>""", unsafe_allow_html=True)

    # Footer section with LinkedIn profile link
    st.markdown("""<a href="https://www.linkedin.com/in/sm980/" style="text-decoration: none; color: white;">
        <div class="footer">
            <p><strong>Developed by Shashwat Mishra</strong></p>
        </div></a>
    """, unsafe_allow_html=True)

# Option for choosing Image/Video or Audio
task = st.selectbox("What do you want to analyze? 🤖", ("Image", "Video", "Audio"))

# Initialize results variable
results = []

# Function for visualizing results with bar chart
def visualize_results(results):
    labels = [result['filename'] for result in results]
    scores = [np.random.rand() for _ in results]  # Replace with actual confidence scores
    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores, color='skyblue')
    plt.xlabel('Files')
    plt.ylabel('Confidence Level')
    plt.title('Deepfake Detection Confidence Levels')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Function for visualizing audio results with line plot
def visualize_audio_results(results):
    labels = [result['filename'] for result in results]
    scores = [np.random.rand() for _ in results]  # Replace with actual confidence scores for audio
    plt.figure(figsize=(10, 5))
    plt.plot(labels, scores, marker='o', linestyle='-', color='orange')
    plt.xlabel('Audio Files')
    plt.ylabel('Confidence Level')
    plt.title('Audio Detection Confidence Levels')
    plt.xticks(rotation=45)
    st.pyplot(plt)

if task == "Image":
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.image(file, caption="Uploaded Image", use_column_width=True)
            with st.spinner('Analyzing image...'):
                result = detect_fake_face(file)
            results.append({'filename': file.name, 'status': result})
        
        # Visualization
        st.write("Analysis Results:")
        for result in results:
            st.write(f"File {result['filename']}: {result['status']}")
        visualize_results(results)

elif task == "Video":
    uploaded_files = st.file_uploader("Upload Videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.video(file)
            with st.spinner('Analyzing video...'):
                result = detect_fake_face(file, is_video=True)
            results.append({'filename': file.name, 'status': result})

        # Visualization
        st.write("Analysis Results:")
        for result in results:
            st.write(f"File {result['filename']}: {result['status']}")
        visualize_results(results)

elif task == "Audio":
    uploaded_files = st.file_uploader("Upload Audio Files", type=["wav", "mp3"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with st.spinner('Analyzing audio...'):
                result = analyze_audio_features(file)
            results.append({'filename': file.name, 'status': result})

        # Visualization
        st.write("Audio Analysis Results:")
        for result in results:
            st.write(f"File {result['filename']}: {result['status']}")
        visualize_audio_results(results)

# Function for real-time analysis with bounding box and alerts
def real_time_analysis():
    cap = cv2.VideoCapture(0)  # Capture video from the first camera
    st.write("Starting real-time analysis...")
    time.sleep(1)  # Pause briefly before starting analysis
    
    # Process video frames
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze the frame
        result, confidence_score = detect_fake_face(frame)  # Assume this function returns a score too
        
        # Prepare label based on the result
        label = f"Status: {result}, Confidence: {confidence_score:.2f}"
        
        # Display results on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if result == "Real" else (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow("Real-Time Analysis", frame)

        # Break after 5 seconds of analysis or if the 'q' key is pressed
        if time.time() - start_time > 5 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    st.write("Real-time analysis finished.")

# Downloadable Report
def generate_pdf_report(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(40, 10, "Deepfake Analysis Report")
    pdf.ln(10)

    for result in results:
        pdf.cell(0, 10, f"{result['filename']}: {result['status']}", 0, 1)

    report_file = "report.pdf"
    pdf.output(report_file)
    return report_file  # Return the filename for download

# Real-time Video Analysis button
col1, col2, col3 = st.columns(3)  # Create three equal columns
with col1:
    if st.button("Start Real-time Analysis"):
        real_time_analysis()

with col2:
    if st.button("Play Analyzed Audio"):
        if os.path.exists("analyzed_audio.wav"):  # Check if file exists
            st.audio("analyzed_audio.wav")  # Use Streamlit's built-in audio player
        else:
            st.error("No analyzed audio file found. Please perform audio analysis first.")

with col3:
    if st.button("Download Report"):
        if results:  # Ensure there are results to report
            generate_pdf_report(results)
            st.success("Report generated!")
