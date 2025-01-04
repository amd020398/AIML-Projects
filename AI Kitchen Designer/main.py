import os
import cv2
import torch
import uvicorn as uvicorn
from fastapi import FastAPI, UploadFile
from diffusers import StableDiffusionPipeline
from pathlib import Path
from typing import List
import streamlit as st

# Step 1: Environment Setup
# Ensure required libraries are installed
# pip install opencv-python fastapi torch diffusers transformers streamlit

# Initialize FastAPI
app = FastAPI()

# Step 2: Space Analysis - Define function for kitchen space analysis
def analyze_space(image_path: str):
    """
    Analyze the kitchen space by detecting key areas.
    """
    print(f"Analyzing space for image: {image_path}")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    analyzed_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    print("Space analysis completed.")
    return analyzed_image

# Step 3: Design Generation - Load Stable Diffusion
print("Loading Stable Diffusion model...")
stable_diffusion = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
print("Stable Diffusion model loaded successfully.")

def generate_design(prompt: str):
    """
    Generate an interior design using Stable Diffusion.
    """
    print(f"Generating design for prompt: {prompt}")
    image = stable_diffusion(prompt)["images"][0]
    print("Design generation completed.")
    return image

# Step 4: Video Processing - Analyze and Design for Videos
def process_video(video_path: str, output_path: str):
    """
    Process a video frame-by-frame to generate designs.
    """
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_count}...")
        # Save the frame to a temporary file
        temp_path = f"temp_frame_{frame_count}.jpg"
        cv2.imwrite(temp_path, frame)

        # Analyze and generate design for the frame
        analyzed_frame = analyze_space(temp_path)
        os.remove(temp_path)

        # Write to the output video
        if out is None:
            height, width, _ = analyzed_frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

        out.write(analyzed_frame)
        frame_count += 1

    cap.release()
    if out:
        out.release()
    print(f"Video processing completed. Output saved to {output_path}")

# Step 5: User Interface - Using Streamlit
st.title("AI Kitchen Interior Designer")

uploaded_file = st.file_uploader("Upload a photo or video of your kitchen", type=["jpg", "jpeg", "png", "mp4"])

design_prompt = st.text_input("Describe your preferred design style (e.g., modern, rustic, minimalistic):", "Modern kitchen interior")

if uploaded_file:
    file_path = Path("uploads") / uploaded_file.name
    print(f"File uploaded: {file_path}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    if file_path.suffix in [".jpg", ".jpeg", ".png"]:
        # Process photo
        print("Processing photo...")
        analyzed_image = analyze_space(str(file_path))
        st.image(analyzed_image, caption="Analyzed Kitchen Space")

        design_image = generate_design(design_prompt)
        st.image(design_image, caption="AI-Generated Design")

    elif file_path.suffix == ".mp4":
        # Process video
        print("Processing video...")
        output_video_path = str(file_path.with_name(f"output_{file_path.stem}.mp4"))
        process_video(str(file_path), output_video_path)
        st.video(output_video_path)

    st.success("Processing completed!")
    print("Processing completed successfully!")

# Run the FastAPI app alongside Streamlit (use uvicorn for FastAPI)
# if __name__ = "__main__":
#     uvicorn main:app --reload
# streamlit run main.py




