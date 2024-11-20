# app.py
import cv2
import os
import logging
import streamlit as st
from sqlalchemy import text
from preprocess import read_image, extract_id_card, save_image
from ocr_engine import extract_text
from postprocess import extract_information
from face_verification import detect_and_extract_face, face_comparison, get_face_embeddings
from mysqldb_operations import insert_records, fetch_records, check_duplicacy

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

def wider_page():
    max_width_str = "max-width: 1200px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{ {max_width_str} }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_custom_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6;
                color: #333333;
            }
            .sidebar .sidebar-content {
                background-color: #ffffff;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def sidebar_section():
    st.sidebar.title("Select ID Card Type")
    id_type = st.sidebar.selectbox("ID Type", ("PAN"))
    if id_type == "PAN":
        pan_format = st.sidebar.selectbox("PAN Format", ("New Format", "Old Format"))
        return id_type, pan_format
    return id_type, None

def capture_face():
    picture = st.camera_input("Take a picture")
    if picture:
        face_image = read_image(picture, is_uploaded=True)
        if face_image is not None:
            face_path = save_image(face_image, "captured_face.jpg", path="data\\02_intermediate_data")
            return face_path
    return None

def header_section(id_type):
    if id_type == "Aadhar":
        st.title("Registration Using Aadhar Card")
    elif id_type == "PAN":
        st.title("Registration Using PAN Card")

def main_content(image_file, face_image_path, id_type, pan_format, conn):
    if image_file is not None and face_image_path is not None:
        image = read_image(image_file, is_uploaded=True)
        if image is not None:
            image_roi, _ = extract_id_card(image)
            face_image_path2 = detect_and_extract_face(img=image_roi)
            
            is_face_verified = face_comparison(image1_path=face_image_path, image2_path=face_image_path2)
            
            if is_face_verified:
                extracted_text = extract_text(image_roi)
                logging.info(f"Text extracted and information parsed from ID card: {extracted_text}")
                text_info = extract_information(extracted_text, id_type, pan_format)
                logging.info(f"Text processed: {text_info}")
            
            if text_info["ID"]:
                records = fetch_records(text_info)
                if records.shape[0] > 0:
                    st.write("Existing Records:")
                    st.write(records)
                
                is_duplicate = check_duplicacy(text_info)
                if is_duplicate:
                    st.error(f"User already present with ID {text_info['ID']}")
                else:
                    st.success("User Information:")
                    st.write(text_info)
                    # text_info['DOB'] = text_info['DOB'].strftime('%Y-%m-%d')
                    text_info['Embedding'] = get_face_embeddings(face_image_path)
                    insert_records(text_info)
                    st.success("Registration Successful!")
            else:
                st.error("Face verification failed. Please try again.")
        else:
            st.error("Failed to process ID card image. Please try again.")
    else:
        if image_file is None:
            st.warning("Please upload an ID card image.")
        if face_image_path is None:
            st.warning("Please capture your face image.")

def main():
    conn = st.connection(
        'mysql',
        type='sql',
        dialect="mysql",
        host="localhost",
        port=3306,
        database="ekyc",
        username="root",
        password="Prasham"
    )
    
    wider_page()
    set_custom_theme()
    
    id_type, pan_format = sidebar_section()
    header_section(id_type)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload ID Card")
        image_file = st.file_uploader("Choose an ID card image", type=['jpg', 'jpeg', 'png'])
        if image_file:
            st.image(image_file, caption="Uploaded ID Card", use_column_width=True)
    
    with col2:
        st.subheader("Capture Face")
        face_image_path = capture_face()
    
    if st.button("Process Registration"):
        main_content(image_file, face_image_path, id_type, pan_format, conn)

if __name__ == "__main__":
    main()