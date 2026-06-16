

import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="Lane Detection", layout="centered")
st.title("🚗 Road Lane Detection App")

# ===== Helper Functions =====

def canny_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_img

def process_image(image):
    canny = canny_edge(image)
    cropped = region_of_interest(canny)
    lines = cv2.HoughLinesP(
        cropped,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=40,
        maxLineGap=5
    )
    line_img = display_lines(image, lines)
    final_img = cv2.addWeighted(image, 0.8, line_img, 1, 1)
    return final_img

# ===== Upload Section =====
uploaded_file = st.file_uploader("Upload a Road Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, 1)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    st.image(rgb_img, caption='🖼️ Original Image', use_column_width=True)

    processed_img = process_image(rgb_img)

    st.image(processed_img, caption='✅ Lane Detected Image', use_column_width=True)


