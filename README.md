# Medicine Identification System Using OpenVINO

This project identifies medicines using YOLO11n classification, OpenVINO, and PaddleOCR.

## Features
- Identification of trained medicines using a deep learning model.
- OCR-based identification of untrained medicines.
- Real-time webcam scanning.
- CPU and GPU inference using OpenVINO.
- FPGA functional simulation using Verilog.

## Technologies Used
Python, YOLO11n, OpenVINO, PaddleOCR, OpenCV, and Verilog.

## Project Results
- Trained medicine classes: 53
- Test dataset accuracy: 99.82%
- CPU inference time: 10.83 ms
- GPU inference time: 2.77 ms
- GPU speedup: 3.91×

**Note:** The reported accuracy is from the held-out image test dataset, not live webcam testing.
