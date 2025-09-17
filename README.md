# 🧑‍⚕️ Face Dark Spot & Eye Detection 🔍

An intelligent **OpenCV-based** application to detect **faces**, **eyes**, and **dark spots around the eyes**.  
Upload a face image, and the system automatically detects edges, highlights the eyes, and identifies dark spots with precise formatting.

---

## 📸 Overview

This project uses **OpenCV** and **Haar Cascade classifiers** to process face images:  
- Detect and extract facial regions.  
- Identify eye positions.  
- Highlight dark spots under the eyes using edge detection.

---

## Tested Outputs:

---

## 🧩 Problem Statement

Detecting facial features like eyes and dark spots can be useful for:  
- **Cosmetic Analysis** (skincare & beauty apps).  
- **Medical Imaging** (detecting under-eye conditions).  
- **Face Preprocessing** for AI-based recognition or emotion detection.

---

## 🚀 Key Features

| Feature                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| 👁 **Face Detection**     | Accurately detects faces using Haar Cascade classifiers.                    |
| 👓 **Eye Detection**      | Locates and labels both eyes with adjustable formatting.                    |
| 🌑 **Dark Spot Detection**| Uses edge detection to highlight darker areas under the eyes.               |
| 📷 **Annotated Output**   | Saves or displays the processed image with highlighted features.            |
| ⚡ **Fast Processing**    | Lightweight and optimized for real-time detection.                          |

---

## 🛠️ Tech Stack

| Layer          | Technology          |
|----------------|-------------------|
| **Language**   | Python             |
| **Libraries**  | OpenCV, NumPy      |
| **IDE**        | VS Code, Jupyter   |
| **Output**     | Image Annotations  |

---

## 📂 Project Structure

```bash
face-darkspot-detector/
│
├── detector.py # Core detection logic for face, eyes, and dark spots
├── utils.py # Helper functions for edge detection and formatting
├── images/ # Sample input images for testing
├── outputs/ # Processed images with annotations
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## ⚙️ How to Run

1. **Clone the Repo**
   ```bash
   git clone https://github.com/your-username/face-darkspot-detector
   cd face-darkspot-detector
   ```

2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the file
   ```bash
   streamlir run app.py
   ```

---

## 📈 Future Enhancements
- 🤖 Deep Learning Integration: Replace Haar cascades with CNN-based detection (e.g., MTCNN or YOLO).
- 📊 Dark Spot Grading: Quantify severity of dark spots for skincare analytics.
- 🎨 Better Visualization: Add GUI with sliders for edge detection thresholds.
- 📱 Mobile Deployment: Wrap in a lightweight mobile app for real-time usage.

---

## 📧 Contact

Developer: Manthan Jadav
LinkedIn: Manthan Jadav[https://www.linkedin.com/in/manthanjadav/]
Email: manthanjadav746@gmail.com[mailto:manthanjadav746@gmail.com]

---

##📢 License

This project is open-source and free to use.
For educational and research purposes only.
