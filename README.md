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
<img width="1916" height="1016" alt="image" src="https://github.com/user-attachments/assets/a6b089ff-767e-48b7-a44d-c8a13588f3c9" />
<img width="603" height="837" alt="image" src="https://github.com/user-attachments/assets/ef8e2d54-241d-45cc-96db-6bb3d50ed561" />

## Multi-face Outpts:
<img width="700" height="397" alt="image" src="https://github.com/user-attachments/assets/1925fd90-5c98-4ff4-b4d0-0c050051885f" />
<img width="615" height="592" alt="image" src="https://github.com/user-attachments/assets/42d321cb-1de1-4f44-b7fe-139be795c4f8" />

---

## 📊 Severity Categorization

This project includes a severity categorization feature to grade dark spot intensity:
if final < 0.1:        <br>
&nbsp;&nbsp;&nbsp;&nbsp; sev = "Minimal"        <br>
elif final < 0.2:        <br>
&nbsp;&nbsp;&nbsp;&nbsp;sev = "Mild"        <br>
elif final < 0.4:        <br>
&nbsp;&nbsp;&nbsp;&nbsp;sev = "Moderate"        <br>
elif final < 0.6:        <br>
&nbsp;&nbsp;&nbsp;&nbsp;sev = "Severe"        <br>
else:                       <br>
&nbsp;&nbsp;&nbsp;&nbsp;sev = "Very Severe"          <br>

- Minimal: Barely noticeable dark spots.
- Mild: Light pigmentation under the eyes.
- Moderate: Clearly visible dark circles.
- Severe: Prominent pigmentation and edges.
- Very Severe: Intense dark spots requiring attention.

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
├── detector.py       # Core detection logic for face, eyes, and dark spots
├── utils.py          # Helper functions for edge detection and formatting
├── images/           # Sample input images for testing
├── outputs/          # Processed images with annotations
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
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

- 📌 Developed by: Manthan Jadav
- 📫 [LinkedIn](https://www.linkedin.com/in/manthanjadav/)
- ✉️ [Email](mailto:manthanjadav746@gmail.com)

---

## 📢 License

This project is open-source and free to use.
For educational and research purposes only.
