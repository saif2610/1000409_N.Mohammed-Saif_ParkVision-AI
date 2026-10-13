import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO


# ============================================================
# PARKVISION AI — PARKING SPACE DETECTION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best.pt"

st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING (HUMANIZED & MODERN UI)
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero-card {
        padding: 35px;
        border-radius: 20px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 30px;
    }

    .hero-title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 17px;
        line-height: 1.5;
        margin-bottom: 0;
    }

    .custom-metric {
        background: #1e293b;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .custom-metric-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }

    .custom-metric-value {
        font-size: 34px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


if not MODEL_PATH.exists():
    st.error(
        "👋 **Oops! We couldn't find your model file (`best.pt`).** "
        "Please drop your trained weights into the same folder as `app.py` and refresh!"
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"⚠️ We ran into a snag loading the YOLO model: {exc}")
    st.stop()


def normalize_class_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


CLASS_NAMES = normalize_class_names(model.names)


# ============================================================
# DETECTION ENGINE
# ============================================================

def detect_parking(
    image_rgb,
    confidence,
    image_size,
    vacant_class_id,
    occupied_class_id,
):
    results = model.predict(
        source=image_rgb,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    vacant_count = 0
    occupied_count = 0
    other_count = 0
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")

            if class_id == vacant_class_id:
                status = "Vacant"
                vacant_count += 1
            elif class_id == occupied_class_id:
                status = "Occupied"
                occupied_count += 1
            else:
                status = "Other"
                other_count += 1

            detections.append(
                {
                    "Class": class_name,
                    "Status": status,
                    "Confidence": round(score, 4),
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                }
            )

    total_spaces = vacant_count + occupied_count
    occupancy = (occupied_count / total_spaces * 100) if total_spaces > 0 else 0.0
    availability = (vacant_count / total_spaces * 100) if total_spaces > 0 else 0.0

    summary = {
        "vacant": vacant_count,
        "occupied": occupied_count,
        "other": other_count,
        "total": total_spaces,
        "occupancy": occupancy,
        "availability": availability,
    }

    return annotated_rgb, summary, detections


# ============================================================
# DASHBOARD COMPONENTS
# ============================================================

def show_summary(summary):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
            <div class="custom-metric">
                <div class="custom-metric-label">Total Spaces</div>
                <div class="custom-metric-value">{summary["total"]}</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="custom-metric">
                <div class="custom-metric-label">🟢 Open Spots</div>
                <div class="custom-metric-value" style="color: #4ade80;">{summary["vacant"]}</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="custom-metric">
                <div class="custom-metric-label">🔴 Occupied</div>
                <div class="custom-metric-value" style="color: #f87171;">{summary["occupied"]}</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="custom-metric">
                <div class="custom-metric-label">Occupancy Rate</div>
                <div class="custom-metric-value" style="color: #38bdf8;">{summary["occupancy"]:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    if summary["total"] == 0:
        st.warning("⚠️ No matching parking spots were spotted. Try lowering your confidence threshold or uploading a clearer picture!")
    else:
        st.progress(
            int(round(summary["occupancy"])),
            text=f"Lot Capacity Fullness: {summary['occupancy']:.1f}% ({summary['vacant']} spots currently available)",
        )


def show_detection_table(detections, prefix):
    if not detections:
        st.info("ℹ️ No detection data available to display.")
        return

    dataframe = pd.DataFrame(detections)
    st.dataframe(dataframe, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Download Detection Data (.csv)",
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=f"{prefix}_report.csv",
        mime="text/csv",
        key=f"{prefix}_csv",
    )


def encode_image(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError("Could not encode image.")
    return buffer.tobytes()


# ============================================================
# SIDEBAR (CONTROL CENTER)
# ============================================================

st.sidebar.title("🎛️ Control Center")
st.sidebar.caption("Fine-tune your detection preferences")
st.sidebar.divider()

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Higher values reduce false positives but might miss some cars/spaces.",
)

image_size = st.sidebar.select_slider(
    "Inference Resolution",
    options=[320, 416, 512, 640, 768, 960],
    value=640,
    help="Larger sizes capture finer details at the cost of processing speed.",
)

with st.sidebar.expander("🏷️ Class Mappings"):
    st.write("Detected classes:", ", ".join(CLASS_NAMES.values()))
    
    vacant_candidates = [c for c, n in CLASS_NAMES.items() if "empty" in n.lower() or "vacant" in n.lower()]
    occupied_candidates = [c for c, n in CLASS_NAMES.items() if "occupied" in n.lower() or "full" in n.lower()]

    default_vacant = vacant_candidates[0] if vacant_candidates else next(iter(CLASS_NAMES))
    other_class_ids = [c for c in CLASS_NAMES if c != default_vacant]
    default_occupied = occupied_candidates[0] if occupied_candidates and occupied_candidates[0] != default_vacant else (other_class_ids[0] if other_class_ids else default_vacant)

    vacant_class_id = st.selectbox(
        "Vacant Space Class",
        options=list(CLASS_NAMES.keys()),
        index=list(CLASS_NAMES.keys()).index(default_vacant),
        format_func=lambda cid: CLASS_NAMES[cid],
    )

    occupied_options = [c for c in CLASS_NAMES if c != vacant_class_id]
    if occupied_options:
        if default_occupied not in occupied_options:
            default_occupied = occupied_options[0]
        occupied_class_id = st.selectbox(
            "Occupied Space Class",
            options=occupied_options,
            index=occupied_options.index(default_occupied),
            format_func=lambda cid: CLASS_NAMES[cid],
        )
    else:
        occupied_class_id = vacant_class_id
        st.warning("Only one model class is available.")

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Adjust sliders on the fly to get cleaner bounds for your specific parking lot angle.")


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🚗 ParkVision AI</div>
        <div class="hero-subtitle">
            Your intelligent assistant for real-time parking space monitoring, vehicle counting, and lot availability tracking. Upload an image, snap a photo, or analyze live feeds effortlessly.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# APPLICATION TABS
# ============================================================

tab_image, tab_camera, tab_video, tab_about = st.tabs(
    ["🖼️ Image Studio", "📷 Live Camera", "🎬 Video Analyzer", "ℹ️ About System"]
)


# ============================================================
# IMAGE STUDIO
# ============================================================

with tab_image:
    st.subheader("📸 Parking Lot Snapshot Analysis")
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded_image is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded_image)).convert("RGB")
            image_rgb = np.array(image)

            left, right = st.columns(2)

            with left:
                st.markdown("##### Original Upload")
                st.image(image_rgb, use_container_width=True)

            if st.button("🚀 Run AI Analysis", type="primary", key="analyze_image"):
                with st.spinner("🔍 Scanning spaces and evaluating occupancy..."):
                    annotated, summary, detections = detect_parking(
                        image_rgb, confidence, image_size, vacant_class_id, occupied_class_id
                    )

                st.session_state["image_result"] = {
                    "filename": uploaded_image.name,
                    "annotated": annotated,
                    "summary": summary,
                    "detections": detections,
                }

            saved = st.session_state.get("image_result")
            if saved and saved["filename"] == uploaded_image.name:
                with right:
                    st.markdown("##### Detection Results")
                    st.image(saved["annotated"], use_container_width=True)

                st.divider()
                show_summary(saved["summary"])

                st.markdown("#### 📋 Detailed Spot Breakdown")
                show_detection_table(saved["detections"], "parking_image")

                st.download_button(
                    "💾 Download Annotated Photo",
                    data=encode_image(saved["annotated"]),
                    file_name="parkvision_analyzed.jpg",
                    mime="image/jpeg",
                )
        except Exception as exc:
            st.error(f"⚠️ We couldn't process this image: {exc}")


# ============================================================
# CAMERA
# ============================================================

with tab_camera:
    st.subheader("📷 Browser Camera Capture")
    st.caption("Point your device camera at a parking slot and snap away.")

    camera_image = st.camera_input("Capture view", key="parking_camera")

    if camera_image is not None:
        if st.button("✨ Analyze Captured Shot", type="primary", key="analyze_camera"):
            try:
                image_rgb = np.array(Image.open(camera_image).convert("RGB"))
                with st.spinner("Analyzing camera snapshot..."):
                    annotated, summary, detections = detect_parking(
                        image_rgb, confidence, image_size, vacant_class_id, occupied_class_id
                    )

                st.session_state["camera_result"] = {
                    "annotated": annotated,
                    "summary": summary,
                    "detections": detections,
                }
            except Exception as exc:
                st.error(f"⚠️ Camera analysis failed: {exc}")

        saved_camera = st.session_state.get("camera_result")
        if saved_camera:
            st.markdown("##### Camera Output")
            st.image(saved_camera["annotated"], use_container_width=True)
            show_summary(saved_camera["summary"])
            show_detection_table(saved_camera["detections"], "camera")


# ============================================================
# VIDEO ANALYSIS
# ============================================================

with tab_video:
    st.subheader("🎬 Video Sequence Processing")
    uploaded_video = st.file_uploader(
        "Upload a video clip of the parking lot",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    max_frames = st.slider("Max Frames to Process", 30, 600, 180, 30)

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("▶️ Start Video Processing", type="primary", key="process_video"):
            input_path, output_path, capture, writer = None, None, None, None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
                    temp.write(uploaded_video.getvalue())
                    input_path = temp.name

                capture = cv2.VideoCapture(input_path)
                if not capture.isOpened():
                    raise RuntimeError("Unable to open the uploaded video.")

                fps = capture.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0:
                    fps = 20.0

                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_limit = min(total_frames, max_frames) if total_frames > 0 else max_frames

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
                    output_path = temp.name

                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )

                progress = st.progress(0)
                status = st.empty()
                preview = st.empty()

                processed = 0
                last_summary, last_detections = None, []

                while processed < frame_limit:
                    success, frame = capture.read()
                    if not success:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated, summary, detections = detect_parking(
                        frame_rgb, confidence, image_size, vacant_class_id, occupied_class_id
                    )

                    writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    processed += 1
                    last_summary, last_detections = summary, detections

                    progress.progress(min(processed / frame_limit, 1.0))
                    status.write(f"⏳ Processing frame {processed} of {frame_limit}...")

                    if processed == 1 or processed % 10 == 0:
                        preview.image(annotated, caption=f"Frame {processed}", use_container_width=True)

                capture.release()
                writer.release()

                with open(output_path, "rb") as vf:
                    output_bytes = vf.read()

                st.session_state["video_result"] = {
                    "bytes": output_bytes,
                    "summary": last_summary,
                    "detections": last_detections,
                    "frames": processed,
                }
                st.success(f"🎉 Successfully processed {processed} frames!")

            except Exception as exc:
                st.error(f"⚠️ Video processing failed: {exc}")
            finally:
                if capture: capture.release()
                if writer: writer.release()
                for path in (input_path, output_path):
                    if path and os.path.exists(path):
                        try: os.remove(path)
                        except: pass

    video_result = st.session_state.get("video_result")
    if video_result:
        st.divider()
        st.subheader("📊 Video Analysis Outcome")
        st.write(f"**Total Frames Analyzed:** {video_result['frames']}")
        show_summary(video_result["summary"])
        
        st.download_button(
            "📥 Download Annotated Video",
            data=video_result["bytes"],
            file_name="parkvision_annotated.mp4",
            mime="video/mp4",
        )
        show_detection_table(video_result["detections"], "video_last_frame")


# ============================================================
# ABOUT
# ============================================================

with tab_about:
    st.subheader("💡 About ParkVision AI")
    st.write(
        "ParkVision AI leverages computer vision and custom YOLO models to instantly "
        "detect empty and filled parking bays, giving you real-time insight into lot availability."
    )
    st.markdown(f"**Active Model Checkpoint:** `{MODEL_PATH.name}`")
    for cid, cname in CLASS_NAMES.items():
        st.markdown(f"- **Class {cid}**: `{cname}`")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("✨ ParkVision AI • Built with Streamlit and Ultralytics YOLO")
