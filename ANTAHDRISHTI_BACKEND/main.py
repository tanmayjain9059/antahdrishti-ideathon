from reportlab.lib.utils import ImageReader
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from datetime import datetime
from pathlib import Path
from database import (
    create_database,
    save_inspection,
    get_inspections,
    get_health_history,
    save_feedback,
    get_feedback,
)

import shutil
import uuid
import json
import cv2


app = FastAPI(
    title="ANTAHDRISHTI",
    description="AI-assisted infrastructure inspection platform",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR.parent / "frontend" / "index.html"
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "results"
REPORT_FOLDER = BASE_DIR / "reports"
MODEL_PATH = BASE_DIR / "best.pt"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)
REPORT_FOLDER.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"best.pt was not found at: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
create_database()

latest_analysis = {
    "inspection_id": None,
    "filename": "No inspection completed",
    "bridge_id": "Not provided",
    "bridge_name": "Not provided",
    "location": "Not provided",
    "engineer_name": "Not provided",
    "bridge_element": "Unknown",
    "image_quality": "Unknown",
    "scale_reference_available": "No",
    "total_risk": 0,
    "detections": [],
    "priority": "UNKNOWN",
    "recommendation": "Complete an inspection first.",
    "average_confidence": 0,
    "health_score": 100,
    "result_image": None,
    "inspection_time": None,
}


@app.get("/")
def home():
    if not FRONTEND_FILE.exists():
        return {
            "status": "error",
            "message": f"Frontend file not found at: {FRONTEND_FILE}",
        }
    return FileResponse(str(FRONTEND_FILE))


@app.get("/model-info")
def model_info():
    return {
        "status": "success",
        "model_path": str(MODEL_PATH),
        "classes": model.names,
    }


# -----------------------------------------------------------------------------
# AI VISUAL INSPECTION PRIORITY ENGINE
# -----------------------------------------------------------------------------

DEFECT_BASE_SEVERITY = {
    "crack": 18.0,
    "corrosion": 24.0,
    "spalling": 34.0,
    "exposed_rebar": 48.0,
    "exposed reinforcement": 48.0,
    "exposed_reinforcement": 48.0,
}

ELEMENT_MULTIPLIER = {
    "parapet": 0.70,
    "deck": 0.90,
    "joint": 1.00,
    "other": 1.00,
    "unknown": 1.00,
    "bearing": 1.20,
    "girder": 1.30,
    "beam": 1.30,
    "pier": 1.30,
    "column": 1.30,
    "abutment": 1.15,
}

CRITICAL_ELEMENTS = {"bearing", "girder", "beam", "pier", "column"}


def normalise_label(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def confidence_factor(confidence: float) -> float:
    """
    Confidence measures class reliability, not physical severity.
    A low-confidence detection must never erase a visually large defect.
    """
    if confidence < 0.30:
        return 0.60
    if confidence < 0.50:
        return 0.70
    if confidence < 0.70:
        return 0.80
    if confidence < 0.85:
        return 0.90
    return 1.00


def area_factor(area_ratio: float) -> float:
    """Bounding-box coverage in the image; not a physical area measurement."""
    if area_ratio < 0.02:
        return 0.60
    if area_ratio < 0.08:
        return 0.80
    if area_ratio < 0.20:
        return 1.00
    if area_ratio < 0.50:
        return 1.20
    return 1.50


def condition_rank_from_severity(severity: float) -> int:
    if severity < 15:
        return 1
    if severity < 30:
        return 2
    if severity < 50:
        return 3
    return 4


def minimum_condition_rank(defect_key: str, area_ratio: float) -> int:
    """Conservative visual floor so large visible defects cannot become CS1."""
    rank = 1

    if defect_key in {"spalling", "exposed_rebar", "exposed_reinforcement"}:
        rank = max(rank, 2)

    if defect_key in {"exposed_rebar", "exposed_reinforcement"}:
        rank = max(rank, 3)

    if area_ratio >= 0.20:
        rank = max(rank, 2)

    if area_ratio >= 0.50 and defect_key in {
        "crack", "corrosion", "spalling", "exposed_rebar", "exposed_reinforcement"
    }:
        rank = max(rank, 3)

    return rank


def visual_condition_state(severity: float, defect_key: str, area_ratio: float) -> str:
    rank = max(
        condition_rank_from_severity(severity),
        minimum_condition_rank(defect_key, area_ratio),
    )
    return {
        1: "CS1 - GOOD",
        2: "CS2 - FAIR",
        3: "CS3 - POOR",
        4: "CS4 - SEVERE",
    }[rank]


def iou(box_a: dict, box_b: dict) -> float:
    x_left = max(box_a["x1"], box_b["x1"])
    y_top = max(box_a["y1"], box_b["y1"])
    x_right = min(box_a["x2"], box_b["x2"])
    y_bottom = min(box_a["y2"], box_b["y2"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def remove_duplicate_detections(detections: list[dict], threshold: float = 0.60) -> list[dict]:
    """Keep the strongest overlapping detection of the same class."""
    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept = []

    for candidate in ordered:
        duplicate = any(
            normalise_label(candidate["defect"]) == normalise_label(existing["defect"])
            and iou(candidate["bounding_box"], existing["bounding_box"]) >= threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)

    return kept


def calculate_visual_priority(detections: list[dict], bridge_element: str, image_quality: str):
    element_key = normalise_label(bridge_element)
    element_weight = ELEMENT_MULTIPLIER.get(element_key, 1.00)

    scored = []
    for detection in detections:
        defect_key = normalise_label(detection["defect"])
        base = DEFECT_BASE_SEVERITY.get(defect_key, 15.0)
        conf_ratio = detection["confidence"] / 100.0
        conf_weight = confidence_factor(conf_ratio)
        extent_weight = area_factor(detection["area_ratio"])

        severity = round(base * conf_weight * extent_weight * element_weight, 2)
        detection["base_severity"] = base
        detection["confidence_factor"] = conf_weight
        detection["area_factor"] = extent_weight
        detection["element_multiplier"] = element_weight
        detection["severity_points"] = severity
        detection["suggested_condition_state"] = visual_condition_state(
            severity, defect_key, detection["area_ratio"]
        )
        detection["confidence_warning"] = conf_ratio < 0.50
        scored.append(detection)

    scored.sort(key=lambda item: item["severity_points"], reverse=True)
    weights = [1.00, 0.50, 0.25]
    total_risk = sum(
        item["severity_points"] * weights[index]
        for index, item in enumerate(scored[:3])
    )
    total_risk = round(min(100.0, total_risk), 2)
    screening_score = round(max(0.0, 100.0 - total_risk))

    if screening_score >= 80:
        priority = "LOW"
        recommendation = "Continue routine monitoring and retain this inspection for comparison."
    elif screening_score >= 60:
        priority = "MEDIUM"
        recommendation = "Engineer review is recommended and the affected element should be monitored."
    elif screening_score >= 30:
        priority = "HIGH"
        recommendation = "Prioritize a detailed engineering inspection of the affected element."
    else:
        priority = "CRITICAL"
        recommendation = "Immediate professional assessment is recommended before maintenance decisions."

    severe_labels = {"exposed_rebar", "exposed_reinforcement", "exposed_reinforcement_", "exposed reinforcement"}
    has_exposed_rebar = any(normalise_label(item["defect"]) in severe_labels for item in scored)
    has_severe_spalling = any(
        normalise_label(item["defect"]) == "spalling" and item["severity_points"] >= 34
        for item in scored
    )

    has_large_visible_defect = any(
        item["area_ratio"] >= 0.50
        and normalise_label(item["defect"]) in {
            "crack", "corrosion", "spalling", "exposed_rebar", "exposed_reinforcement"
        }
        for item in scored
    )
    has_low_confidence_large_defect = any(
        item["area_ratio"] >= 0.20 and item["confidence"] < 50
        for item in scored
    )

    if element_key in CRITICAL_ELEMENTS and (has_exposed_rebar or has_severe_spalling):
        if priority in {"LOW", "MEDIUM"}:
            priority = "HIGH"
            recommendation = "A critical structural element shows a significant visible defect. Prioritize detailed engineering inspection."

    if has_large_visible_defect and priority in {"LOW", "MEDIUM"}:
        priority = "HIGH"
        recommendation = (
            "A large visible defect region was detected. Prioritize engineer review; "
            "the bounding-box coverage is image-relative and must be verified on site."
        )

    if has_low_confidence_large_defect:
        recommendation = (
            "A visually large defect region was detected with low AI confidence. "
            "Do not treat this as a healthy result; capture a clearer image and obtain engineer verification."
        )

    if normalise_label(image_quality) == "poor":
        priority = "MANUAL_REVIEW"
        recommendation = "Image quality is insufficient for reliable AI screening. Capture a clearer image and perform manual review."

    return scored, screening_score, total_risk, priority, recommendation


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    bridge_id: str = Form(...),
    bridge_name: str = Form(...),
    location: str = Form(...),
    engineer_name: str = Form(...),
    bridge_element: str = Form("Unknown"),
    image_quality: str = Form("Good"),
    scale_reference_available: str = Form("No"),
):
    global latest_analysis

    bridge_id = bridge_id.strip()
    bridge_name = bridge_name.strip()
    location = location.strip()
    engineer_name = engineer_name.strip()
    bridge_element = bridge_element.strip() or "Unknown"
    image_quality = image_quality.strip() or "Good"
    scale_reference_available = scale_reference_available.strip() or "No"

    if not bridge_id:
        return {"status": "error", "message": "Bridge ID is required."}
    if not bridge_name:
        return {"status": "error", "message": "Bridge name is required."}
    if not location:
        return {"status": "error", "message": "Location is required."}
    if not engineer_name:
        return {"status": "error", "message": "Engineer name is required."}

    original_extension = Path(file.filename).suffix.lower()
    if original_extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        return {
            "status": "error",
            "message": "Please upload a JPG, JPEG, PNG or WEBP image.",
        }

    unique_name = f"{uuid.uuid4().hex}{original_extension}"
    upload_path = UPLOAD_FOLDER / unique_name

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(
        source=str(upload_path),
        conf=0.25,
        save=False,
        verbose=False,
    )

    result = results[0]
    image_height, image_width = result.orig_shape
    image_area = max(1, image_width * image_height)
    detections = []
    total_confidence = 0.0

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = model.names[class_id]
            coordinates = box.xyxy[0].tolist()

            detections.append({
                "defect": class_name,
                "confidence": round(confidence * 100, 2),
                "bounding_box": {
                    "x1": round(coordinates[0], 2),
                    "y1": round(coordinates[1], 2),
                    "x2": round(coordinates[2], 2),
                    "y2": round(coordinates[3], 2),
                },
                "area_ratio": round(
                    max(0.0, coordinates[2] - coordinates[0])
                    * max(0.0, coordinates[3] - coordinates[1])
                    / image_area,
                    4,
                ),
            })
            total_confidence += confidence

    plotted_image = result.plot()
    result_filename = f"detected_{unique_name}"
    result_path = RESULT_FOLDER / result_filename

    if not cv2.imwrite(str(result_path), plotted_image):
        return {
            "status": "error",
            "message": "The detected result image could not be saved.",
        }

    detections = remove_duplicate_detections(detections)
    defect_count = len(detections)

    detections, health_score, total_risk, priority, recommendation = calculate_visual_priority(
        detections=detections,
        bridge_element=bridge_element,
        image_quality=image_quality,
    )

    average_confidence = (
        round((total_confidence / defect_count) * 100, 2)
        if defect_count > 0
        else 0
    )

    inspection_time = datetime.now().strftime("%d-%m-%Y %H:%M")

    inspection_id = save_inspection(
        bridge_id=bridge_id,
        bridge_name=bridge_name,
        location=location,
        image_name=file.filename,
        defect_count=defect_count,
        detections=json.dumps(detections),
        average_confidence=average_confidence,
        health_score=health_score,
        priority=priority,
        recommendation=recommendation,
        inspection_time=inspection_time,
    )

    latest_analysis = {
        "inspection_id": inspection_id,
        "filename": file.filename,
        "bridge_id": bridge_id,
        "bridge_name": bridge_name,
        "location": location,
        "engineer_name": engineer_name,
        "bridge_element": bridge_element,
        "image_quality": image_quality,
        "scale_reference_available": scale_reference_available,
        "total_risk": total_risk,
        "detections": detections,
        "priority": priority,
        "recommendation": recommendation,
        "average_confidence": average_confidence,
        "health_score": health_score,
        "result_image": result_filename,
        "inspection_time": inspection_time,
    }

    return {
        "status": "success",
        "inspection_id": inspection_id,
        "original_filename": file.filename,
        "bridge_id": bridge_id,
        "bridge_name": bridge_name,
        "location": location,
        "engineer_name": engineer_name,
        "bridge_element": bridge_element,
        "image_quality": image_quality,
        "scale_reference_available": scale_reference_available,
        "model_classes": model.names,
        "defect_count": defect_count,
        "detections": detections,
        "average_confidence": average_confidence,
        "visual_priority_score": health_score,
        "health_score": health_score,
        "total_risk_points": total_risk,
        "preliminary_priority": priority,
        "recommendation": recommendation,
        "inspection_time": inspection_time,
        "result_image_url": f"/result-image/{result_filename}",
        "note": (
            "This is an AI visual inspection priority score, not an official structural condition rating or safety certification."
        ),
    }


@app.get("/result-image/{filename}")
def get_result_image(filename: str):
    safe_filename = Path(filename).name
    file_path = RESULT_FOLDER / safe_filename

    if not file_path.exists():
        return {"status": "error", "message": "Result image not found."}

    return FileResponse(str(file_path))


@app.get("/inspections")
def inspection_history():
    return {
        "status": "success",
        "inspections": get_inspections(),
    }


@app.get("/dashboard-summary")
def dashboard_summary():
    inspections = get_inspections()

    priority_counts = {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
        "CRITICAL": 0,
        "MANUAL_REVIEW": 0,
    }

    total_defects = 0
    reviewed_inspections = 0

    for inspection in inspections:
        priority = inspection.get("priority", "UNKNOWN")
        if priority in priority_counts:
            priority_counts[priority] += 1

        total_defects += int(inspection.get("defect_count", 0))

        if inspection.get("has_feedback"):
            reviewed_inspections += 1

    total_inspections = len(inspections)

    return {
        "status": "success",
        "total_inspections": total_inspections,
        "unique_bridges": len({
            inspection["bridge_id"]
            for inspection in inspections
        }),
        "total_defects": total_defects,
        "reviewed_inspections": reviewed_inspections,
        "pending_review": total_inspections - reviewed_inspections,
        "priority_counts": priority_counts,
        "latest_inspection": inspections[0] if inspections else None,
    }


@app.get("/health-history/{bridge_id}")
def health_history(bridge_id: str):
    return {
        "status": "success",
        "bridge_id": bridge_id,
        "history": get_health_history(bridge_id),
    }


@app.post("/submit-feedback")
def submit_feedback(
    inspection_id: int = Form(...),
    engineer_decision: str = Form(...),
    corrected_defect: str = Form(""),
    final_priority: str = Form(""),
    engineer_remarks: str = Form(""),
):
    engineer_decision = engineer_decision.strip().upper()
    corrected_defect = corrected_defect.strip()
    final_priority = final_priority.strip().upper()
    engineer_remarks = engineer_remarks.strip()

    allowed_decisions = {"ACCEPTED", "NEEDS_CORRECTION", "REJECTED"}
    allowed_priorities = {"", "LOW", "MEDIUM", "HIGH", "CRITICAL"}

    if engineer_decision not in allowed_decisions:
        return {
            "status": "error",
            "message": "Invalid engineer decision.",
        }

    if final_priority not in allowed_priorities:
        return {
            "status": "error",
            "message": "Invalid final priority.",
        }

    inspection_ids = {
        int(inspection["id"])
        for inspection in get_inspections()
    }

    if inspection_id not in inspection_ids:
        return {
            "status": "error",
            "message": "Inspection ID was not found.",
        }

    feedback_time = datetime.now().strftime("%d-%m-%Y %H:%M")

    feedback_id = save_feedback(
        inspection_id=inspection_id,
        engineer_decision=engineer_decision,
        corrected_defect=corrected_defect or None,
        final_priority=final_priority or None,
        engineer_remarks=engineer_remarks or None,
        feedback_time=feedback_time,
    )

    return {
        "status": "success",
        "message": "Engineer feedback saved successfully.",
        "feedback_id": feedback_id,
        "inspection_id": inspection_id,
        "feedback_time": feedback_time,
    }


@app.get("/feedback")
def feedback_history(
    inspection_id: int | None = Query(default=None),
):
    return {
        "status": "success",
        "feedback": get_feedback(inspection_id),
    }


def draw_wrapped_text(c, text, x, y, max_chars=82, line_height=15):
    words = str(text).split()
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()

        if len(test_line) <= max_chars:
            current_line = test_line
        else:
            c.drawString(x, y, current_line)
            y -= line_height
            current_line = word

    if current_line:
        c.drawString(x, y, current_line)
        y -= line_height

    return y


@app.get("/generate-report")
def generate_report():
    report_id = "INS-" + str(uuid.uuid4())[:8].upper()
    filename = f"{report_id}.pdf"
    filepath = REPORT_FOLDER / filename

    c = canvas.Canvas(str(filepath), pagesize=A4)
    width, height = A4

    c.setFillColor(colors.HexColor("#15324A"))
    c.rect(0, height - 105, width, 105, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 23)
    c.drawString(42, height - 45, "ANTAHDRISHTI")

    c.setFont("Helvetica", 10)
    c.drawString(
        42,
        height - 68,
        "AI-Assisted Infrastructure Inspection Report",
    )

    c.setFont("Helvetica-Bold", 9)
    c.drawRightString(
        width - 42,
        height - 44,
        f"Report ID: {report_id}",
    )

    c.setFont("Helvetica", 9)
    c.drawRightString(
        width - 42,
        height - 65,
        datetime.now().strftime("%d-%m-%Y %H:%M"),
    )

    y = height - 135

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, y, "Inspection Details")
    y -= 24

    details = [
        ("Inspection ID", latest_analysis.get("inspection_id", "Not available")),
        ("Bridge ID", latest_analysis.get("bridge_id", "Not provided")),
        ("Bridge Name", latest_analysis.get("bridge_name", "Not provided")),
        ("Location", latest_analysis.get("location", "Not provided")),
        ("Engineer", latest_analysis.get("engineer_name", "Not provided")),
        ("Bridge Element", latest_analysis.get("bridge_element", "Unknown")),
        ("Image Quality", latest_analysis.get("image_quality", "Unknown")),
        ("Scale Reference", latest_analysis.get("scale_reference_available", "No")),
        ("Inspection Time", latest_analysis.get("inspection_time", "Not available")),
        ("Image", latest_analysis.get("filename", "Unknown")),
    ]

    for label, value in details:
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(52, y, f"{label}:")
        c.setFont("Helvetica", 9.5)
        c.drawString(150, y, str(value))
        y -= 17

    y -= 8

    health_score = latest_analysis.get("health_score", 100)
    priority = latest_analysis.get("priority", "UNKNOWN")
    average_confidence = latest_analysis.get("average_confidence", 0)

    c.setFillColor(colors.HexColor("#F2F5F7"))
    c.roundRect(42, y - 88, width - 84, 84, 10, fill=1, stroke=0)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(56, y - 24, "AI Visual Inspection Priority Score")

    c.setFont("Helvetica-Bold", 26)
    c.drawString(56, y - 58, f"{health_score} / 100")

    priority_color = {
        "LOW": colors.green,
        "MEDIUM": colors.orange,
        "HIGH": colors.red,
        "CRITICAL": colors.darkred,
    }.get(priority, colors.black)

    c.setFillColor(priority_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(270, y - 43, f"Priority: {priority}")

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 9.5)
    c.drawString(
        270,
        y - 63,
        f"Average AI Confidence: {average_confidence}%",
    )

    y -= 112

    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, y, "Detected Defects")
    y -= 22

    detections = latest_analysis.get("detections", [])

    if not detections:
        c.setFont("Helvetica", 9.5)
        c.drawString(52, y, "No visible defects were detected by the AI model.")
        y -= 20
    else:
        c.setFillColor(colors.HexColor("#E8EEF2"))
        c.rect(48, y - 18, width - 96, 22, fill=1, stroke=0)

        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(58, y - 11, "Defect")
        c.drawString(205, y - 11, "Confidence")
        c.drawString(300, y - 11, "Extent")
        c.drawString(385, y - 11, "Severity")
        y -= 30

        for detection in detections:
            c.setFont("Helvetica", 9.5)
            c.drawString(58, y, str(detection["defect"]))
            c.drawString(205, y, f'{detection["confidence"]}%')
            c.drawString(300, y, f'{round(detection.get("area_ratio", 0) * 100, 2)}%')
            c.drawString(385, y, str(detection.get("severity_points", 0)))
            y -= 18

    y -= 10

    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, y, "Recommendation")
    y -= 22

    c.setFont("Helvetica", 9.5)
    y = draw_wrapped_text(
        c,
        latest_analysis.get(
            "recommendation",
            "Complete an inspection before generating the report.",
        ),
        52,
        y,
    )

    y -= 8

    # ENGINEER VERIFICATION — safely inside generate_report()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, y, "Engineer Verification")
    y -= 22

    inspection_id = latest_analysis.get("inspection_id")
    feedback_records = get_feedback(inspection_id) if inspection_id else []
    latest_feedback = feedback_records[0] if feedback_records else None

    if latest_feedback:
        verification_details = [
            ("Decision", latest_feedback.get("engineer_decision", "Not provided")),
            (
                "Corrected Defect",
                latest_feedback.get("corrected_defect") or "No correction",
            ),
            (
                "Final Priority",
                latest_feedback.get("final_priority") or priority,
            ),
            (
                "Feedback Time",
                latest_feedback.get("feedback_time", "Not available"),
            ),
        ]

        for label, value in verification_details:
            c.setFont("Helvetica-Bold", 9.5)
            c.drawString(52, y, f"{label}:")
            c.setFont("Helvetica", 9.5)
            c.drawString(155, y, str(value))
            y -= 17

        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(52, y, "Engineer Remarks:")
        y -= 17

        c.setFont("Helvetica", 9.5)
        y = draw_wrapped_text(
            c,
            latest_feedback.get("engineer_remarks") or "No remarks provided.",
            52,
            y,
            max_chars=80,
        )
    else:
        c.setFont("Helvetica", 9.5)
        c.drawString(52, y, "Engineer review is pending for this inspection.")
        y -= 18

    result_filename = latest_analysis.get("result_image")

    if result_filename:
        result_image_path = RESULT_FOLDER / result_filename

        if result_image_path.exists():
            c.showPage()

            c.setFillColor(colors.HexColor("#15324A"))
            c.rect(0, height - 86, width, 86, fill=1, stroke=0)

            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 19)
            c.drawString(42, height - 44, "AI Detection Evidence")

            c.setFont("Helvetica", 9.5)
            c.drawString(
                42,
                height - 64,
                f"Bridge: {latest_analysis.get('bridge_name', 'Not provided')}",
            )

            image = ImageReader(str(result_image_path))
            image_width, image_height = image.getSize()

            max_width = width - 84
            max_height = height - 165
            scale = min(
                max_width / image_width,
                max_height / image_height,
            )

            display_width = image_width * scale
            display_height = image_height * scale

            x_position = (width - display_width) / 2
            y_position = (height - 105 - display_height) / 2

            c.drawImage(
                image,
                x_position,
                y_position,
                width=display_width,
                height=display_height,
                preserveAspectRatio=True,
                mask="auto",
            )

    c.setFillColor(colors.HexColor("#64748B"))
    c.setFont("Helvetica", 7.5)
    c.drawString(
        42,
        42,
        "Preliminary AI visual screening only. Not an official condition rating or structural safety certification.",
    )

    c.save()

    return FileResponse(
        str(filepath),
        media_type="application/pdf",
        filename=filename,
    )