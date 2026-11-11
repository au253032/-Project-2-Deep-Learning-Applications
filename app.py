
"""
Project 2 API: Week 6 EuroSAT image classification
and Week 7 text evidence analysis.
"""

import io
import os
import re
import time
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms


# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "week6_model.pt")

POSITIVE = os.getenv("POSITIVE", "River")

URGENT = [
    "flood",
    "fire",
    "burn",
    "landslide",
    "erosion",
    "illegal",
    "encroachment",
]

THRESHOLD = float(os.getenv("THRESHOLD", "0.3"))

API_KEY = os.getenv("API_KEY")

MAX_BYTES = 5 * 1024 * 1024

PATTERNS = {
    "cnic": r"\b\d{5}-\d{7}-\d\b",
    "mobile": r"(?:\+?92[- ]?|0)3\d{2}[- ]?\d{7}\b",
    "amount": r"(?:PKR|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)",
    "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
}

state = {}


# --------------------------------------------------
# MODEL REBUILD
# --------------------------------------------------

def rebuild(checkpoint):

    name = checkpoint["backbone"]
    number_of_classes = len(checkpoint["classes"])

    if name.startswith("resnet"):

        model = getattr(models, name)(weights=None)

        model.fc = nn.Linear(
            model.fc.in_features,
            number_of_classes
        )

    else:

        model = models.efficientnet_b0(weights=None)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            number_of_classes
        )

    model.load_state_dict(checkpoint["state_dict"])

    return model.eval()


# --------------------------------------------------
# LOAD MODEL WHEN API STARTS
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app):

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    state["ck"] = checkpoint
    state["model"] = rebuild(checkpoint)

    state["tf"] = transforms.Compose(

        (
            [transforms.Grayscale(3)]
            if checkpoint["grayscale"]
            else []
        )
        +
        [
            transforms.Resize(
                (checkpoint["img_size"],) * 2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ]
    )

    print("Model loaded successfully!")
    print("Classes:", checkpoint["classes"])

    yield

    state.clear()


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Project 2 Vision + Text API",
    description=(
        "EuroSAT image classification and "
        "text evidence analysis API"
    ),
    version="1.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# RESPONSE MODELS
# --------------------------------------------------

class Prediction(BaseModel):

    label: str

    probabilities: dict[str, float]

    latency_ms: float


class CaseRecord(BaseModel):

    prediction: Prediction

    note_masked: str

    fields: dict[str, list[str]]

    urgent_terms: list[str]

    decision: str

    reasons: list[str]

    disclaimer: str


# --------------------------------------------------
# API KEY CHECK
# --------------------------------------------------

def check_key(x_api_key):

    if API_KEY and x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------

async def read_image(file: UploadFile):

    if file.content_type not in (
        "image/jpeg",
        "image/png"
    ):

        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or PNG image"
        )

    data = await file.read()

    if len(data) > MAX_BYTES:

        raise HTTPException(
            status_code=413,
            detail="Image larger than 5 MB"
        )

    try:

        img = Image.open(io.BytesIO(data))

        img.load()

        return img

    except (
        UnidentifiedImageError,
        OSError
    ):

        raise HTTPException(
            status_code=400,
            detail="File is not a readable image"
        )


# --------------------------------------------------
# IMAGE CLASSIFICATION
# --------------------------------------------------

def classify(img):

    checkpoint = state["ck"]

    start_time = time.perf_counter()

    image_mode = (
        "L"
        if checkpoint["grayscale"]
        else "RGB"
    )

    x = state["tf"](
        img.convert(image_mode)
    )

    with torch.inference_mode():

        probabilities = F.softmax(
            state["model"](x.unsqueeze(0)),
            dim=1
        )[0].tolist()

    probs = {
        class_name: round(float(probability), 4)
        for class_name, probability in zip(
            checkpoint["classes"],
            probabilities
        )
    }

    prediction = Prediction(

        label=max(
            probs,
            key=probs.get
        ),

        probabilities=probs,

        latency_ms=round(
            (time.perf_counter() - start_time) * 1000,
            1
        ),
    )

    return prediction


# --------------------------------------------------
# PERSONAL INFORMATION MASKING
# --------------------------------------------------

def mask_pii(text):

    for key, tag in [
        ("cnic", "[CNIC]"),
        ("mobile", "[PHONE]"),
    ]:

        text = re.sub(
            PATTERNS[key],
            tag,
            text
        )

    return text


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "classes": (
            state["ck"]["classes"]
            if "ck" in state
            else []
        ),
        "positive": POSITIVE,
    }


# --------------------------------------------------
# PREDICT ENDPOINT
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=Prediction
)
async def predict(

    file: UploadFile = File(
        ...,
        description="Upload a JPEG or PNG image"
    ),

    x_api_key: str | None = Header(None),

):

    check_key(x_api_key)

    img = await read_image(file)

    result = await run_in_threadpool(
        classify,
        img
    )

    return result


# --------------------------------------------------
# ANALYZE ENDPOINT
# --------------------------------------------------

@app.post(
    "/analyze",
    response_model=CaseRecord
)
async def analyze(

    file: UploadFile = File(
        ...,
        description="Upload a JPEG or PNG image"
    ),

    note: str = Form(
        ...,
        description="Enter text evidence or incident details"
    ),

    x_api_key: str | None = Header(None),

):

    check_key(x_api_key)

    if len(note) > 2000:

        raise HTTPException(
            status_code=422,
            detail="Note longer than 2,000 characters"
        )

    img = await read_image(file)

    prediction = await run_in_threadpool(
        classify,
        img
    )

    # Mask personal information first
    safe_note = mask_pii(note)

    # Extract fields from masked text
    fields = {

        key: re.findall(
            pattern,
            safe_note,
            flags=re.IGNORECASE
        )

        for key, pattern in PATTERNS.items()
    }

    fields = {
        key: value
        for key, value in fields.items()
        if value
    }

    # Find urgent words
    urgent_terms = [

        word
        for word in URGENT
        if word in safe_note.lower()

    ]

    reasons = []

    # Image-based decision
    positive_probability = prediction.probabilities.get(
        POSITIVE,
        0
    )

    if positive_probability >= THRESHOLD:

        reasons.append(
            f"image: P({POSITIVE}) = "
            f"{positive_probability:.2f}"
        )

    # Text-based decision
    if urgent_terms:

        reasons.append(
            "note mentions: "
            + ", ".join(urgent_terms)
        )

    decision = (
        "REFER"
        if reasons
        else "ROUTINE"
    )

    return CaseRecord(

        prediction=prediction,

        note_masked=safe_note,

        fields=fields,

        urgent_terms=urgent_terms,

        decision=decision,

        reasons=reasons,

        disclaimer=(
            "Decision support only; "
            "a qualified human decides."
        ),
    )
