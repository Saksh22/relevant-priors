import logging
from fastapi import FastAPI
from app.schemas import RequestPayload, ResponsePayload, Prediction
from app.model import predict_batch_v2

logging.basicConfig(level=logging.INFO)

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=ResponsePayload)
def predict(payload: RequestPayload):
    predictions = [
    Prediction(**p)
    for p in predict_batch_v2(payload.cases)
]

    total_priors = sum(len(c.prior_studies) for c in payload.cases)

    logging.info(
        "challenge_id=%s cases=%d priors=%d",
        payload.challenge_id,
        len(payload.cases),
        total_priors,
    )
    
    return ResponsePayload(predictions=predictions)