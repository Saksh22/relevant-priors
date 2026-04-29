import logging
from fastapi import FastAPI
from app.schemas import RequestPayload, ResponsePayload, Prediction
from app.model import predict_one

logging.basicConfig(level=logging.INFO)

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=ResponsePayload)
def predict(payload: RequestPayload):
    predictions = []

    total_priors = sum(len(c.prior_studies) for c in payload.cases)

    logging.info(
        "challenge_id=%s cases=%d priors=%d",
        payload.challenge_id,
        len(payload.cases),
        total_priors,
    )

    for case in payload.cases:
        current_desc = case.current_study.study_description

        for prior in case.prior_studies:
            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=predict_one(
                        current_desc,
                        prior.study_description,
                    ),
                )
            )

    return ResponsePayload(predictions=predictions)