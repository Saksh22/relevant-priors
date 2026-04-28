from pydantic import BaseModel
from typing import List


class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str


class Case(BaseModel):
    case_id: str
    patient_id: str | None = None
    patient_name: str | None = None
    current_study: Study
    prior_studies: List[Study]


class RequestPayload(BaseModel):
    challenge_id: str
    schema_version: int
    generated_at: str | None = None
    cases: List[Case]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class ResponsePayload(BaseModel):
    predictions: List[Prediction]