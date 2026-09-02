"""Pydantic schemas for API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    title: str = Field(..., description="Article title/headline")
    body: str = Field(..., description="Article body text")

    @field_validator("title", "body")
    @classmethod
    def not_both_empty(cls, v: str) -> str:
        return v or ""

    def validate_english(self) -> None:
        # lightweight: if langdetect available, check
        try:
            from langdetect import detect  # type: ignore

            text = f"{self.title} {self.body}".strip()
            if len(text) > 20:
                lang = detect(text[:2000])
                if lang != "en":
                    raise ValueError(f"Non-English input detected: {lang}")
        except ImportError:
            pass
        except ValueError:
            raise
        except Exception:
            pass


class TokenScore(BaseModel):
    token: str
    score: float


class PredictResponse(BaseModel):
    label: str = Field(..., description='"real" or "fake"')
    probability_fake: float = Field(..., ge=0, le=1)
    top_features: list[TokenScore] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
