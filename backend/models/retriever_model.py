# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, FrozenSet, List, Optional
from pydantic import BaseModel, Field, field_validator

# Must match the allowlist enforced by answer_generation_service. Deployments
# can override it with a comma-separated ALLOWED_LLM_MODELS env variable.
DEFAULT_ALLOWED_MODELS = (
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
)


def allowed_model_names() -> FrozenSet[str]:
  configured = [
      m.strip()
      for m in os.environ.get("ALLOWED_LLM_MODELS", "").split(",")
      if m.strip()
  ]
  return frozenset(configured or DEFAULT_ALLOWED_MODELS)


class ModelParams(BaseModel):
  temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
  max_output_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
  top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
  top_k: Optional[int] = Field(default=None, ge=1, le=40)

  def to_dict(self):
    return {k: v for k, v in self.dict().items() if v is not None}


class RetrieverModelPair(BaseModel):
  model_name: str
  retriever_name: str

  @field_validator("model_name")
  @classmethod
  def _check_model_name(cls, v: str) -> str:
    if v not in allowed_model_names():
      raise ValueError(f"Unsupported model_name: {v}")
    return v


class CompareRetrieverRequest(BaseModel):
  queries: List[str]
  retriever_model_pairs: List[RetrieverModelPair]
  index_name: str
  model_params: Dict[str, ModelParams]
  is_rerank: bool


class CompareRetrieverResponse(BaseModel):
  results: List[Dict]
