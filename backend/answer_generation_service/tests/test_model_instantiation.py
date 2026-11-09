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

"""Security tests for model/model_params allowlisting."""

import importlib
import os
import sys
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVICE_DIR = os.path.dirname(_HERE)
_BACKEND_DIR = os.path.dirname(_SERVICE_DIR)


def _stub(name, **attrs):
  mod = types.ModuleType(name)
  mod.__path__ = []
  for k, v in attrs.items():
    setattr(mod, k, v)
  return mod


def _import_main():
  """Imports answer_generation_service/main.py with all GCP deps stubbed."""
  fake_vertexai = mock.MagicMock(name="VertexAI")
  stubs = {
      "flask": _stub("flask", jsonify=lambda payload: payload),
      "functions_framework": _stub("functions_framework", http=lambda f: f),
      "google.cloud.firestore": _stub("google.cloud.firestore", Client=mock.MagicMock()),
      "google.cloud.aiplatform": _stub("google.cloud.aiplatform"),
      "google.cloud.aiplatform.matching_engine": _stub(
          "google.cloud.aiplatform.matching_engine"
      ),
      "google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint": _stub(
          "google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint",
          Namespace=mock.MagicMock(),
      ),
      "langchain": _stub("langchain"),
      "langchain.retrievers": _stub("langchain.retrievers"),
      "langchain.retrievers.contextual_compression": _stub(
          "langchain.retrievers.contextual_compression",
          ContextualCompressionRetriever=mock.MagicMock(),
      ),
      "langchain.retrievers.document_compressors": _stub(
          "langchain.retrievers.document_compressors",
          LLMChainFilter=mock.MagicMock(),
      ),
      "langchain.retrievers.multi_query": _stub(
          "langchain.retrievers.multi_query", MultiQueryRetriever=mock.MagicMock()
      ),
      "langchain_core": _stub("langchain_core"),
      "langchain_core.prompts": _stub(
          "langchain_core.prompts", PromptTemplate=mock.MagicMock()
      ),
      "langchain_core.runnables": _stub(
          "langchain_core.runnables",
          RunnableParallel=mock.MagicMock(),
          RunnablePassthrough=mock.MagicMock(),
          chain=lambda f: f,
      ),
      "langchain_google_community": _stub("langchain_google_community"),
      "langchain_google_community.vertex_rank": _stub(
          "langchain_google_community.vertex_rank", VertexAIRank=mock.MagicMock()
      ),
      "langchain_google_vertexai": _stub(
          "langchain_google_vertexai",
          VectorSearchVectorStoreDatastore=mock.MagicMock(),
          VertexAI=fake_vertexai,
          VertexAIEmbeddings=mock.MagicMock(),
      ),
      "nest_asyncio": _stub("nest_asyncio", apply=lambda: None),
  }
  try:
    importlib.import_module("google.cloud")
  except ImportError:
    stubs["google"] = _stub("google")
    stubs["google.cloud"] = _stub("google.cloud")
  with mock.patch.dict(sys.modules, stubs):
    google_cloud = sys.modules["google.cloud"]
    # Make `from google.cloud import firestore/aiplatform` resolve to stubs.
    with mock.patch.object(
        google_cloud, "firestore", stubs["google.cloud.firestore"], create=True
    ), mock.patch.object(
        google_cloud,
        "aiplatform",
        stubs["google.cloud.aiplatform"],
        create=True,
    ):
      mod = _load_main()
  return mod, fake_vertexai


def _load_main():
  spec = importlib.util.spec_from_file_location(
      "answer_generation_main", os.path.join(_SERVICE_DIR, "main.py")
  )
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


class _Request:

  def __init__(self, payload):
    self._payload = payload

  def get_json(self):
    return self._payload


def _payload(**overrides):
  data = {
      "retriever_name": "TopK",
      "query": "What is RAG?",
      "model_name": "gemini-1.5-flash-001",
      "model_params": {"temperature": 0.2, "max_output_tokens": 256},
      "index_name": "my-index",
  }
  data.update(overrides)
  return data


class AnswerGenerationModelAllowlistTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.main, cls.fake_vertexai = _import_main()

  def setUp(self):
    super().setUp()
    self.fake_vertexai.reset_mock()
    for name in (
        "get_vector_store_details",
        "get_vectordb",
        "get_retriever",
    ):
      p = mock.patch.object(self.main, name, return_value=mock.MagicMock())
      p.start()
      self.addCleanup(p.stop)
    p = mock.patch.object(
        self.main,
        "create_rag_answer",
        return_value=("answer", {"context": "ctx"}),
    )
    p.start()
    self.addCleanup(p.stop)

  def test_unknown_model_rejected_before_instantiation(self):
    for bad in ("projects/attacker/models/evil", "text-bison", "", None, 123):
      with self.subTest(model_name=bad):
        body, status = self.main.execute_rag_query(
            _Request(_payload(model_name=bad))
        )
        self.assertEqual(status, 400)
        self.assertIn("Unsupported model_name", body["error"])
    self.fake_vertexai.assert_not_called()

  def test_dangerous_model_params_rejected(self):
    bad_params = [
        {"api_endpoint": "attacker.example.com"},
        {"client_options": {"api_endpoint": "evil"}},
        {"credentials": "x"},
        {"project": "attacker-project"},
        {"temperature": 5},
        {"temperature": True},
        {"max_output_tokens": "100"},
        {"top_k": 1000},
        "not-a-dict",
    ]
    for params in bad_params:
      with self.subTest(model_params=params):
        body, status = self.main.execute_rag_query(
            _Request(_payload(model_params=params))
        )
        self.assertEqual(status, 400)
    self.fake_vertexai.assert_not_called()

  def test_non_object_payload_rejected(self):
    body, status = self.main.execute_rag_query(_Request(["x"]))
    self.assertEqual(status, 400)
    self.fake_vertexai.assert_not_called()

  def test_internal_errors_are_not_leaked(self):
    self.main.get_vector_store_details.side_effect = RuntimeError(
        "secret internal detail"
    )
    body, status = self.main.execute_rag_query(_Request(_payload()))
    self.assertEqual(status, 500)
    self.assertEqual(body, {"error": "Internal server error"})

  def test_allowlisted_model_and_params_work(self):
    body = self.main.execute_rag_query(
        _Request(
            _payload(
                model_params={
                    "temperature": 0.2,
                    "max_output_tokens": 256,
                    "top_p": 0.9,
                    "top_k": 20,
                }
            )
        )
    )
    self.assertEqual(body["answer"], "answer")
    self.fake_vertexai.assert_called_once_with(
        model_name="gemini-1.5-flash-001",
        temperature=0.2,
        max_output_tokens=256,
        top_p=0.9,
        top_k=20,
    )

  def test_env_override_of_model_allowlist(self):
    with mock.patch.dict(os.environ, {"ALLOWED_LLM_MODELS": "gemini-2.0-flash-001"}):
      allowed = self.main._load_allowed_models()
    self.assertEqual(allowed, {"gemini-2.0-flash-001": "gemini-2.0-flash-001"})


class RetrieverModelValidationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    try:
      import pydantic  # pylint: disable=g-import-not-at-top,unused-import
    except ImportError:
      raise unittest.SkipTest("pydantic not installed")
    spec = importlib.util.spec_from_file_location(
        "retriever_model", os.path.join(_BACKEND_DIR, "models", "retriever_model.py")
    )
    cls.rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cls.rm)

  def test_unknown_model_rejected(self):
    with self.assertRaises(Exception):
      self.rm.RetrieverModelPair(model_name="evil-model", retriever_name="TopK")

  def test_model_params_bounds(self):
    with self.assertRaises(Exception):
      self.rm.ModelParams(temperature=9)
    with self.assertRaises(Exception):
      self.rm.ModelParams(top_k=0)

  def test_valid_request_accepted(self):
    pair = self.rm.RetrieverModelPair(
        model_name="gemini-1.5-pro-001", retriever_name="TopK"
    )
    self.assertEqual(pair.model_name, "gemini-1.5-pro-001")
    self.assertEqual(
        self.rm.ModelParams(temperature=0.5, top_k=10).to_dict(),
        {"temperature": 0.5, "top_k": 10},
    )

  def test_env_override(self):
    with mock.patch.dict(os.environ, {"ALLOWED_LLM_MODELS": "gemini-2.0-flash-001"}):
      self.rm.RetrieverModelPair(
          model_name="gemini-2.0-flash-001", retriever_name="TopK"
      )
      with self.assertRaises(Exception):
        self.rm.RetrieverModelPair(
            model_name="gemini-1.5-pro-001", retriever_name="TopK"
        )


if __name__ == "__main__":
  unittest.main()
