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

"""Security tests for Document AI egress validation."""

import contextlib
import os
import sys
import unittest
from unittest import mock

import _stubs  # pylint: disable=unused-import
from custom_transforms import docai_parser
from custom_transforms import gcs_loader_fn

_PROCESSOR = "projects/123456/locations/us/processors/abc123"
_ENV_KEYS = (
    "ALLOWED_GCP_PROJECTS",
    "PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "ALLOWED_DOCAI_LOCATIONS",
    "ALLOWED_DOCAI_PROCESSORS",
    "ALLOWED_GCS_BUCKETS",
    "RAG_GCS_BUCKET",
)
_GOOD_ENV = {
    "ALLOWED_GCP_PROJECTS": "my-project",
    "ALLOWED_DOCAI_PROCESSORS": _PROCESSOR,
    "ALLOWED_GCS_BUCKETS": "rag-docs,rag-output",
}


class _Blob:

  def __init__(self, path, mimetype="application/pdf"):
    self.path = path
    self.mimetype = mimetype


def _element(blob_path="gs://rag-docs/docs/a.pdf", **loader_overrides):
  loader_cfg = {
      "project_id": "my-project",
      "location": "us",
      "processor_name": _PROCESSOR,
      "gcs_output_path": "gs://rag-output/out",
  }
  loader_cfg.update(loader_overrides)
  return {
      "blob": _Blob(blob_path),
      "base_element": {
          "data_loader": {"config": loader_cfg},
          "document_splitter": {"config": {"chunk_size": 500}},
          "vector_store": {"config": {"index_name": "my-index"}},
      },
  }


class _EnvTestCase(unittest.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ, {}, clear=False)
    patcher.start()
    self.addCleanup(patcher.stop)
    for k in _ENV_KEYS:
      os.environ.pop(k, None)


class GCSLoaderFnSecurityTest(_EnvTestCase):

  def setUp(self):
    super().setUp()
    p = mock.patch.object(gcs_loader_fn, "DocAIParser")
    self.parser_cls = p.start()
    self.addCleanup(p.stop)
    self.parser_cls.return_value.online_process.return_value = []

  def _run(self, element, expect_error=True):
    fn = gcs_loader_fn.GCSLoaderFn()
    fn.setup()
    ctx = (
        self.assertLogs(level="ERROR")
        if expect_error
        else contextlib.nullcontext()
    )
    with ctx:
      return list(fn.process(element))

  def test_attacker_controlled_values_rejected(self):
    os.environ.update(_GOOD_ENV)
    attacks = [
        dict(location="attacker.example.com/x"),
        dict(location="evil.com#"),
        dict(location="us-central1"),  # valid region, but not allowlisted
        dict(project_id="attacker-project"),
        dict(processor_name="projects/999/locations/us/processors/evil"),
        dict(processor_name="projects/123456/locations/eu/processors/abc123"),
        dict(gcs_output_path="gs://attacker-bucket/out"),
        dict(gcs_output_path="https://attacker.example.com/out"),
        dict(gcs_output_path="gs://rag-output/../x"),
    ]
    for override in attacks:
      with self.subTest(**override):
        self.assertEqual(self._run(_element(**override)), [])
    self.parser_cls.assert_not_called()

  def test_attacker_blob_path_rejected(self):
    os.environ.update(_GOOD_ENV)
    for path in (
        "gs://victim-secrets/a.pdf",
        "gs://rag-docs/../victim/a.pdf",
        "gs://rag-docs/",
        "http://169.254.169.254/computeMetadata/v1/",
        None,
    ):
      with self.subTest(path=path):
        self.assertEqual(self._run(_element(blob_path=path)), [])
    self.parser_cls.assert_not_called()

  def test_fail_closed_when_allowlists_unset(self):
    self.assertEqual(self._run(_element()), [])
    self.parser_cls.assert_not_called()
    for fn, arg in (
        (docai_parser.validate_project_id, "my-project"),
        (docai_parser.validate_processor_name, _PROCESSOR),
        (docai_parser.validate_gcs_uri, "gs://rag-docs/a.pdf"),
    ):
      with self.subTest(fn=fn.__name__):
        with self.assertRaisesRegex(ValueError, "allowlist"):
          fn(arg)

  def test_allowlisted_config_processes_document(self):
    os.environ.update(_GOOD_ENV)
    doc = mock.Mock()
    doc.metadata = {"source": "gs://rag-docs/docs/a.pdf"}
    self.parser_cls.return_value.online_process.return_value = [doc]
    out = self._run(_element(), expect_error=False)
    self.parser_cls.assert_called_once_with(
        project_id="my-project",
        location="us",
        processor_name=_PROCESSOR,
        gcs_output_path="gs://rag-output/out",
    )
    self.assertEqual(len(out), 1)
    self.assertEqual(out[0][0], "gs://rag-docs/docs/a.pdf")
    self.assertEqual(doc.metadata["index_name"], "my-index")

  def test_object_names_with_special_chars_allowed(self):
    os.environ.update(_GOOD_ENV)
    self.assertEqual(
        docai_parser.validate_gcs_uri("gs://rag-docs/q&a #1?.pdf"),
        "gs://rag-docs/q&a #1?.pdf",
    )


class DocAIParserSecurityTest(_EnvTestCase):

  def test_location_host_injection_blocked_before_client_creation(self):
    os.environ.update(_GOOD_ENV)
    client_options = mock.MagicMock()
    stubs = {
        "google.api_core.client_options": mock.MagicMock(
            ClientOptions=client_options
        ),
        "google.cloud.documentai": mock.MagicMock(),
    }
    with mock.patch.dict(sys.modules, stubs):
      for loc in ("attacker.example.com/", "us.evil.com", "US", "us-"):
        with self.subTest(location=loc):
          with self.assertRaises(ValueError):
            docai_parser.DocAIParser(
                project_id="my-project", location=loc, processor_name=_PROCESSOR
            )
    client_options.assert_not_called()

  def test_online_process_revalidates_blob_and_processor(self):
    os.environ.update(_GOOD_ENV)
    client = mock.MagicMock()
    parser = docai_parser.DocAIParser(client=client, processor_name=_PROCESSOR)
    documentai = mock.MagicMock()
    stubs = {
        "google.cloud.documentai": documentai,
        "google.cloud.documentai_v1": mock.MagicMock(),
        "google.cloud.documentai_v1.types": mock.MagicMock(),
        "google.cloud.documentai_toolbox": mock.MagicMock(),
        "google.cloud.documentai_toolbox.wrappers": mock.MagicMock(),
        "google.cloud.documentai_toolbox.wrappers.page": mock.MagicMock(),
    }
    google_cloud = sys.modules.get("google.cloud")
    patches = [mock.patch.dict(sys.modules, stubs)]
    if google_cloud is not None:
      patches.append(
          mock.patch.object(google_cloud, "documentai", documentai, create=True)
      )
    with contextlib.ExitStack() as stack:
      for p in patches:
        stack.enter_context(p)
      with self.assertRaisesRegex(ValueError, "allowlist"):
        list(parser.online_process(_Blob("gs://victim-secrets/a.pdf")))
      client.process_document.assert_not_called()

      parser._processor_name = "projects/1/locations/us/processors/evil"
      with self.assertRaisesRegex(ValueError, "allowlist"):
        list(parser.online_process(_Blob("gs://rag-docs/a.pdf")))
      client.process_document.assert_not_called()

      parser._processor_name = _PROCESSOR
      client.process_document.return_value.document.pages = []
      list(parser.online_process(_Blob("gs://rag-docs/a.pdf")))
      client.process_document.assert_called_once()
      documentai.GcsDocument.assert_called_once_with(
          gcs_uri="gs://rag-docs/a.pdf", mime_type="application/pdf"
      )


if __name__ == "__main__":
  unittest.main()
