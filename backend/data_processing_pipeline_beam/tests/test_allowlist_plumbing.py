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

"""Tests that security allowlists flow from pipeline options to the DoFns."""

import argparse
import contextlib
import os
import pickle
import unittest
from unittest import mock

import _stubs
from custom_transforms import docai_parser
from custom_transforms import gcs_input_fn
from custom_transforms import gcs_loader_fn
from custom_transforms import upsert_vector_search_fn
import main as pipeline_main

_PROCESSOR = "projects/123456/locations/us/processors/abc123"
_ENV_KEYS = (
    "ALLOWED_GCS_BUCKETS",
    "RAG_GCS_BUCKET",
    "ALLOWED_GCS_PREFIXES",
    "ALLOWED_GCP_PROJECTS",
    "GOOGLE_CLOUD_PROJECT",
    "PROJECT_ID",
    "ALLOWED_DOCAI_LOCATIONS",
    "ALLOWED_DOCAI_PROCESSORS",
    "ALLOWED_VECTOR_STORE_PROJECTS",
    "VERTEX_AI_PROJECT_ID",
    "ALLOWED_VECTOR_STORE_REGIONS",
    "VERTEX_AI_REGION",
    "REGION",
    "ALLOWED_VECTOR_STORE_INDEX_IDS",
    "VERTEX_AI_INDEX_ID",
    "ALLOWED_VECTOR_STORE_ENDPOINT_IDS",
    "VERTEX_AI_ENDPOINT_ID",
    "VERTEX_AI_INDEX_ENDPOINT_ID",
    "ALLOWED_EMBEDDING_MODELS",
)
_ARGV = [
    "--input_subscription=projects/my-project/subscriptions/in",
    "--output_topic=projects/my-project/topics/out",
    "--allowed_gcs_buckets=rag-docs, rag-output",
    "--allowed_gcp_projects=my-project",
    "--allowed_gcs_prefixes=docs/",
    f"--allowed_docai_processors={_PROCESSOR}",
    "--allowed_docai_locations=us",
    "--allowed_vector_store_projects=my-project",
    "--allowed_vector_store_regions=us-central1",
    "--allowed_vector_store_index_ids=7028698449302257664",
    "--allowed_vector_store_endpoint_ids=6309811358783242240",
    "--allowed_embedding_models=text-embedding-004",
]


def _parse_options(argv):
  """Parses argv into CustomPipelineOptions (real Beam when available)."""
  if _stubs.REAL_BEAM:
    from apache_beam.options.pipeline_options import PipelineOptions  # pylint: disable=g-import-not-at-top
    return PipelineOptions(argv).view_as(pipeline_main.CustomPipelineOptions)
  parser = argparse.ArgumentParser()
  pipeline_main.CustomPipelineOptions._add_argparse_args(parser)
  return parser.parse_known_args(argv)[0]


def _roundtrip(dofn):
  """Serializes a DoFn the way it is shipped to Dataflow workers."""
  if _stubs.REAL_BEAM:
    from apache_beam.internal import pickler  # pylint: disable=g-import-not-at-top
    return pickler.loads(pickler.dumps(dofn))
  return pickle.loads(pickle.dumps(dofn))


class _CleanEnvTestCase(unittest.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ, {}, clear=False)
    patcher.start()
    self.addCleanup(patcher.stop)
    for k in _ENV_KEYS:
      os.environ.pop(k, None)


class PipelineOptionsPlumbingTest(_CleanEnvTestCase):

  def test_options_reach_each_dofn_and_survive_pickling(self):
    allowlists = pipeline_main.security_allowlists(_parse_options(_ARGV))

    gcs_in = _roundtrip(gcs_input_fn.GCSInputFn(**allowlists["GCSInputFn"]))
    self.assertEqual(gcs_in.allowed_buckets, ("rag-docs", "rag-output"))
    self.assertEqual(gcs_in.allowed_projects, ("my-project",))
    self.assertEqual(gcs_in.allowed_prefixes, ("docs/",))

    loader = _roundtrip(gcs_loader_fn.GCSLoaderFn(**allowlists["GCSLoaderFn"]))
    self.assertEqual(loader.allowed_projects, ("my-project",))
    self.assertEqual(loader.allowed_locations, ("us",))
    self.assertEqual(loader.allowed_processors, (_PROCESSOR,))
    self.assertEqual(loader.allowed_buckets, ("rag-docs", "rag-output"))

    upsert = _roundtrip(
        upsert_vector_search_fn.UpsertToVectorSearchFn(
            **allowlists["UpsertToVectorSearchFn"]
        )
    )
    upsert.setup()  # Runs on the worker; must not drop the pickled values.
    self.assertEqual(upsert.allowed_projects, ("my-project",))
    self.assertEqual(upsert.allowed_regions, ("us-central1",))
    self.assertEqual(upsert.allowed_index_ids, ("7028698449302257664",))
    self.assertEqual(upsert.allowed_endpoint_ids, ("6309811358783242240",))
    self.assertEqual(upsert.allowed_embedding_models, ("text-embedding-004",))

  def test_unset_or_empty_options_fall_back_to_env(self):
    argv = _ARGV[:2] + ["--allowed_gcs_buckets="]
    allowlists = pipeline_main.security_allowlists(_parse_options(argv))
    for dofn_kwargs in allowlists.values():
      for value in dofn_kwargs.values():
        self.assertIsNone(value)

  @unittest.skipUnless(_stubs.REAL_BEAM, "needs apache_beam")
  def test_run_passes_allowlists_to_pardo(self):
    import apache_beam as beam  # pylint: disable=g-import-not-at-top

    captured = []
    real_pardo = beam.ParDo

    def recording_pardo(fn, *args, **kwargs):
      captured.append(fn)
      return real_pardo(fn, *args, **kwargs)

    class _NoRunPipeline(beam.Pipeline):

      def run(self, test_runner_api="AUTO"):
        return mock.MagicMock()

    with mock.patch.object(
        pipeline_main.beam, "ParDo", recording_pardo
    ), mock.patch.object(pipeline_main.beam, "Pipeline", _NoRunPipeline):
      pipeline_main.run(_ARGV)

    by_type = {type(fn).__name__: fn for fn in captured}
    self.assertEqual(
        by_type["GCSInputFn"].allowed_buckets, ("rag-docs", "rag-output")
    )
    self.assertEqual(by_type["GCSLoaderFn"].allowed_processors, (_PROCESSOR,))
    self.assertEqual(
        by_type["UpsertToVectorSearchFn"].allowed_index_ids,
        ("7028698449302257664",),
    )


class GCSInputFnPlumbingTest(_CleanEnvTestCase):

  def setUp(self):
    super().setUp()
    p = mock.patch("google.cloud.storage.Client")
    self.client_cls = p.start()
    self.addCleanup(p.stop)
    blob = mock.Mock()
    blob.name = "docs/a.pdf"
    blob.bucket.name = "rag-docs"
    self.client_cls.return_value.list_blobs.return_value = [blob]

  def _element(self, bucket="rag-docs", prefix="docs/"):
    return {
        "input": {
            "config": {
                "project_name": "my-project",
                "bucket_name": bucket,
                "prefix": prefix,
            }
        }
    }

  def _run(self, fn, element, expect_error=False):
    ctx = (
        self.assertLogs(level="ERROR")
        if expect_error
        else contextlib.nullcontext()
    )
    with ctx:
      return list(fn.process(element))

  def test_constructor_args_used_without_env(self):
    fn = _roundtrip(
        gcs_input_fn.GCSInputFn(
            allowed_buckets=["rag-docs"],
            allowed_projects=["my-project"],
            allowed_prefixes=["docs"],
        )
    )
    self.assertEqual(len(self._run(fn, self._element())), 1)
    self.assertEqual(
        self._run(fn, self._element(prefix="private/"), expect_error=True), []
    )
    self.assertEqual(
        self._run(fn, self._element(bucket="victim"), expect_error=True), []
    )

  def test_env_fallback(self):
    os.environ["ALLOWED_GCS_BUCKETS"] = "rag-docs"
    fn = gcs_input_fn.GCSInputFn()
    self.assertEqual(len(self._run(fn, self._element())), 1)

  def test_fail_closed_without_args_or_env(self):
    fn = gcs_input_fn.GCSInputFn()
    self.assertEqual(self._run(fn, self._element(), expect_error=True), [])
    self.client_cls.assert_not_called()


class GCSLoaderFnPlumbingTest(_CleanEnvTestCase):

  def setUp(self):
    super().setUp()
    p = mock.patch.object(gcs_loader_fn, "DocAIParser")
    self.parser_cls = p.start()
    self.addCleanup(p.stop)
    self.parser_cls.return_value.online_process.return_value = []

  def _element(self, blob_path="gs://rag-docs/a.pdf", **overrides):
    cfg = {
        "project_id": "my-project",
        "location": "us",
        "processor_name": _PROCESSOR,
        "gcs_output_path": "",
    }
    cfg.update(overrides)
    blob = mock.Mock()
    blob.path = blob_path
    return {
        "blob": blob,
        "base_element": {
            "data_loader": {"config": cfg},
            "document_splitter": {"config": {"chunk_size": 500}},
            "vector_store": {"config": {"index_name": "idx"}},
        },
    }

  def _run(self, fn, element, expect_error=False):
    fn.setup()
    ctx = (
        self.assertLogs(level="ERROR")
        if expect_error
        else contextlib.nullcontext()
    )
    with ctx:
      return list(fn.process(element))

  def test_constructor_args_used_and_passed_to_parser(self):
    fn = _roundtrip(
        gcs_loader_fn.GCSLoaderFn(
            allowed_projects=["my-project"],
            allowed_locations=["us"],
            allowed_processors=[_PROCESSOR],
            allowed_buckets=["rag-docs"],
        )
    )
    self._run(fn, self._element())
    self.parser_cls.assert_called_once_with(
        project_id="my-project",
        location="us",
        processor_name=_PROCESSOR,
        gcs_output_path=None,  # Empty string from the frontend means unset.
        allowed_projects=("my-project",),
        allowed_locations=("us",),
        allowed_processors=(_PROCESSOR,),
        allowed_buckets=("rag-docs",),
    )

  def test_constructor_args_override_env(self):
    os.environ.update({
        "ALLOWED_GCP_PROJECTS": "my-project",
        "ALLOWED_DOCAI_PROCESSORS": _PROCESSOR,
        "ALLOWED_GCS_BUCKETS": "rag-docs",
    })
    fn = gcs_loader_fn.GCSLoaderFn(
        allowed_processors=["projects/1/locations/us/processors/other"]
    )
    self.assertEqual(self._run(fn, self._element(), expect_error=True), [])
    self.parser_cls.assert_not_called()

  def test_env_fallback(self):
    os.environ.update({
        "ALLOWED_GCP_PROJECTS": "my-project",
        "ALLOWED_DOCAI_PROCESSORS": _PROCESSOR,
        "ALLOWED_GCS_BUCKETS": "rag-docs",
    })
    self._run(gcs_loader_fn.GCSLoaderFn(), self._element())
    self.parser_cls.assert_called_once()

  def test_fail_closed_without_args_or_env(self):
    fn = gcs_loader_fn.GCSLoaderFn()
    self.assertEqual(self._run(fn, self._element(), expect_error=True), [])
    self.parser_cls.assert_not_called()


class DocAIParserPlumbingTest(_CleanEnvTestCase):

  def test_parser_uses_its_own_allowlists(self):
    client = mock.MagicMock()
    parser = docai_parser.DocAIParser(
        client=client,
        processor_name=_PROCESSOR,
        gcs_output_path="",
        allowed_processors=[_PROCESSOR],
        allowed_buckets=["rag-docs"],
    )
    self.assertIsNone(parser._gcs_output_path)
    self.assertEqual(
        parser._validate_gcs_uri("gs://rag-docs/a.pdf"), "gs://rag-docs/a.pdf"
    )
    with self.assertRaisesRegex(ValueError, "allowlist"):
      parser._validate_gcs_uri("gs://victim/a.pdf")
    with self.assertRaisesRegex(ValueError, "allowlist"):
      docai_parser.DocAIParser(
          client=client,
          processor_name="projects/1/locations/us/processors/evil",
          allowed_processors=[_PROCESSOR],
      )


class UpsertPlumbingTest(_CleanEnvTestCase):

  def test_constructor_args_survive_setup_and_env_fallback(self):
    os.environ["ALLOWED_VECTOR_STORE_PROJECTS"] = "env-project"
    fn = _roundtrip(
        upsert_vector_search_fn.UpsertToVectorSearchFn(
            allowed_projects=["arg-project"]
        )
    )
    fn.setup()
    self.assertEqual(fn.allowed_projects, ("arg-project",))
    fn = upsert_vector_search_fn.UpsertToVectorSearchFn()
    fn.setup()
    self.assertEqual(fn.allowed_projects, ("env-project",))


if __name__ == "__main__":
  unittest.main()
