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

"""Security tests for GCS bucket validation of PubSub input."""

import contextlib
import os
import unittest
from unittest import mock

import _stubs  # pylint: disable=unused-import
from custom_transforms import custom_gcs_directory_loader as loader_mod
from custom_transforms.custom_gcs_directory_loader import CustomGCSDirectoryLoader
from custom_transforms.gcs_input_fn import GCSInputFn

_ENV_KEYS = (
    "ALLOWED_GCS_BUCKETS",
    "RAG_GCS_BUCKET",
    "ALLOWED_GCS_PREFIXES",
    "ALLOWED_GCP_PROJECTS",
    "GOOGLE_CLOUD_PROJECT",
    "PROJECT_ID",
)


def _element(bucket="rag-docs", project="my-project", prefix="docs/"):
  return {
      "input": {
          "type": "gcs",
          "config": {
              "project_name": project,
              "bucket_name": bucket,
              "prefix": prefix,
          },
      },
  }


def _fake_blob(bucket, name):
  blob = mock.Mock()
  blob.name = name
  blob.bucket.name = bucket
  return blob


class GCSInputSecurityTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ, {}, clear=False)
    patcher.start()
    self.addCleanup(patcher.stop)
    for k in _ENV_KEYS:
      os.environ.pop(k, None)
    storage_patcher = mock.patch("google.cloud.storage.Client")
    self.mock_client_cls = storage_patcher.start()
    self.addCleanup(storage_patcher.stop)

  def _run(self, element, expect_error=False, **kwargs):
    ctx = (
        self.assertLogs(level="ERROR")
        if expect_error
        else contextlib.nullcontext()
    )
    with ctx:
      return list(GCSInputFn(**kwargs).process(element))

  def test_attacker_bucket_rejected(self):
    os.environ["ALLOWED_GCS_BUCKETS"] = "rag-docs"
    out = self._run(_element(bucket="victim-secrets"), expect_error=True)
    self.assertEqual(out, [])
    self.mock_client_cls.assert_not_called()

  def test_fail_closed_when_allowlist_unset(self):
    out = self._run(_element(), expect_error=True)
    self.assertEqual(out, [])
    self.mock_client_cls.assert_not_called()
    with self.assertRaisesRegex(ValueError, "No allowed GCS buckets"):
      loader_mod.validate_gcs_config("my-project", "rag-docs")

  def test_invalid_formats_and_traversal_rejected(self):
    os.environ["ALLOWED_GCS_BUCKETS"] = "rag-docs"
    bad = [
        dict(project="my-project", bucket="rag-docs/../x", prefix=""),
        dict(project="my-project", bucket="RAG_DOCS", prefix=""),
        dict(project="Bad Project!", bucket="rag-docs", prefix=""),
        dict(project="my-project", bucket="rag-docs", prefix="../other"),
        dict(project="my-project", bucket="rag-docs", prefix="/abs"),
        dict(project="my-project", bucket="rag-docs", prefix="a\\b"),
    ]
    for kw in bad:
      with self.subTest(**kw):
        with self.assertRaises(ValueError):
          loader_mod.validate_gcs_config(
              project_name=kw["project"], bucket=kw["bucket"], prefix=kw["prefix"]
          )
    self.mock_client_cls.assert_not_called()

  def test_project_allowlist_enforced(self):
    os.environ["ALLOWED_GCS_BUCKETS"] = "rag-docs"
    os.environ["ALLOWED_GCP_PROJECTS"] = "my-project"
    with self.assertRaisesRegex(ValueError, "Unauthorized GCP project_name"):
      loader_mod.validate_gcs_config("attacker-proj", "rag-docs")

  def test_prefix_allowlist_enforced(self):
    os.environ["ALLOWED_GCS_BUCKETS"] = "rag-docs"
    os.environ["ALLOWED_GCS_PREFIXES"] = "docs"
    loader_mod.validate_gcs_config("my-project", "rag-docs", "docs/a.pdf")
    with self.assertRaisesRegex(ValueError, "ALLOWED_GCS_PREFIXES"):
      loader_mod.validate_gcs_config("my-project", "rag-docs", "private/")

  def test_loader_rejects_bucket_mutated_after_init(self):
    loader = CustomGCSDirectoryLoader(
        project_name="my-project",
        bucket="rag-docs",
        allowed_buckets=["rag-docs"],
    )
    loader.bucket = "victim-secrets"
    with self.assertRaises(ValueError):
      loader.load()
    self.mock_client_cls.assert_not_called()

  def test_allowlisted_bucket_loads_and_skips_bad_blobs(self):
    os.environ["ALLOWED_GCS_BUCKETS"] = "rag-docs, other-bucket"
    self.mock_client_cls.return_value.list_blobs.return_value = [
        _fake_blob("rag-docs", "docs/a.pdf"),
        _fake_blob("rag-docs", "docs/"),
        _fake_blob("rag-docs", "docs/../../etc"),
        _fake_blob("evil-bucket", "docs/b.pdf"),
    ]
    out = self._run(_element())
    self.mock_client_cls.return_value.list_blobs.assert_called_once_with(
        "rag-docs", prefix="docs/"
    )
    self.assertEqual([o["blob"].path for o in out], ["gs://rag-docs/docs/a.pdf"])
    self.assertEqual(out[0]["type"], "gcs")

  def test_constructor_allowlist_overrides_env(self):
    os.environ["ALLOWED_GCS_BUCKETS"] = "victim-secrets"
    self.mock_client_cls.return_value.list_blobs.return_value = []
    out = self._run(
        _element(bucket="victim-secrets"),
        allowed_buckets=["rag-docs"],
        expect_error=True,
    )
    self.assertEqual(out, [])
    self.mock_client_cls.assert_not_called()


if __name__ == "__main__":
  unittest.main()
