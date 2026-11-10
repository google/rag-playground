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

"""Security tests for Vector Search egress allowlisting."""

import os
import unittest
from unittest import mock

import _stubs  # pylint: disable=unused-import
from custom_transforms import upsert_vector_search_fn as upsert_mod

_ENV_KEYS = (
    "ALLOWED_VECTOR_STORE_PROJECTS",
    "VERTEX_AI_PROJECT_ID",
    "PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
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

_GOOD_ENV = {
    "ALLOWED_VECTOR_STORE_PROJECTS": "my-project",
    "ALLOWED_VECTOR_STORE_REGIONS": "us-central1",
    "ALLOWED_VECTOR_STORE_INDEX_IDS": "7028698449302257664",
    "ALLOWED_VECTOR_STORE_ENDPOINT_IDS": "6309811358783242240",
}


def _vs_config(**overrides):
  cfg = {
      "project_id": "my-project",
      "region": "us-central1",
      "index_id": "7028698449302257664",
      "endpoint_id": "6309811358783242240",
      "embedding_model": "text-embedding-004",
      "index_name": "my-index",
  }
  cfg.update(overrides)
  return cfg


def _element(vs_config):
  doc = mock.Mock()
  doc.page_content = "hello world"
  doc.metadata = {"base_element": {"vector_store": {"config": vs_config}}}
  return ("doc-key", [doc])


class UpsertVectorSearchSecurityTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    patcher = mock.patch.dict(os.environ, {}, clear=False)
    patcher.start()
    self.addCleanup(patcher.stop)
    for k in _ENV_KEYS:
      os.environ.pop(k, None)
    self.sdk = self._patch("VectorSearchSDKManager")
    self.datastore = self._patch("VectorSearchVectorStoreDatastore")
    self.embeddings = self._patch("VertexAIEmbeddings")

  def _patch(self, name):
    p = mock.patch.object(upsert_mod, name, create=True)
    m = p.start()
    self.addCleanup(p.stop)
    return m

  def _run(self, vs_config, **kwargs):
    fn = upsert_mod.UpsertToVectorSearchFn(**kwargs)
    fn.setup()
    with self.assertLogs(level="INFO"):
      return list(fn.process(_element(vs_config)))

  def _assert_no_egress(self):
    self.sdk.assert_not_called()
    self.datastore.from_components.assert_not_called()
    self.embeddings.assert_not_called()

  def test_attacker_destination_rejected(self):
    os.environ.update(_GOOD_ENV)
    attacks = [
        dict(project_id="attacker-project"),
        dict(region="europe-west4"),
        dict(index_id="1111"),
        dict(endpoint_id="2222"),
        dict(embedding_model="projects/attacker/models/x"),
        dict(project_id="my-project/../attacker"),
        dict(region="us-central1.attacker.com"),
        dict(project_id=None),
    ]
    for override in attacks:
      with self.subTest(**{k: str(v) for k, v in override.items()}):
        out = self._run(_vs_config(**override))
        self.assertEqual(out[0][0], False)
    self._assert_no_egress()

  def test_fail_closed_when_allowlists_unset(self):
    out = self._run(_vs_config())
    self.assertEqual(out[0][0], False)
    self._assert_no_egress()
    with self.assertRaisesRegex(ValueError, "not configured"):
      upsert_mod.UpsertToVectorSearchFn()._validate_and_resolve_config(
          _vs_config()
      )

  def test_partial_allowlist_still_fails_closed(self):
    os.environ["ALLOWED_VECTOR_STORE_PROJECTS"] = "my-project"
    out = self._run(_vs_config())
    self.assertEqual(out[0][0], False)
    self._assert_no_egress()

  def test_allowlisted_destination_upserts(self):
    os.environ.update(_GOOD_ENV)
    out = self._run(_vs_config())
    self.assertEqual(out[0][0], True)
    self.sdk.assert_called_once_with(project_id="my-project", region="us-central1")
    kwargs = self.datastore.from_components.call_args.kwargs
    self.assertEqual(kwargs["project_id"], "my-project")
    self.assertEqual(kwargs["index_id"], "7028698449302257664")
    self.assertEqual(kwargs["endpoint_id"], "6309811358783242240")
    self.embeddings.assert_called_once_with(model_name="text-embedding-004")
    self.datastore.from_components.return_value.add_texts.assert_called_once()

  def test_constructor_allowlists_take_precedence(self):
    os.environ.update(_GOOD_ENV)
    out = self._run(
        _vs_config(), allowed_projects=["other-project"]
    )
    self.assertEqual(out[0][0], False)
    self._assert_no_egress()

  def test_returned_values_come_from_trusted_allowlist(self):
    fn = upsert_mod.UpsertToVectorSearchFn(
        allowed_projects=["my-project"],
        allowed_regions=["us-central1"],
        allowed_index_ids=["7028698449302257664"],
        allowed_endpoint_ids=["6309811358783242240"],
    )
    trusted = fn._validate_and_resolve_config(_vs_config())
    self.assertEqual(
        trusted,
        {
            "project_id": "my-project",
            "region": "us-central1",
            "index_id": "7028698449302257664",
            "endpoint_id": "6309811358783242240",
            "embedding_model": "text-embedding-004",
        },
    )


if __name__ == "__main__":
  unittest.main()
