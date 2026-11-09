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

import json
import logging
import os
import re
from typing import Any, Dict, Optional, Sequence, Tuple
import apache_beam as beam
from langchain_core.documents import Document
from langchain_google_vertexai import (
    VectorSearchVectorStoreDatastore,
    VertexAIEmbeddings,
)
from langchain_google_vertexai.vectorstores._sdk_manager import VectorSearchSDKManager
from langchain_google_vertexai.vectorstores.document_storage import (
    DataStoreDocumentStorage,
    DocumentStorage,
)

DEFAULT_ALLOWED_REGIONS: Tuple[str, ...] = ("us-central1",)
DEFAULT_ALLOWED_EMBEDDING_MODELS: Tuple[str, ...] = (
    "text-embedding-004",
    "text-embedding-preview-0815",
    "text-multilingual-embedding-002",
)

_PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_REGION_PATTERN = re.compile(r"^[a-z]+-[a-z]+[0-9]+$")
_RESOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _parse_env_allowlist(*env_vars: str) -> Tuple[str, ...]:
  """Parse a deduplicated tuple of allowlisted values from environment variables."""
  values = []
  for env_var in env_vars:
    raw = os.environ.get(env_var, "")
    if raw:
      for item in raw.split(","):
        stripped = item.strip()
        if stripped and stripped not in values:
          values.append(stripped)
  return tuple(values)


class UpsertToVectorSearchFn(beam.DoFn):
  """DoFn for upserting documents to Vector Search."""

  def __init__(
      self,
      allowed_projects: Optional[Sequence[str]] = None,
      allowed_regions: Optional[Sequence[str]] = None,
      allowed_index_ids: Optional[Sequence[str]] = None,
      allowed_endpoint_ids: Optional[Sequence[str]] = None,
      allowed_embedding_models: Optional[Sequence[str]] = None,
  ):
    super().__init__()
    self._init_allowed_projects = (
        tuple(allowed_projects) if allowed_projects is not None else None
    )
    self._init_allowed_regions = (
        tuple(allowed_regions) if allowed_regions is not None else None
    )
    self._init_allowed_index_ids = (
        tuple(allowed_index_ids) if allowed_index_ids is not None else None
    )
    self._init_allowed_endpoint_ids = (
        tuple(allowed_endpoint_ids)
        if allowed_endpoint_ids is not None
        else None
    )
    self._init_allowed_embedding_models = (
        tuple(allowed_embedding_models)
        if allowed_embedding_models is not None
        else None
    )
    self.vector_stores = {}
    self.datastore_clients = {}
    self._load_allowlists()

  def _load_allowlists(self) -> None:
    """Load trusted Vector Search destination allowlists from init args or environment."""
    self.allowed_projects = (
        self._init_allowed_projects
        if self._init_allowed_projects is not None
        else _parse_env_allowlist(
            "ALLOWED_VECTOR_STORE_PROJECTS",
            "VERTEX_AI_PROJECT_ID",
            "PROJECT_ID",
            "GOOGLE_CLOUD_PROJECT",
        )
    )
    self.allowed_regions = (
        self._init_allowed_regions
        if self._init_allowed_regions is not None
        else (
            _parse_env_allowlist(
                "ALLOWED_VECTOR_STORE_REGIONS",
                "VERTEX_AI_REGION",
                "REGION",
            )
            or DEFAULT_ALLOWED_REGIONS
        )
    )
    self.allowed_index_ids = (
        self._init_allowed_index_ids
        if self._init_allowed_index_ids is not None
        else _parse_env_allowlist(
            "ALLOWED_VECTOR_STORE_INDEX_IDS",
            "VERTEX_AI_INDEX_ID",
        )
    )
    self.allowed_endpoint_ids = (
        self._init_allowed_endpoint_ids
        if self._init_allowed_endpoint_ids is not None
        else _parse_env_allowlist(
            "ALLOWED_VECTOR_STORE_ENDPOINT_IDS",
            "VERTEX_AI_ENDPOINT_ID",
            "VERTEX_AI_INDEX_ENDPOINT_ID",
        )
    )
    self.allowed_embedding_models = (
        self._init_allowed_embedding_models
        if self._init_allowed_embedding_models is not None
        else (
            _parse_env_allowlist("ALLOWED_EMBEDDING_MODELS")
            or DEFAULT_ALLOWED_EMBEDDING_MODELS
        )
    )

  @staticmethod
  def _resolve_allowed_value(
      untrusted_value: Any,
      allowlist: Sequence[str],
      field_name: str,
      pattern: Optional[re.Pattern[str]] = None,
  ) -> str:
    """Validate an untrusted config value against a trusted allowlist and return the trusted string."""
    if not allowlist:
      raise ValueError(
          f"Trusted allowlist for '{field_name}' is not configured; "
          "refusing untrusted egress connection."
      )
    if not isinstance(untrusted_value, str) or not untrusted_value:
      raise ValueError(
          f"Invalid or missing '{field_name}' in vector_store_config."
      )
    if pattern is not None and not pattern.fullmatch(untrusted_value):
      raise ValueError(
          f"Invalid format for '{field_name}': {untrusted_value!r}."
      )
    for allowed in allowlist:
      if allowed == untrusted_value:
        return allowed
    raise ValueError(
        f"Unauthorized '{field_name}' value {untrusted_value!r} is not in the "
        "configured allowlist."
    )

  def _validate_and_resolve_config(
      self, vector_store_config: Any
  ) -> Dict[str, str]:
    """Validate vector_store_config against trusted allowlists and return trusted values."""
    if not isinstance(vector_store_config, dict):
      raise ValueError("vector_store_config must be a dictionary.")

    if not hasattr(self, "allowed_projects"):
      self._load_allowlists()

    return {
        "project_id": self._resolve_allowed_value(
            vector_store_config.get("project_id"),
            self.allowed_projects,
            "project_id",
            _PROJECT_ID_PATTERN,
        ),
        "region": self._resolve_allowed_value(
            vector_store_config.get("region"),
            self.allowed_regions,
            "region",
            _REGION_PATTERN,
        ),
        "index_id": self._resolve_allowed_value(
            vector_store_config.get("index_id"),
            self.allowed_index_ids,
            "index_id",
            _RESOURCE_ID_PATTERN,
        ),
        "endpoint_id": self._resolve_allowed_value(
            vector_store_config.get("endpoint_id"),
            self.allowed_endpoint_ids,
            "endpoint_id",
            _RESOURCE_ID_PATTERN,
        ),
        "embedding_model": self._resolve_allowed_value(
            vector_store_config.get("embedding_model"),
            self.allowed_embedding_models,
            "embedding_model",
        ),
    }

  def setup(self):
    self.vector_stores = {}
    self.datastore_clients = {}
    self._load_allowlists()

  def process(self, element):
    """Process and upsert documents to Vector Search.

    Args:
        element (tuple): A tuple containing the document key and the
          document(s).

    Yields:
        str: A status message indicating the result of the upsert operation.
    """
    key, docs = element
    base_element = None
    try:
      if not isinstance(docs, list):
        docs = [docs]

      base_element = docs[0].metadata["base_element"]
      vector_store_config = base_element["vector_store"]["config"]
      trusted_config = self._validate_and_resolve_config(vector_store_config)

      if key not in self.vector_stores:
        DATASTORE_DB_NAME = ""

        sdk_manager = VectorSearchSDKManager(
            project_id=trusted_config["project_id"],
            region=trusted_config["region"],
        )

        self.datastore_clients[key] = sdk_manager.get_datastore_client(
            database=DATASTORE_DB_NAME
        )

        self.vector_stores[key] = (
            VectorSearchVectorStoreDatastore.from_components(
                project_id=trusted_config["project_id"],
                region=trusted_config["region"],
                index_id=trusted_config["index_id"],
                endpoint_id=trusted_config["endpoint_id"],
                embedding=VertexAIEmbeddings(
                    model_name=trusted_config["embedding_model"]
                ),
                stream_update=True,
            )
        )

        class CustomDataStoreDocumentStorage(
            DataStoreDocumentStorage, DocumentStorage
        ):

          def mset(
              self, key_value_pairs: Sequence[Tuple[str, Document]]
          ) -> None:
            logging.info("setting override mset")
            ids = [key for key, _ in key_value_pairs]
            documents = [document for _, document in key_value_pairs]

            with self._client.transaction():
              keys = [self._client.key(self._kind, id_) for id_ in ids]
              entities = []
              for key, document in zip(keys, documents):
                entity = self._client.entity(
                    key=key,
                    exclude_from_indexes=(
                        self._text_property_name,
                        self._metadata_property_name,
                    ),
                )
                entity[self._text_property_name] = document.page_content
                entity[self._metadata_property_name] = document.metadata
                entities.append(entity)

              self._client.put_multi(entities)

        self.vector_stores[key]._document_storage = (
            CustomDataStoreDocumentStorage(
                datastore_client=self.datastore_clients[key]
            )
        )

      vector_store = self.vector_stores[key]

      texts = [doc.page_content for doc in docs]

      # Modify the metadata
      metadatas = []
      for doc in docs:
        metadata = (
            doc.metadata.copy()
        )  # Create a copy to avoid modifying the original
        if "base_element" in metadata:
          del metadata["base_element"]  # Remove base_element
        metadata["index_name"] = vector_store_config.get(
            "index_name"
        )  # Add index_name
        metadatas.append(metadata)

      logging.info(
          f"Preparing to upsert {len(texts)} texts for document: {key}"
      )
      logging.info(f"First text (truncated to 200 chars): {texts[0][:200]}")
      logging.info(f"First metadata: {json.dumps(metadatas[0], indent=2)}")

      vector_store.add_texts(
          texts=texts,
          metadatas=metadatas,
          is_complete_overwrite=False,
      )
      logging.info(
          f"Successfully upserted {len(texts)} splits for document: {key}"
      )
      yield (True, base_element)
    except Exception as e:
      logging.error(
          f"Error upserting to Vector Search for document {key}: {str(e)}"
      )
      logging.exception("Full traceback:")
      yield (False, base_element)
