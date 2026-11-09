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

import logging
import os
import re
from typing import Collection, List, Optional, Set
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.gcs_directory import GCSDirectoryLoader
from langchain_community.utilities.vertexai import get_client_info
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_GCS_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
_GCP_PROJECT_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")


def get_allowed_gcs_buckets(
    allowed_buckets: Optional[Collection[str]] = None,
) -> Set[str]:
  """Return the set of allowlisted GCS bucket names."""
  if allowed_buckets is not None:
    return {
        b.strip()
        for b in allowed_buckets
        if isinstance(b, str) and b.strip()
    }
  env_buckets = os.environ.get("ALLOWED_GCS_BUCKETS", "")
  if env_buckets.strip():
    return {b.strip() for b in env_buckets.split(",") if b.strip()}
  single_bucket = os.environ.get("RAG_GCS_BUCKET", "").strip()
  if single_bucket:
    return {single_bucket}
  return set()


def get_allowed_gcp_projects(
    allowed_projects: Optional[Collection[str]] = None,
) -> Set[str]:
  """Return the set of allowlisted GCP project IDs if configured."""
  if allowed_projects is not None:
    return {
        p.strip()
        for p in allowed_projects
        if isinstance(p, str) and p.strip()
    }
  env_projects = os.environ.get("ALLOWED_GCP_PROJECTS", "")
  if env_projects.strip():
    return {p.strip() for p in env_projects.split(",") if p.strip()}
  for env_key in ("GOOGLE_CLOUD_PROJECT", "PROJECT_ID"):
    val = os.environ.get(env_key, "").strip()
    if val:
      return {val}
  return set()


def validate_gcs_config(
    project_name: str,
    bucket: str,
    prefix: str = "",
    allowed_buckets: Optional[Collection[str]] = None,
    allowed_projects: Optional[Collection[str]] = None,
) -> None:
  """Validate GCS project, bucket, and prefix against strict allowlists."""
  if not isinstance(project_name, str) or not _GCP_PROJECT_PATTERN.fullmatch(
      project_name
  ):
    raise ValueError(f"Invalid GCP project_name format: {project_name!r}")

  allowed_projects_set = get_allowed_gcp_projects(allowed_projects)
  if allowed_projects_set and project_name not in allowed_projects_set:
    raise ValueError(
        f"Unauthorized GCP project_name {project_name!r}; not in allowlist."
    )

  if (
      not isinstance(bucket, str)
      or ".." in bucket
      or not _GCS_BUCKET_PATTERN.fullmatch(bucket)
  ):
    raise ValueError(f"Invalid GCS bucket name format: {bucket!r}")

  allowed_buckets_set = get_allowed_gcs_buckets(allowed_buckets)
  if not allowed_buckets_set:
    raise ValueError(
        "No allowed GCS buckets configured (set ALLOWED_GCS_BUCKETS); "
        "denying GCS access by default."
    )
  if bucket not in allowed_buckets_set:
    raise ValueError(
        f"Unauthorized GCS bucket {bucket!r}; not in ALLOWED_GCS_BUCKETS."
    )

  if prefix is None:
    prefix = ""
  if not isinstance(prefix, str):
    raise ValueError("Invalid GCS prefix: must be a string.")
  if (
      "\x00" in prefix
      or "\\" in prefix
      or prefix.startswith("/")
      or ".." in prefix.split("/")
  ):
    raise ValueError(f"Invalid GCS prefix {prefix!r}: path traversal detected.")

  env_prefixes = os.environ.get("ALLOWED_GCS_PREFIXES", "").strip()
  if env_prefixes:
    allowed_prefix_list = [
        p.strip() for p in env_prefixes.split(",") if p.strip()
    ]
    if allowed_prefix_list and not any(
        prefix == p or prefix.startswith(p.rstrip("/") + "/")
        for p in allowed_prefix_list
    ):
      raise ValueError(
          f"Unauthorized GCS prefix {prefix!r}; not in ALLOWED_GCS_PREFIXES."
      )


class CustomGCSDirectoryLoader(GCSDirectoryLoader, BaseLoader):

  def __init__(
      self,
      project_name: str,
      bucket: str,
      prefix: str = "",
      continue_on_failure: bool = False,
      allowed_buckets: Optional[Collection[str]] = None,
      allowed_projects: Optional[Collection[str]] = None,
  ) -> None:
    self.allowed_buckets = allowed_buckets
    self.allowed_projects = allowed_projects
    validate_gcs_config(
        project_name=project_name,
        bucket=bucket,
        prefix=prefix,
        allowed_buckets=self.allowed_buckets,
        allowed_projects=self.allowed_projects,
    )
    super().__init__(
        project_name=project_name,
        bucket=bucket,
        prefix=prefix,
        continue_on_failure=continue_on_failure,
    )

  def load(self, file_pattern=None) -> List[Document]:
    """Load documents."""
    validate_gcs_config(
        project_name=self.project_name,
        bucket=self.bucket,
        prefix=self.prefix,
        allowed_buckets=getattr(self, "allowed_buckets", None),
        allowed_projects=getattr(self, "allowed_projects", None),
    )
    try:
      from google.cloud import storage
    except ImportError:
      raise ImportError(
          "Could not import google-cloud-storage python package. "
          "Please install it with `pip install google-cloud-storage`."
      )
    client = storage.Client(
        project=self.project_name,
        client_info=get_client_info(module="google-cloud-storage"),
    )

    regex = None
    if file_pattern:
      regex = re.compile(r"{}".format(file_pattern))

    docs = []
    for blob in client.list_blobs(self.bucket, prefix=self.prefix):
      # we shall just skip directories since GCSFileLoader creates
      # intermediate directories on the fly
      if blob.name.endswith("/"):
        continue
      if ".." in blob.name.split("/") or blob.bucket.name != self.bucket:
        logger.warning(f"Skipping invalid or mismatched blob: {blob.name}")
        continue
      if regex and not regex.match(blob.name):
        continue
      # Use the try-except block here
      try:
        logger.info(f"Processing {blob.name}")
        temp_blob = Blob(path=f"gs://{self.bucket}/{blob.name}")
        docs.append(temp_blob)
      except Exception as e:
        if self.continue_on_failure:
          logger.warning(f"Problem processing blob {blob.name}, message: {e}")
          continue
        else:
          raise e
    return docs
