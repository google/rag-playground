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
from typing import Collection, Optional
import apache_beam as beam
from custom_transforms.custom_gcs_directory_loader import (
    CustomGCSDirectoryLoader,
    validate_gcs_config,
)


def _as_tuple(values: Optional[Collection[str]]) -> Optional[tuple]:
  return tuple(values) if values is not None else None


class LogInputFn(beam.DoFn):
  """DoFn for logging input elements."""

  def process(self, element):
    """Log the received input element.

    Args:
        element: The input element to log.

    Yields:
        The input element unchanged.
    """
    logging.info(f"Received input: {element}")
    yield element


class GCSInputFn(beam.DoFn):
  """DoFn for processing GCS input elements."""

  def __init__(
      self,
      allowed_buckets: Optional[Collection[str]] = None,
      allowed_projects: Optional[Collection[str]] = None,
      allowed_prefixes: Optional[Collection[str]] = None,
  ):
    """Initializes the DoFn.

    Allowlists are stored on the instance so they are pickled to Dataflow
    workers. When an allowlist is None, the matching environment variable
    (ALLOWED_GCS_BUCKETS, ALLOWED_GCP_PROJECTS, ALLOWED_GCS_PREFIXES) is used.
    """
    super().__init__()
    self.allowed_buckets = _as_tuple(allowed_buckets)
    self.allowed_projects = _as_tuple(allowed_projects)
    self.allowed_prefixes = _as_tuple(allowed_prefixes)

  def _validate_element(self, element: dict) -> dict:
    """Validate incoming PubSub message element before accessing GCS."""
    if not isinstance(element, dict):
      raise ValueError("Input element must be a dictionary.")
    input_section = element.get("input")
    if not isinstance(input_section, dict):
      raise ValueError("Missing or invalid 'input' section in message.")
    input_config = input_section.get("config")
    if not isinstance(input_config, dict):
      raise ValueError("Missing or invalid 'input.config' section in message.")

    project_name = input_config.get("project_name", "")
    bucket_name = input_config.get("bucket_name", "")
    prefix = input_config.get("prefix", "")

    validate_gcs_config(
        project_name=project_name,
        bucket=bucket_name,
        prefix=prefix,
        allowed_buckets=self.allowed_buckets,
        allowed_projects=self.allowed_projects,
        allowed_prefixes=self.allowed_prefixes,
    )

    return input_config

  def process(self, element):
    """Process a GCS input element.

    Args:
        element (dict): The input element containing GCS configuration.

    Yields:
        dict: A dictionary containing the processed GCS blob and base element.
    """
    try:
      input_config = self._validate_element(element)

      loader = CustomGCSDirectoryLoader(
          project_name=input_config["project_name"],
          bucket=input_config["bucket_name"],
          prefix=input_config.get("prefix", ""),
          allowed_buckets=self.allowed_buckets,
          allowed_projects=self.allowed_projects,
          allowed_prefixes=self.allowed_prefixes,
      )

      doc_blobs = loader.load()
      for blob in doc_blobs:
        yield {
            "type": "gcs",
            "blob": blob,
            "base_element": element,
        }

    except Exception as e:
      logging.error(f"Error processing input: {str(e)}")
