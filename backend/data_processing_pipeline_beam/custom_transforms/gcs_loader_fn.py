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
from custom_transforms.docai_parser import (
    DocAIParser,
    validate_gcs_uri,
    validate_location,
    validate_processor_name,
    validate_project_id,
)


def _as_tuple(values: Optional[Collection[str]]) -> Optional[tuple]:
  return tuple(values) if values is not None else None


class GCSLoaderFn(beam.DoFn):

  def __init__(
      self,
      allowed_projects: Optional[Collection[str]] = None,
      allowed_locations: Optional[Collection[str]] = None,
      allowed_processors: Optional[Collection[str]] = None,
      allowed_buckets: Optional[Collection[str]] = None,
  ):
    """Initializes the DoFn.

    Allowlists are stored on the instance so they are pickled to Dataflow
    workers. A None allowlist falls back to the matching environment variable
    (ALLOWED_GCP_PROJECTS, ALLOWED_DOCAI_LOCATIONS, ALLOWED_DOCAI_PROCESSORS,
    ALLOWED_GCS_BUCKETS).
    """
    super().__init__()
    self.allowed_projects = _as_tuple(allowed_projects)
    self.allowed_locations = _as_tuple(allowed_locations)
    self.allowed_processors = _as_tuple(allowed_processors)
    self.allowed_buckets = _as_tuple(allowed_buckets)

  def setup(self):
    """DoFn for loading and processing GCS documents."""
    self.parsers = {}

  def process(self, element):
    """Process a GCS document element.

    Args:
        element (dict): The input element containing GCS blob and configuration.

    Yields:
        tuple: A tuple containing the document source and the processed
        document.
    """
    try:
      base_element = element["base_element"]
      data_loader_config = base_element["data_loader"]["config"]
      document_splitter_config = base_element["document_splitter"]["config"]
      vector_store_config = base_element["vector_store"]["config"]

      project_id = validate_project_id(
          data_loader_config.get("project_id"), self.allowed_projects
      )
      location = validate_location(
          data_loader_config.get("location"), self.allowed_locations
      )
      processor_name = validate_processor_name(
          data_loader_config.get("processor_name"),
          expected_location=location,
          allowed_processors=self.allowed_processors,
      )
      # The frontend sends "" when no output path is entered.
      gcs_output_path = data_loader_config.get("gcs_output_path") or None
      if gcs_output_path is not None:
        gcs_output_path = validate_gcs_uri(
            gcs_output_path,
            allow_prefix_only=True,
            allowed_buckets=self.allowed_buckets,
        )

      blob = element["blob"]
      validate_gcs_uri(
          getattr(blob, "path", None), allowed_buckets=self.allowed_buckets
      )

      parser_key = (
          project_id,
          location,
          processor_name,
      )
      if parser_key not in self.parsers:
        self.parsers[parser_key] = DocAIParser(
            project_id=project_id,
            location=location,
            processor_name=processor_name,
            gcs_output_path=gcs_output_path,
            allowed_projects=self.allowed_projects,
            allowed_locations=self.allowed_locations,
            allowed_processors=self.allowed_processors,
            allowed_buckets=self.allowed_buckets,
        )

      parser = self.parsers[parser_key]
      chunk_size = document_splitter_config["chunk_size"]
      index_name = vector_store_config["index_name"]

      docs = list(
          parser.online_process(
              blob,
              chunk_size=chunk_size,
              include_ancestor_headings=True,
          )
      )
      logging.info(f"Loaded {len(docs)} documents from GCS")

      for doc in docs:
        doc.metadata["index_name"] = index_name
        doc.metadata["input_type"] = "gcs"
        doc.metadata["base_element"] = base_element
        source = doc.metadata.get("source", f"gcs_{blob}")
        yield (source, doc)

    except Exception as e:
      logging.error(f"Error processing GCS document: {str(e)}")
