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
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Set

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document
from langchain_core.utils.iter import batch_iterate
from langchain_google_community._utils import get_client_info

if TYPE_CHECKING:
  from google.api_core.operation import Operation  # type: ignore[import]
  from google.cloud.documentai import (  # type: ignore[import]
      DocumentProcessorServiceClient,
  )


logger = logging.getLogger(__name__)

_PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9\-]{4,28}[a-z0-9]$")
_LOCATION_PATTERN = re.compile(r"^[a-z]{2,}(?:-[a-z]+[0-9]+)?$")
_VALID_GCP_DOCAI_LOCATIONS = frozenset({
    "us",
    "eu",
    "us-central1",
    "us-east1",
    "us-east4",
    "us-west1",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "asia-east1",
    "asia-northeast1",
    "asia-southeast1",
})
_PROCESSOR_NAME_PATTERN = re.compile(
    r"^projects/([0-9]+)/locations/([a-z0-9\-]+)/processors/([a-z0-9]+)"
    r"(?:/processorVersions/[a-zA-Z0-9\-._]+)?$"
)
_GCS_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._\-]{1,61}[a-z0-9]$")


def _parse_env_allowlist(*env_vars: str) -> Set[str]:
  """Reads a set of allowed values from one or more environment variables."""
  allowed: Set[str] = set()
  for var in env_vars:
    raw = os.environ.get(var, "")
    for item in raw.split(","):
      cleaned = item.strip()
      if cleaned:
        allowed.add(cleaned)
  return allowed


def _get_allowed_project_ids() -> Set[str]:
  return _parse_env_allowlist(
      "ALLOWED_GCP_PROJECTS",
      "PROJECT_ID",
      "GOOGLE_CLOUD_PROJECT",
  )


def _get_allowed_locations() -> Set[str]:
  configured = _parse_env_allowlist("ALLOWED_DOCAI_LOCATIONS")
  return configured if configured else {"us", "eu"}


def _get_allowed_processors() -> Set[str]:
  return _parse_env_allowlist("ALLOWED_DOCAI_PROCESSORS")


def _get_allowed_gcs_buckets() -> Set[str]:
  return _parse_env_allowlist("ALLOWED_GCS_BUCKETS", "RAG_GCS_BUCKET")


def validate_project_id(project_id: Optional[str]) -> str:
  """Validates a GCP project_id against format and configured allowlist."""
  if not project_id or not isinstance(project_id, str):
    raise ValueError("A valid GCP project_id must be provided.")
  if not _PROJECT_ID_PATTERN.fullmatch(project_id):
    raise ValueError(f"Invalid GCP project_id format: {project_id!r}")
  allowed_projects = _get_allowed_project_ids()
  if not allowed_projects or project_id not in allowed_projects:
    raise ValueError(
        f"Project ID {project_id!r} is not in the configured allowlist."
    )
  return project_id


def validate_location(location: Optional[str]) -> str:
  """Validates a Document AI location against strict format and allowlist."""
  if not location or not isinstance(location, str):
    raise ValueError("A valid Document AI location must be provided.")
  if (
      not _LOCATION_PATTERN.fullmatch(location)
      or location not in _VALID_GCP_DOCAI_LOCATIONS
  ):
    raise ValueError(f"Invalid Document AI location: {location!r}")
  allowed_locations = _get_allowed_locations()
  if location not in allowed_locations:
    raise ValueError(
        f"Document AI location {location!r} is not in the configured allowlist."
    )
  return location


def validate_processor_name(
    processor_name: Optional[str],
    expected_location: Optional[str] = None,
) -> str:
  """Validates a Document AI processor resource name and checks allowlist."""
  if not processor_name or not isinstance(processor_name, str):
    raise ValueError("A Document AI processor name should be provided.")
  match = _PROCESSOR_NAME_PATTERN.fullmatch(processor_name)
  if not match:
    raise ValueError(
        f"Processor name {processor_name} has the wrong format. If your"
        " prediction endpoint looks like"
        " https://us-documentai.googleapis.com/v1/projects/PROJECT_ID/locations/us/processors/PROCESSOR_ID:process,"
        " use only projects/PROJECT_ID/locations/us/processors/PROCESSOR_ID"
        " part."
    )
  proc_location = match.group(2)
  if expected_location and proc_location != expected_location:
    raise ValueError(
        f"Processor location {proc_location!r} does not match configured"
        f" client location {expected_location!r}."
    )
  allowed_processors = _get_allowed_processors()
  if not allowed_processors or processor_name not in allowed_processors:
    raise ValueError(
        f"Processor name {processor_name!r} is not in the configured allowlist."
    )
  return processor_name


def validate_gcs_uri(
    gcs_uri: Optional[str], allow_prefix_only: bool = False
) -> str:
  """Validates a gs:// URI format, prevents traversal, and checks bucket allowlist."""
  if not gcs_uri or not isinstance(gcs_uri, str):
    raise ValueError("A valid GCS URI must be provided.")
  if not gcs_uri.startswith("gs://"):
    raise ValueError(f"Invalid GCS URI: {gcs_uri!r}")
  bucket, _, path = gcs_uri[len("gs://") :].partition("/")
  if not _GCS_BUCKET_PATTERN.fullmatch(bucket) or ".." in bucket:
    raise ValueError(f"Invalid GCS bucket name in URI: {gcs_uri!r}")
  if "\\" in path or "\x00" in path:
    raise ValueError(f"Invalid characters in GCS URI: {gcs_uri!r}")
  segments = [seg for seg in path.split("/") if seg]
  if any(seg in (".", "..") for seg in segments):
    raise ValueError(f"Path traversal detected in GCS URI: {gcs_uri!r}")
  if not allow_prefix_only and (not segments or path.endswith("/")):
    raise ValueError(
        f"GCS URI must point to a specific object blob: {gcs_uri!r}"
    )
  allowed_buckets = _get_allowed_gcs_buckets()
  if not allowed_buckets or bucket not in allowed_buckets:
    raise ValueError(
        f"GCS bucket {bucket!r} is not in the configured allowlist."
    )
  return gcs_uri


@dataclass
class DocAIParsingResults:
  """A dataclass to store Document AI parsing results."""

  source_path: str
  parsed_path: str


class DocAIParser(BaseBlobParser):
  """`Google Cloud Document AI` parser.

  For a detailed explanation of Document AI, refer to the product documentation.
  https://cloud.google.com/document-ai/docs/overview
  """

  def __init__(
      self,
      *,
      client: Optional["DocumentProcessorServiceClient"] = None,
      project_id: Optional[str] = None,
      location: Optional[str] = None,
      gcs_output_path: Optional[str] = None,
      processor_name: Optional[str] = None,
  ) -> None:
    """Initializes the parser.

    Args:
        client: a DocumentProcessorServiceClient to use
        location: a Google Cloud location where a Document AI processor is
          located
        gcs_output_path: a path on Google Cloud Storage to store parsing results
        processor_name: full resource name of a Document AI processor or
          processor version

    You should provide either a client or location (and then a client
        would be instantiated).
    """

    if bool(client) == bool(location):
      raise ValueError(
          "You must specify either a client or a location to instantiate "
          "a client."
      )

    if location is not None:
      location = validate_location(location)
    if project_id is not None or not client:
      project_id = validate_project_id(project_id)
    if processor_name is not None or not client:
      processor_name = validate_processor_name(
          processor_name, expected_location=location
      )
    if gcs_output_path is not None:
      gcs_output_path = validate_gcs_uri(
          gcs_output_path, allow_prefix_only=True
      )

    self._location = location
    self._gcs_output_path = gcs_output_path
    self._processor_name = processor_name
    if client:
      self._client = client
      self._use_layout_parser = False
    else:
      try:
        from google.api_core.client_options import ClientOptions
        from google.cloud.documentai import DocumentProcessorServiceClient
      except ImportError as exc:
        raise ImportError(
            "Could not import google-cloud-documentai python package. "
            "Please, install docai dependency group: "
            "`pip install langchain-google-community[docai]`"
        ) from exc
      options = ClientOptions(
          quota_project_id=project_id,
          api_endpoint=f"{location}-documentai.googleapis.com",
      )
      self._client = DocumentProcessorServiceClient(
          client_options=options,
          client_info=get_client_info(module="document-ai"),
      )
      # get processor type
      self._processor_type = self._client.get_processor(
          name=processor_name
      ).type
      if self._processor_type == "LAYOUT_PARSER_PROCESSOR":
        self._use_layout_parser = True
      else:
        self._use_layout_parser = False

  def lazy_parse(self, blob: Blob) -> Iterator[Document]:
    """Parses a blob lazily.

    Args:
        blobs: a Blob to parse

    This is a long-running operation. A recommended way is to batch
        documents together and use the `batch_parse()` method.
    """
    yield from self.batch_parse([blob], gcs_output_path=self._gcs_output_path)

  def online_process(
      self,
      blob: Blob,
      enable_native_pdf_parsing: bool = True,
      field_mask: Optional[str] = None,
      page_range: Optional[List[int]] = None,
      chunk_size: int = 500,
      include_ancestor_headings: bool = True,
  ) -> Iterator[Document]:
    """Parses a blob lazily using online processing.

    Args:
        blob: a blob to parse.
        enable_native_pdf_parsing: enable pdf embedded text extraction
        field_mask: a comma-separated list of which fields to include in the
          Document AI response.
            suggested: "text,pages.pageNumber,pages.layout"
        page_range: list of page numbers to parse. If `None`, entire document
          will be parsed.
        chunk_size: the maximum number of characters per chunk
        include_ancestor_headings: whether to include ancestor headings in the
          chunks
            https://cloud.google.com/document-ai/docs/reference/rpc/google.cloud.documentai.v1beta3#chunkingconfig
    """
    try:
      from google.cloud import documentai
      from google.cloud.documentai_v1.types import (  # type: ignore[import, attr-defined]
          OcrConfig,
          ProcessOptions,
      )
    except ImportError as exc:
      raise ImportError(
          "Could not import google-cloud-documentai python package. "
          "Please, install docai dependency group: "
          "`pip install langchain-google-community[docai]`"
      ) from exc
    try:
      from google.cloud.documentai_toolbox.wrappers.page import (  # type: ignore[import]
          _text_from_layout,
      )
    except ImportError as exc:
      raise ImportError(
          "documentai_toolbox package not found, please install it with "
          "`pip install langchain-google-community[docai]`"
      ) from exc

    if self._use_layout_parser:
      layout_config = ProcessOptions.LayoutConfig(
          chunking_config=ProcessOptions.LayoutConfig.ChunkingConfig(
              chunk_size=chunk_size,
              include_ancestor_headings=include_ancestor_headings,
          )
      )
      individual_page_selector = (
          ProcessOptions.IndividualPageSelector(pages=page_range)
          if page_range
          else None
      )
      process_options = ProcessOptions(
          layout_config=layout_config,
          individual_page_selector=individual_page_selector,
      )
    else:
      ocr_config = (
          OcrConfig(enable_native_pdf_parsing=enable_native_pdf_parsing)
          if enable_native_pdf_parsing
          else None
      )
      individual_page_selector = (
          ProcessOptions.IndividualPageSelector(pages=page_range)
          if page_range
          else None
      )
      process_options = ProcessOptions(
          ocr_config=ocr_config,
          individual_page_selector=individual_page_selector,
      )

    validated_processor_name = validate_processor_name(
        self._processor_name, expected_location=self._location
    )
    validated_gcs_uri = validate_gcs_uri(getattr(blob, "path", None))

    response = self._client.process_document(
        documentai.ProcessRequest(
            name=validated_processor_name,
            gcs_document=documentai.GcsDocument(
                gcs_uri=validated_gcs_uri,
                mime_type=blob.mimetype or "application/pdf",
            ),
            process_options=process_options,
            skip_human_review=True,
            field_mask=field_mask,
        )
    )

    if self._use_layout_parser:
      yield from (
          Document(
              page_content=chunk.content,
              metadata={
                  "chunk_id": chunk.chunk_id,
                  "source": blob.path,
              },
          )
          for chunk in response.document.chunked_document.chunks
      )
    else:
      yield from (
          Document(
              page_content=_text_from_layout(
                  page.layout, response.document.text
              ),
              metadata={
                  "page": page.page_number,
                  "source": blob.path,
              },
          )
          for page in response.document.pages
      )

  def batch_parse(
      self,
      blobs: Sequence[Blob],
      gcs_output_path: Optional[str] = None,
      timeout_sec: int = 3600,
      check_in_interval_sec: int = 60,
      chunk_size: int = 500,
      include_ancestor_headings: bool = True,
  ) -> Iterator[Document]:
    """Parses a list of blobs lazily.

    Args:
        blobs: a list of blobs to parse.
        gcs_output_path: a path on Google Cloud Storage to store parsing
          results.
        timeout_sec: a timeout to wait for Document AI to complete, in seconds.
        check_in_interval_sec: an interval to wait until next check whether
          parsing operations have been completed, in seconds

    This is a long-running operation. A recommended way is to decouple
        parsing from creating LangChain Documents:
        >>> operations = parser.docai_parse(blobs, gcs_path)
        >>> parser.is_running(operations)
        You can get operations names and save them:
        >>> names = [op.operation.name for op in operations]
        And when all operations are finished, you can use their results:
        >>> operations = parser.operations_from_names(operation_names)
        >>> results = parser.get_results(operations)
        >>> docs = parser.parse_from_results(results)
    """
    output_path = gcs_output_path or self._gcs_output_path
    if not output_path:
      raise ValueError(
          "An output path on Google Cloud Storage should be provided."
      )
    operations = self.docai_parse(
        blobs,
        gcs_output_path=output_path,
        chunk_size=chunk_size,
        include_ancestor_headings=include_ancestor_headings,
    )
    operation_names = [op.operation.name for op in operations]
    logger.debug(
        "Started parsing with Document AI, submitted operations %s",
        operation_names,
    )
    time_elapsed = 0
    while self.is_running(operations):
      time.sleep(check_in_interval_sec)
      time_elapsed += check_in_interval_sec
      if time_elapsed > timeout_sec:
        raise TimeoutError(
            f"Timeout exceeded! Check operations {operation_names} later!"
        )
      logger.debug(".")

    results = self.get_results(operations=operations)
    yield from self.parse_from_results(results)

  def parse_from_results(
      self, results: List[DocAIParsingResults]
  ) -> Iterator[Document]:
    try:
      from google.cloud.documentai_toolbox.utilities.gcs_utilities import (  # type: ignore[import]
          split_gcs_uri,
      )
      from google.cloud.documentai_toolbox.wrappers.document import (  # type: ignore[import]
          _get_shards,
      )
      from google.cloud.documentai_toolbox.wrappers.page import _text_from_layout
    except ImportError as exc:
      raise ImportError(
          "documentai_toolbox package not found, please install it with "
          "`pip install langchain-google-community[docai]`"
      ) from exc
    for result in results:
      validate_gcs_uri(result.parsed_path, allow_prefix_only=True)
      print(f"processing: {result.parsed_path}")
      gcs_bucket_name, gcs_prefix = split_gcs_uri(result.parsed_path)
      shards = _get_shards(gcs_bucket_name, gcs_prefix + "/")
      if self._use_layout_parser:
        yield from (
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "source": result.source_path,
                },
            )
            for shard in shards
            for chunk in shard.chunked_document.chunks
        )
      else:
        yield from (
            Document(
                page_content=_text_from_layout(page.layout, shard.text),
                metadata={
                    "page": page.page_number,
                    "source": result.source_path,
                },
            )
            for shard in shards
            for page in shard.pages
        )

  def operations_from_names(
      self, operation_names: List[str]
  ) -> List["Operation"]:
    """Initializes Long-Running Operations from their names."""
    try:
      from google.longrunning.operations_pb2 import (  # type: ignore[import]
          GetOperationRequest,
      )
    except ImportError as exc:
      raise ImportError(
          "long running operations package not found, please install it with"
          "`pip install langchain-google-community[docai]`"
      ) from exc

    return [
        self._client.get_operation(request=GetOperationRequest(name=name))
        for name in operation_names
    ]

  def is_running(self, operations: List["Operation"]) -> bool:
    return any(not op.done() for op in operations)

  def docai_parse(
      self,
      blobs: Sequence[Blob],
      *,
      gcs_output_path: Optional[str] = None,
      processor_name: Optional[str] = None,
      batch_size: int = 1000,
      enable_native_pdf_parsing: bool = True,
      field_mask: Optional[str] = None,
      chunk_size: Optional[int] = 500,
      include_ancestor_headings: Optional[bool] = True,
  ) -> List["Operation"]:
    """Runs Google Document AI PDF Batch Processing on a list of blobs.

    Args:
        blobs: a list of blobs to be parsed
        gcs_output_path: a path (folder) on GCS to store results
        processor_name: name of a Document AI processor.
        batch_size: amount of documents per batch
        enable_native_pdf_parsing: a config option for the parser
        field_mask: a comma-separated list of which fields to include in the
          Document AI response.
            suggested: "text,pages.pageNumber,pages.layout"
        chunking_config: Serving config for chunking when using layout parser
          processor. Specify config parameters as dictionary elements.
            https://cloud.google.com/document-ai/docs/reference/rpc/google.cloud.documentai.v1beta3#chunkingconfig

    Document AI has a 1000 file limit per batch, so batches larger than that
      need
    to be split into multiple requests.
    Batch processing is an async long-running operation
    and results are stored in a output GCS bucket.
    """
    try:
      from google.cloud import documentai
      from google.cloud.documentai_v1.types import OcrConfig, ProcessOptions
    except ImportError as exc:
      raise ImportError(
          "documentai package not found, please install it with "
          "`pip install langchain-google-community[docai]`"
      ) from exc

    output_path = gcs_output_path or self._gcs_output_path
    if output_path is None:
      raise ValueError(
          "An output path on Google Cloud Storage should be provided."
      )
    output_path = validate_gcs_uri(output_path, allow_prefix_only=True)
    processor_name = processor_name or self._processor_name
    if processor_name is None:
      raise ValueError("A Document AI processor name should be provided.")
    processor_name = validate_processor_name(
        processor_name, expected_location=self._location
    )

    operations = []
    for batch in batch_iterate(size=batch_size, iterable=blobs):
      input_config = documentai.BatchDocumentsInputConfig(
          gcs_documents=documentai.GcsDocuments(
              documents=[
                  documentai.GcsDocument(
                      gcs_uri=validate_gcs_uri(getattr(blob, "path", None)),
                      mime_type=blob.mimetype or "application/pdf",
                  )
                  for blob in batch
              ]
          )
      )

      output_config = documentai.DocumentOutputConfig(
          gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
              gcs_uri=output_path, field_mask=field_mask
          )
      )

      if self._use_layout_parser:
        layout_config = ProcessOptions.LayoutConfig(
            chunking_config=ProcessOptions.LayoutConfig.ChunkingConfig(
                chunk_size=chunk_size,
                include_ancestor_headings=include_ancestor_headings,
            )
        )
        process_options = ProcessOptions(layout_config=layout_config)
      else:
        process_options = (
            ProcessOptions(
                ocr_config=OcrConfig(
                    enable_native_pdf_parsing=enable_native_pdf_parsing
                )
            )
            if enable_native_pdf_parsing
            else None
        )
      operations.append(
          self._client.batch_process_documents(
              documentai.BatchProcessRequest(
                  name=processor_name,
                  input_documents=input_config,
                  document_output_config=output_config,
                  process_options=process_options,
                  skip_human_review=True,
              )
          )
      )
    return operations

  def get_results(
      self, operations: List["Operation"]
  ) -> List[DocAIParsingResults]:
    try:
      from google.cloud.documentai_v1 import (  # type: ignore[import]
          BatchProcessMetadata,
      )
    except ImportError as exc:
      raise ImportError(
          "documentai package not found, please install it with "
          "`pip install langchain-google-community[docai]`"
      ) from exc

    return [
        DocAIParsingResults(
            source_path=status.input_gcs_source,
            parsed_path=status.output_gcs_destination,
        )
        for op in operations
        for status in (
            op.metadata.individual_process_statuses
            if isinstance(op.metadata, BatchProcessMetadata)
            else BatchProcessMetadata.deserialize(
                op.metadata.value
            ).individual_process_statuses
        )
    ]
