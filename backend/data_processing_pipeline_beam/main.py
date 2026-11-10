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
import sys

import apache_beam as beam
from apache_beam.io.gcp.pubsub import PubsubMessage, ReadFromPubSub, WriteToPubSub
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
# Import custom transforms
from custom_transforms.gcs_input_fn import GCSInputFn, LogInputFn
from custom_transforms.gcs_loader_fn import GCSLoaderFn
from custom_transforms.split_document_fn import SplitDocumentFn
from custom_transforms.upsert_vector_search_fn import UpsertToVectorSearchFn
from custom_transforms.url_input_fn import URLInputFn
from custom_transforms.url_loader_fn import URLLoaderFn
from custom_transforms.write_success_message_fn import WriteSuccessMessageFn


def is_gcs_input(element):
  """Determine if the input element is a GCS input.

  Args:
      element (dict): The input element to check.

  Returns:
      bool: True if the input type is 'gcs', False otherwise.
  """
  return element["input"]["type"] == "gcs"


def is_url_input(element):
  """Determine if the input element is a URL input.

  Args:
      element (dict): The input element to check.

  Returns:
      bool: True if the input type is 'url', False otherwise.
  """
  return element["input"]["type"] == "url"


_ALLOWLIST_OPTIONS = (
    ("allowed_gcs_buckets", "GCS buckets that may be read or written."),
    ("allowed_gcp_projects", "GCP projects for GCS input and Document AI."),
    ("allowed_gcs_prefixes", "Optional allowed GCS object prefixes."),
    ("allowed_docai_processors", "Document AI processor resource names."),
    ("allowed_docai_locations", "Document AI locations (default: us,eu)."),
    ("allowed_vector_store_projects", "Vector Search projects."),
    ("allowed_vector_store_regions", "Vector Search regions."),
    ("allowed_vector_store_index_ids", "Vector Search index IDs."),
    ("allowed_vector_store_endpoint_ids", "Vector Search endpoint IDs."),
    ("allowed_embedding_models", "Vertex AI embedding models."),
)


class CustomPipelineOptions(PipelineOptions):

  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument(
        "--input_subscription", required=True, help="Input PubSub subscription"
    )
    parser.add_argument(
        "--output_topic",
        required=True,
        help="Output PubSub topic for success messages",
    )
    parser.add_argument(
        "--window_size",
        type=float,
        default=1.0,
        help="Output file's window size in minutes.",
    )
    # Security allowlists (comma-separated). Each one is passed to the DoFn
    # constructors and pickled to workers. When an option is unset, the DoFn
    # falls back to the matching environment variable, and fails closed if
    # that is unset too.
    for option, help_text in _ALLOWLIST_OPTIONS:
      parser.add_argument(f"--{option}", default=None, help=help_text)


def _split_allowlist(value):
  """Parses a comma-separated option into a list.

  Returns None when the option is unset or empty so the DoFn falls back to its
  environment variable (and fails closed if that is unset as well).
  """
  if value is None:
    return None
  if isinstance(value, (list, tuple)):
    items = value
  else:
    items = str(value).split(",")
  parsed = [item.strip() for item in items if item and item.strip()]
  return parsed or None


def security_allowlists(custom_options):
  """Builds the allowlist constructor kwargs for each security-relevant DoFn.

  Args:
      custom_options (CustomPipelineOptions): The parsed pipeline options.

  Returns:
      dict: Constructor kwargs keyed by DoFn name.
  """
  opt = lambda name: _split_allowlist(getattr(custom_options, name, None))
  return {
      "GCSInputFn": {
          "allowed_buckets": opt("allowed_gcs_buckets"),
          "allowed_projects": opt("allowed_gcp_projects"),
          "allowed_prefixes": opt("allowed_gcs_prefixes"),
      },
      "GCSLoaderFn": {
          "allowed_projects": opt("allowed_gcp_projects"),
          "allowed_locations": opt("allowed_docai_locations"),
          "allowed_processors": opt("allowed_docai_processors"),
          "allowed_buckets": opt("allowed_gcs_buckets"),
      },
      "UpsertToVectorSearchFn": {
          "allowed_projects": opt("allowed_vector_store_projects"),
          "allowed_regions": opt("allowed_vector_store_regions"),
          "allowed_index_ids": opt("allowed_vector_store_index_ids"),
          "allowed_endpoint_ids": opt("allowed_vector_store_endpoint_ids"),
          "allowed_embedding_models": opt("allowed_embedding_models"),
      },
  }


def batch_if_needed(key_value_pair):
  """Batch documents if the number exceeds 1000.

  Args:
      key_value_pair (tuple): A tuple containing a key and a list of documents.

  Yields:
      tuple: A key and a batch of documents (max 1000 per batch).
  """
  key, docs = key_value_pair
  if isinstance(docs, list):
    doc_list = docs
  else:
    doc_list = [docs]

  if len(doc_list) > 1000:
    for i in range(0, len(doc_list), 1000):
      yield (key, doc_list[i : i + 1000])
  else:
    yield (key, doc_list)


def run(argv=None):
  """Main function to run the Apache Beam pipeline.

  Args:
      argv (list): Command line arguments.

  Returns:
      apache_beam.Pipeline: The executed pipeline object.
  """
  # set pipeline option save_main_session=True

  pipeline_options = PipelineOptions(argv, save_main_session=True)
  pipeline_options.view_as(StandardOptions).streaming = True
  custom_options = pipeline_options.view_as(CustomPipelineOptions)
  allowlists = security_allowlists(custom_options)

  with beam.Pipeline(options=pipeline_options) as p:
    inputs = (
        p
        | "ReadFromPubSub"
        >> ReadFromPubSub(subscription=custom_options.input_subscription)
        | "LogInput" >> beam.ParDo(LogInputFn())
        | "DecodeJSON" >> beam.Map(lambda x: json.loads(x.decode("utf-8")))
    )

    # Split inputs by type using Filters
    gcs_inputs = inputs | "FilterGCSInputs" >> beam.Filter(is_gcs_input)
    url_inputs = inputs | "FilterURLInputs" >> beam.Filter(is_url_input)

    # Process each input type
    processed_gcs = gcs_inputs | "ProcessGCSInputs" >> beam.ParDo(
        GCSInputFn(**allowlists["GCSInputFn"])
    )
    processed_url = url_inputs | "ProcessURLInputs" >> beam.ParDo(URLInputFn())

    # Load documents using separate loaders
    loaded_gcs_docs = processed_gcs | "LoadGCSDocuments" >> beam.ParDo(
        GCSLoaderFn(**allowlists["GCSLoaderFn"])
    )
    loaded_url_docs = processed_url | "LoadURLDocuments" >> beam.ParDo(
        URLLoaderFn()
    )

    # Merge loaded documents
    merged_docs = (
        loaded_gcs_docs,
        loaded_url_docs,
    ) | "MergeLoadedDocs" >> beam.Flatten()

    # Split documents
    split_docs = merged_docs | "SplitDocuments" >> beam.ParDo(SplitDocumentFn())

    # Window and batch splits
    windowed_splits = split_docs | "Windowing" >> beam.WindowInto(
        beam.window.FixedWindows(int(custom_options.window_size * 60))
    )
    grouped_splits = windowed_splits | "GroupByKey" >> beam.GroupByKey()
    batched_splits = grouped_splits | "BatchIfNeeded" >> beam.FlatMap(
        batch_if_needed
    )

    upsert_results = (
        batched_splits
        | "UpsertToVectorSearch"
        >> beam.ParDo(
            UpsertToVectorSearchFn(**allowlists["UpsertToVectorSearchFn"])
        )
        | "LogUpsertResults"
        >> beam.Map(lambda x: logging.info(f"Upsert result: {x}") or x)
    )

    successful_upserts = (
        upsert_results
        | "FilterSuccessfulUpserts" >> beam.Filter(lambda result: result[0])
        | "LogSuccessfulUpserts"
        >> beam.Map(lambda x: logging.info(f"Successful upsert: {x}") or x)
    )

    success_messages = (
        successful_upserts
        | "ExtractOriginalElement" >> beam.Map(lambda result: result[1])
        | "LogExtractedElement"
        >> beam.Map(lambda x: logging.info(f"Extracted element: {x}") or x)
        | "PrepareSuccessMessage" >> beam.ParDo(WriteSuccessMessageFn())
        | "WriteSuccessMessages"
        >> WriteToPubSub(
            topic=custom_options.output_topic, with_attributes=True
        )
    )

    # Log the prepared messages
    # _ = success_messages | "LogPreparedMessage" >> beam.Map(lambda x: logging.info(f"Prepared message: {x.decode('utf-8')}"))

    # Write messages to Pub/Sub
    # _ = success_messages | "WriteSuccessMessages" >> WriteToPubSub(topic=custom_options.output_topic, with_attributes=True)

    return p


if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  run()
