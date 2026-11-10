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

resource "docker_image" "dataflow-image" {
  provider = docker.private
  name     = "${google_artifact_registry_repository.rag_repo.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.rag_repo.name}/dataflow-template:latest"
  build {
    context = "../backend/data_processing_pipeline_beam/"  
  }
}

resource "docker_registry_image" "dataflow" {
  provider      = docker.private
  name          = docker_image.dataflow-image.name
  keep_remotely = true
}

module "gcloud" {
  source  = "terraform-google-modules/gcloud/google"
  version = "3.4.1"
  platform              = "linux"
  additional_components = ["beta"]
  skip_download         = false  # Skip downloading gcloud if it's already available
  upgrade = true
  # create_cmd_entrypoint = "bash"
  create_cmd_body       = "beta dataflow flex-template build gs://${var.project_id}-rag/${var.dataflow_metadata_filename} --image ${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_id}/dataflow-template:latest --sdk-language PYTHON"
  module_depends_on     = [docker_registry_image.dataflow]
}

resource "google_dataflow_flex_template_job" "dataflow_template_job" {
  provider                = google-beta
  name                    = var.dataflow_job_name
  container_spec_gcs_path = "gs://${var.project_id}-rag/${var.dataflow_metadata_filename}"
  parameters = merge(
    {
      input_subscription = "projects/${var.project_id}/subscriptions/${var.input_topic_subscription_name}"
      output_topic       = "projects/${var.project_id}/topics/${var.output_topic_name}"
    },
    local.dataflow_allowlist_parameters,
  )
  depends_on = [module.gcloud, google_pubsub_topic.output, google_pubsub_subscription.input-subscription]
}

# --- Security allowlists for the Dataflow pipeline ---
# Messages on the input topic are untrusted, so the pipeline only touches the
# GCS buckets, Document AI processors and Vector Search resources listed here.
# Each dataflow_allowed_* variable overrides the default derived from the
# resources this configuration creates. An empty list means "use the default".
data "google_project" "project" {
  project_id = var.project_id
}

locals {
  # Document AI returns processor names with the project number, which is the
  # format the frontend asks users to enter.
  default_docai_processor_name = "projects/${data.google_project.project.number}/locations/${google_document_ai_processor.layout_parser_processor.location}/processors/${google_document_ai_processor.layout_parser_processor.name}"

  dataflow_allowlists = {
    allowed_gcs_buckets               = length(var.dataflow_allowed_gcs_buckets) > 0 ? var.dataflow_allowed_gcs_buckets : [google_storage_bucket.rag_bucket.name]
    allowed_gcp_projects              = length(var.dataflow_allowed_gcp_projects) > 0 ? var.dataflow_allowed_gcp_projects : [var.project_id]
    allowed_gcs_prefixes              = var.dataflow_allowed_gcs_prefixes
    allowed_docai_processors          = length(var.dataflow_allowed_docai_processors) > 0 ? var.dataflow_allowed_docai_processors : [local.default_docai_processor_name]
    allowed_docai_locations           = length(var.dataflow_allowed_docai_locations) > 0 ? var.dataflow_allowed_docai_locations : [google_document_ai_processor.layout_parser_processor.location]
    allowed_vector_store_projects     = length(var.dataflow_allowed_vector_store_projects) > 0 ? var.dataflow_allowed_vector_store_projects : [var.project_id]
    allowed_vector_store_regions      = length(var.dataflow_allowed_vector_store_regions) > 0 ? var.dataflow_allowed_vector_store_regions : [var.region]
    allowed_vector_store_index_ids    = length(var.dataflow_allowed_vector_store_index_ids) > 0 ? var.dataflow_allowed_vector_store_index_ids : [google_vertex_ai_index.custom_index.name]
    allowed_vector_store_endpoint_ids = length(var.dataflow_allowed_vector_store_endpoint_ids) > 0 ? var.dataflow_allowed_vector_store_endpoint_ids : [google_vertex_ai_index_endpoint.index_endpoint.name]
    allowed_embedding_models          = var.dataflow_allowed_embedding_models
  }

  # Flex Template parameters are strings; empty lists are omitted so the
  # pipeline uses its built-in defaults.
  dataflow_allowlist_parameters = {
    for name, values in local.dataflow_allowlists : name => join(",", values) if length(values) > 0
  }
}
