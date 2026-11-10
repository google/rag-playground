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

"""Hermetic stand-ins for heavy third-party deps used by the Beam transforms.

Stubs are only installed for modules that cannot be imported, so the tests
run both in a bare interpreter and in the full pipeline environment.
"""

import importlib
import os
import sys
import types

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
  sys.path.insert(0, _PIPELINE_DIR)


def _importable(name):
  try:
    importlib.import_module(name)
    return True
  except Exception:  # pylint: disable=broad-except
    return False


def _module(name, **attrs):
  """Creates (or extends) a stub module and its parent packages."""
  parts = name.split(".")
  for i in range(1, len(parts) + 1):
    sub = ".".join(parts[:i])
    if sub not in sys.modules:
      mod = types.ModuleType(sub)
      mod.__path__ = []
      sys.modules[sub] = mod
      if i > 1:
        setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
  mod = sys.modules[name]
  for k, v in attrs.items():
    setattr(mod, k, v)
  return mod


class _DoFn:

  def __init__(self, *args, **kwargs):
    pass


class _PipelineOptions:
  """Minimal stand-in; tests parse options via _add_argparse_args instead."""

  def __init__(self, *args, **kwargs):
    pass


class _Blob:

  def __init__(self, path=None, mimetype=None, **kwargs):
    self.path = path
    self.mimetype = mimetype

  @classmethod
  def from_path(cls, path, **kwargs):
    return cls(path=path, **kwargs)


class _Document:

  def __init__(self, page_content="", metadata=None, **kwargs):
    self.page_content = page_content
    self.metadata = metadata or {}


class _BaseLoader:
  pass


class _GCSDirectoryLoader:

  def __init__(
      self,
      project_name,
      bucket,
      prefix="",
      loader_func=None,
      continue_on_failure=False,
  ):
    self.project_name = project_name
    self.bucket = bucket
    self.prefix = prefix
    self._loader_func = loader_func
    self.continue_on_failure = continue_on_failure


class _BaseBlobParser:
  pass


REAL_BEAM = _importable("apache_beam")


def install():
  """Installs stubs for any dependency that is not importable."""
  if not _importable("apache_beam"):
    _module("apache_beam", DoFn=_DoFn)
    _module(
        "apache_beam.options.pipeline_options",
        PipelineOptions=_PipelineOptions,
        StandardOptions=_PipelineOptions,
    )
    _module(
        "apache_beam.io.gcp.pubsub",
        PubsubMessage=object,
        ReadFromPubSub=object,
        WriteToPubSub=object,
    )
  if not _importable("apache_beam.io") or not hasattr(
      sys.modules["apache_beam.io"], "PubsubMessage"
  ):
    _module("apache_beam.io", PubsubMessage=object)
  if not _importable("langchain_text_splitters"):
    _module(
        "langchain_text_splitters",
        RecursiveCharacterTextSplitter=type(
            "RecursiveCharacterTextSplitter", (), {}
        ),
    )
  if not _importable("langchain_community.document_loaders") or not hasattr(
      sys.modules["langchain_community.document_loaders"], "WebBaseLoader"
  ):
    _module(
        "langchain_community.document_loaders",
        WebBaseLoader=type("WebBaseLoader", (), {}),
    )
  if not _importable("langchain_core.documents"):
    _module("langchain_core.documents", Document=_Document)
  if not _importable("langchain_core.document_loaders.blob_loaders"):
    _module("langchain_core.document_loaders.blob_loaders", Blob=_Blob)
  if not _importable("langchain_core.document_loaders"):
    _module("langchain_core.document_loaders", BaseBlobParser=_BaseBlobParser)
  elif not hasattr(
      sys.modules["langchain_core.document_loaders"], "BaseBlobParser"
  ):
    _module("langchain_core.document_loaders", BaseBlobParser=_BaseBlobParser)
  if not _importable("langchain_core.utils.iter"):
    def batch_iterate(size, iterable):
      items = list(iterable)
      for i in range(0, len(items), size):
        yield items[i : i + size]
    _module("langchain_core.utils.iter", batch_iterate=batch_iterate)
  if not _importable("langchain_community.document_loaders.base"):
    _module("langchain_community.document_loaders.base", BaseLoader=_BaseLoader)
  if not _importable("langchain_community.document_loaders.gcs_directory"):
    _module(
        "langchain_community.document_loaders.gcs_directory",
        GCSDirectoryLoader=_GCSDirectoryLoader,
    )
  if not _importable("langchain_community.utilities.vertexai"):
    _module(
        "langchain_community.utilities.vertexai",
        get_client_info=lambda module=None: None,
    )
  if not _importable("langchain_google_community._utils"):
    _module(
        "langchain_google_community._utils",
        get_client_info=lambda module=None: None,
    )
  if not _importable("google.cloud.storage"):
    _module("google.cloud.storage", Client=object)
  if not _importable("langchain_google_vertexai"):
    _module(
        "langchain_google_vertexai",
        VectorSearchVectorStoreDatastore=type(
            "VectorSearchVectorStoreDatastore", (), {}
        ),
        VertexAIEmbeddings=type("VertexAIEmbeddings", (), {}),
    )
  if not _importable("langchain_google_vertexai.vectorstores._sdk_manager"):
    _module(
        "langchain_google_vertexai.vectorstores._sdk_manager",
        VectorSearchSDKManager=type("VectorSearchSDKManager", (), {}),
    )
  if not _importable("langchain_google_vertexai.vectorstores.document_storage"):
    _module(
        "langchain_google_vertexai.vectorstores.document_storage",
        DataStoreDocumentStorage=type(
            "DataStoreDocumentStorage",
            (),
            {"__init__": lambda self, *args, **kwargs: None},
        ),
        DocumentStorage=type("DocumentStorage", (), {}),
    )


install()
