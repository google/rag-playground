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

from setuptools import find_packages, setup

setup(
    name="custom_transforms",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "aiohttp==3.9.5",
        "aiosignal==1.3.1",
        "annotated-types==0.7.0",
        "apache-beam==2.57.0",
        "attrs==23.2.0",
        "beautifulsoup4==4.12.3",
        "Bottleneck==1.4.0",
        "bs4==0.0.2",
        "cachetools==5.3.3",
        "certifi==2024.7.4",
        "charset-normalizer==3.3.2",
        "cloudpickle==2.2.1",
        "crcmod==1.7",
        "dataclasses-json==0.6.7",
        "db-dtypes==1.2.0",
        "decorator==5.1.1",
        "Deprecated==1.2.14",
        "dill==0.3.1.1",
        "dnspython==2.6.1",
        "docopt==0.6.2",
        "docstring_parser==0.16",
        "fastavro==1.9.5",
        "fasteners==0.19",
        "frozenlist==1.4.1",
        "fsspec==2024.6.1",
        "future==1.0.0",
        "gapic-google-longrunning==0.11.2",
        "gcsfs==2024.6.1",
        "google-api-core==2.19.1",
        "google-api-python-client==2.137.0",
        "google-apitools==0.5.31",
        "google-auth==2.32.0",
        "google-auth-httplib2==0.2.0",
        "google-auth-oauthlib==1.2.1",
        "google-cloud-aiplatform==1.59.0",
        "google-cloud-bigquery==3.25.0",
        "google-cloud-bigquery-storage==2.25.0",
        "google-cloud-bigtable==2.24.0",
        "google-cloud-contentwarehouse==0.7.8",
        "google-cloud-core==2.4.1",
        "google-cloud-datastore==2.19.0",
        "google-cloud-dlp==3.18.1",
        "google-cloud-documentai==2.29.2",
        "google-cloud-documentai-toolbox==0.13.5a0",
        "google-cloud-language==2.13.4",
        "google-cloud-pubsub==2.22.0",
        "google-cloud-pubsublite==1.11.0",
        "google-cloud-recommendations-ai==0.10.11",
        "google-cloud-resource-manager==1.12.4",
        "google-cloud-spanner==3.47.0",
        "google-cloud-storage==2.17.0",
        "google-cloud-videointelligence==2.13.4",
        "google-cloud-vision==3.7.3",
        "google-crc32c==1.5.0",
        "google-gax==0.14.1",
        "google-resumable-media==2.7.1",
        "googleapis-common-protos==1.63.2",
        "greenlet==3.0.3",
        "grpc-google-iam-v1==0.13.1",
        "grpc-interceptor==0.15.4",
        "grpcio==1.65.0",
        "grpcio-status==1.62.2",
        "hdfs==2.7.3",
        "httplib2==0.22.0",
        "idna==3.7",
        "immutabledict==4.2.0",
        "intervaltree==3.1.0",
        "Jinja2==3.1.4",
        "Js2Py==0.74",
        "jsonpatch==1.33",
        "jsonpickle==3.2.2",
        "jsonpointer==3.0.0",
        "jsonschema==4.23.0",
        "jsonschema-specifications==2023.12.1",
        "langchain==0.2.7",
        "langchain-community==0.2.7",
        "langchain-core==0.2.13",
        "langchain-google-community==1.0.6",
        (
            "langchain-google-vertexai @"
            " git+https://github.com/Abhishekbhagwat/langchain-google.git@dd6357210e003245f9d3c84006b813e88078a895#subdirectory=libs/vertexai"
        ),
        "langchain-text-splitters==0.2.2",
        "langsmith==0.1.85",
        "llvmlite==0.43.0",
        "lxml==5.2.2",
        "MarkupSafe==2.1.5",
        "marshmallow==3.21.3",
        "multidict==6.0.5",
        "mypy-extensions==1.0.0",
        "numba==0.60.0",
        "numexpr==2.10.1",
        "numpy==1.26.4",
        "oauth2client==3.0.0",
        "oauthlib==3.2.2",
        "objsize==0.7.0",
        "orjson==3.10.6",
        "overrides==7.7.0",
        "packaging==24.1",
        "pandas==2.2.2",
        "pandas-gbq==0.23.1",
        "pikepdf==8.15.1",
        "pillow==10.4.0",
        "ply==3.8",
        "proto-plus==1.24.0",
        "protobuf==4.25.3",
        "psutil==6.0.0",
        "pyarrow==15.0.2",
        "pyarrow-hotfix==0.6",
        "pyasn1==0.6.0",
        "pyasn1_modules==0.4.0",
        "pydantic==2.8.2",
        "pydantic_core==2.20.1",
        "pydata-google-auth==1.8.2",
        "pydot==1.4.2",
        "pyjsparser==2.7.1",
        "pymongo==4.8.0",
        "pyparsing==3.1.2",
        "python-dateutil==2.9.0.post0",
        "pytz==2024.1",
        "PyYAML==6.0.1",
        "redis==5.0.7",
        "referencing==0.35.1",
        "regex==2024.5.15",
        "requests==2.31.0",
        "requests-oauthlib==2.0.0",
        "rpds-py==0.19.0",
        "rsa==4.9",
        "ruff==0.5.1",
        "shapely==2.0.4",
        "six==1.16.0",
        "sortedcontainers==2.4.0",
        "soupsieve==2.5",
        "SQLAlchemy==2.0.31",
        "sqlparse==0.5.0",
        "tabulate==0.9.0",
        "tenacity==8.3.0",
        "typing-inspect==0.9.0",
        "typing_extensions==4.12.2",
        "tzdata==2024.1",
        "tzlocal==5.2",
        "uritemplate==4.1.1",
        "urllib3==2.2.2",
        "wrapt==1.16.0",
        "yarl==1.9.4",
        "zstandard==0.22.0",
    ],
)