#!/usr/bin/env python3
"""
Cloud Storage Module for ML MCP System
Integration with AWS S3, Azure Blob, and Google Cloud Storage
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# AWS S3
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None

# Azure Blob Storage
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    BlobServiceClient = None

# Google Cloud Storage
try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    gcs = None


class S3Storage:
    """AWS S3 storage interface"""

    def __init__(self, bucket_name: str, region: Optional[str] = None):
        """
        Initialize S3 storage

        Args:
            bucket_name: S3 bucket name
            region: AWS region
        """
        if not S3_AVAILABLE:
            raise ImportError("boto3 required. Install with: pip install boto3")

        self.bucket_name = bucket_name
        self.region = region
        self.client = boto3.client('s3', region_name=region)
        self.resource = boto3.resource('s3', region_name=region)
        self.bucket = self.resource.Bucket(bucket_name)

    def upload_file(self, local_path: str, s3_key: str) -> Dict[str, Any]:
        """
        Upload file to S3

        Args:
            local_path: Local file path
            s3_key: S3 object key

        Returns:
            Upload result
        """
        try:
            self.client.upload_file(local_path, self.bucket_name, s3_key)

            return {
                'success': True,
                'bucket': self.bucket_name,
                's3_key': s3_key,
                'local_path': local_path
            }

        except ClientError as e:
            return {
                'success': False,
                'error': str(e)
            }

    def download_file(self, s3_key: str, local_path: str) -> Dict[str, Any]:
        """
        Download file from S3

        Args:
            s3_key: S3 object key
            local_path: Local destination path

        Returns:
            Download result
        """
        try:
            self.client.download_file(self.bucket_name, s3_key, local_path)

            return {
                'success': True,
                'bucket': self.bucket_name,
                's3_key': s3_key,
                'local_path': local_path
            }

        except ClientError as e:
            return {
                'success': False,
                'error': str(e)
            }

    def list_objects(self, prefix: str = '') -> List[str]:
        """
        List objects in bucket

        Args:
            prefix: Key prefix filter

        Returns:
            List of object keys
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []

        except ClientError:
            return []

    def read_csv(self, s3_key: str) -> pd.DataFrame:
        """
        Read CSV directly from S3

        Args:
            s3_key: S3 object key

        Returns:
            DataFrame
        """
        obj = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
        return pd.read_csv(obj['Body'])

    def write_csv(self, df: pd.DataFrame, s3_key: str) -> Dict[str, Any]:
        """
        Write DataFrame to S3 as CSV

        Args:
            df: DataFrame
            s3_key: S3 object key

        Returns:
            Write result
        """
        csv_buffer = df.to_csv(index=False)
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=csv_buffer
        )

        return {
            'success': True,
            'bucket': self.bucket_name,
            's3_key': s3_key,
            'rows': len(df)
        }


class AzureBlobStorage:
    """Azure Blob Storage interface"""

    def __init__(self, connection_string: str, container_name: str):
        """
        Initialize Azure Blob Storage

        Args:
            connection_string: Azure storage connection string
            container_name: Container name
        """
        if not AZURE_AVAILABLE:
            raise ImportError("azure-storage-blob required. Install with: pip install azure-storage-blob")

        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def upload_file(self, local_path: str, blob_name: str) -> Dict[str, Any]:
        """
        Upload file to Azure Blob

        Args:
            local_path: Local file path
            blob_name: Blob name

        Returns:
            Upload result
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)

            return {
                'success': True,
                'container': self.container_name,
                'blob_name': blob_name,
                'local_path': local_path
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def download_file(self, blob_name: str, local_path: str) -> Dict[str, Any]:
        """
        Download file from Azure Blob

        Args:
            blob_name: Blob name
            local_path: Local destination path

        Returns:
            Download result
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            with open(local_path, 'wb') as download_file:
                download_file.write(blob_client.download_blob().readall())

            return {
                'success': True,
                'container': self.container_name,
                'blob_name': blob_name,
                'local_path': local_path
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def list_blobs(self, prefix: str = '') -> List[str]:
        """
        List blobs in container

        Args:
            prefix: Name prefix filter

        Returns:
            List of blob names
        """
        try:
            blob_list = self.container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blob_list]

        except Exception:
            return []

    def read_csv(self, blob_name: str) -> pd.DataFrame:
        """
        Read CSV directly from Azure Blob

        Args:
            blob_name: Blob name

        Returns:
            DataFrame
        """
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )

        stream = blob_client.download_blob()
        return pd.read_csv(stream)


class GCSStorage:
    """Google Cloud Storage interface"""

    def __init__(self, bucket_name: str, project_id: Optional[str] = None):
        """
        Initialize GCS storage

        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID
        """
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage required. Install with: pip install google-cloud-storage")

        self.bucket_name = bucket_name
        self.client = gcs.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, local_path: str, blob_name: str) -> Dict[str, Any]:
        """
        Upload file to GCS

        Args:
            local_path: Local file path
            blob_name: Blob name

        Returns:
            Upload result
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(local_path)

            return {
                'success': True,
                'bucket': self.bucket_name,
                'blob_name': blob_name,
                'local_path': local_path
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def download_file(self, blob_name: str, local_path: str) -> Dict[str, Any]:
        """
        Download file from GCS

        Args:
            blob_name: Blob name
            local_path: Local destination path

        Returns:
            Download result
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(local_path)

            return {
                'success': True,
                'bucket': self.bucket_name,
                'blob_name': blob_name,
                'local_path': local_path
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def list_blobs(self, prefix: str = '') -> List[str]:
        """
        List blobs in bucket

        Args:
            prefix: Name prefix filter

        Returns:
            List of blob names
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]

        except Exception:
            return []

    def read_csv(self, blob_name: str) -> pd.DataFrame:
        """
        Read CSV directly from GCS

        Args:
            blob_name: Blob name

        Returns:
            DataFrame
        """
        blob = self.bucket.blob(blob_name)
        content = blob.download_as_text()

        from io import StringIO
        return pd.read_csv(StringIO(content))


class CloudStorageFactory:
    """Factory for creating cloud storage instances"""

    @staticmethod
    def create(provider: str, **kwargs) -> Union[S3Storage, AzureBlobStorage, GCSStorage]:
        """
        Create cloud storage instance

        Args:
            provider: 's3', 'azure', or 'gcs'
            **kwargs: Provider-specific arguments

        Returns:
            Storage instance
        """
        if provider == 's3':
            return S3Storage(
                bucket_name=kwargs['bucket_name'],
                region=kwargs.get('region')
            )

        elif provider == 'azure':
            return AzureBlobStorage(
                connection_string=kwargs['connection_string'],
                container_name=kwargs['container_name']
            )

        elif provider == 'gcs':
            return GCSStorage(
                bucket_name=kwargs['bucket_name'],
                project_id=kwargs.get('project_id')
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Get available cloud storage providers"""
        return {
            's3': S3_AVAILABLE,
            'azure': AZURE_AVAILABLE,
            'gcs': GCS_AVAILABLE
        }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python cloud_storage.py <action>")
        print("Actions: check_availability")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'check_availability':
            providers = CloudStorageFactory.get_available_providers()

            result = {
                'providers': providers,
                'install_commands': {
                    's3': 'pip install boto3',
                    'azure': 'pip install azure-storage-blob',
                    'gcs': 'pip install google-cloud-storage'
                }
            }

        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()