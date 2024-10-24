import os
import json
from google.oauth2 import service_account
from google.cloud import storage

# 環境変数からサービスアカウント情報を取得
credentials_info = json.loads(os.environ.get('GOOGLE_CREDENTIALS'))
credentials = service_account.Credentials.from_service_account_info(credentials_info)
storage_client = storage.Client(credentials=credentials)

def download_model(bucket_name, source_blob_name, destination_file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

if __name__ == "__main__":
    bucket_name = 'foodphotograph'
    multi_task_model_blob = 'model/multi_task_model.pth'
    score_model_blob = 'model/rn50_photo1.pth'
    multi_task_model_local = 'multi_task_model.pth'
    score_model_local = 'rn50_photo1.pth'

    download_model(bucket_name, multi_task_model_blob, multi_task_model_local)
    download_model(bucket_name, score_model_blob, score_model_local)
