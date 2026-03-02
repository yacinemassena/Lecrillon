import boto3
from pathlib import Path

s3 = boto3.client('s3',
    endpoint_url='https://2a139e9393f803634546ad9d541d37b9.r2.cloudflarestorage.com',
    aws_access_key_id='fdfa18bf64b18c61bbee64fda98ca20b',
    aws_secret_access_key='394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8'
)

bucket = 'europe'

# Download RV file first
rv_remote = 'datasets/spy_daily_rv_30d.parquet'
rv_local = Path('/TCN/datasets/2022-2023/spy_daily_rv.parquet')
rv_local.parent.mkdir(parents=True, exist_ok=True)
print(f'Downloading RV file: {rv_remote}')
s3.download_file(bucket, rv_remote, str(rv_local))
print(f'  -> {rv_local}')

# Download index data
prefix = 'datasets/index_data/'
local_dir = Path('/TCN/datasets/2022-2023/index_data')
local_dir.mkdir(parents=True, exist_ok=True)

print('Downloading index_data from R2...')
paginator = s3.get_paginator('list_objects_v2')
count = 0
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    if 'Contents' not in page:
        continue
    for obj in page['Contents']:
        key = obj['Key']
        if key.endswith('/'):
            continue
        local_file = local_dir / key.replace(prefix, '')
        local_file.parent.mkdir(parents=True, exist_ok=True)
        print(f'  [{count}] {key}')
        s3.download_file(bucket, key, str(local_file))
        count += 1

print(f'Download complete! {count} index files + RV file downloaded.')
