import io
import time
import zipfile
import os
import gdown
import shutil

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from load_dataloader import *
from model_eval import Eval
from firebase import Firebase

SCOPES = ['https://www.googleapis.com/auth/drive']
f = Firebase()
delete = '.done'

metrics = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'r2': r2_score
}

def cleanup():
    allowed = ['.git',
        '.gitignore',
        '.venv',
        'credentials',
        'creds.py',
        'executables',
        'firebase.py',
        'google_api_test.ipynb',
        'load_dataloader.py',
        'main.py',
        'models',
        'model_eval.py',
        'README.md',
        'requirements.txt',
        'running.py',
        'test.ipynb',
        '__pycache__'
    ]
    for file in os.listdir():
        if file in allowed:
            continue
        if os.path.isfile(file):
            os.remove(file)
        else:
            shutil.rmtree(file)

def validate_creds():
    creds = None
    if os.path.exists("credentials/token.json"):
        creds = Credentials.from_authorized_user_file("credentials/token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials/google_creds.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("credentials/token.json", "w") as token:
            token.write(creds.to_json())
    return creds

def list_models(service):
    query = f"'1kzq60ZAhyl9Z7uKe9T0bsr8NrOJKhuP9' in parents"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    return items

def download_file(service, file):
    request = service.files().get_media(fileId=file['id'])
    file_path = os.path.join(file['name'])

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    # Save the file to the disk
    with open(file_path, 'wb') as f:
        f.write(fh.getvalue())
    print(f"Downloaded {file['name']}")

def delete_file_from_drive(service, file):
    service.files().update(fileId=file['id'], body={'name': delete}).execute()

def evaluate_model(dir):
    competition = f.get_competition(1)
    score_func = competition['function']

    res = {i: 0 for i in metrics}

    for model in os.listdir(dir):
        split_name = model[:-3].split('_')
        if len(split_name) != 2:
            print('Zip file structure incorrect')
            return
        try:
            func_id = int(split_name[1])
        except ValueError:
            print('Zip file structure incorrect')
            return

        dataloader = load_dataloader(func_id)

        try:
            model_load = Eval(f'user_models/{model}', dataloader)
            results = model_load.eval([(name, metrics[name]) for name in score_func])
        except Exception as e:
            print(f'Issue running {model}')
            return
        for metric in results:
            res[metric] += results[metric] / len(os.listdir(dir))

        print(f'Your results for function {func_id}:')
        for name in res:
            print(f'{name}: {res[name]:.4f}')

    f.add_submission({
        'name': file['user'],
        'computing_id': file['computing_id'],
        'competition': 1,
        'score': sum(score_func[metric] * res[metric] for metric in score_func),
        'metrics': res
    })
    print('Done')

while True:
    service = build("drive", "v3", credentials=validate_creds())
    files = list_models(service)
    print(files)
    for file in files:
        if file['name'] == delete:
            continue
        if not file['name'].endswith('.zip'):
            delete_file_from_drive(service, file)
            continue

        file_name = file['name'][:-4].split('_')
        if len(file_name) != 2:
            delete_file_from_drive(service, file)
            continue

        file['user'], file['computing_id'] = file_name
        download_file(service, file)

        dataloader_url = f.get_competition(1)['url']
        gdown.download(dataloader_url, 'dataloaders.zip', quiet=True)
        with zipfile.ZipFile('dataloaders.zip', 'r') as zip_ref:
            zip_ref.extractall('dataloaders')
            print(f'Extracted dataloaders')

        with zipfile.ZipFile(file['name'], 'r') as zip_ref:
            zip_ref.extractall('user_models')
            print(f'Extracted {file["name"]}')
        
        evaluate_model('user_models')

        cleanup()

        delete_file_from_drive(service, file)
    time.sleep(5)
