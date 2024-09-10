import io
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from os import path, remove

from load_dataloader import *
from model_eval import Eval
from firebase import Firebase

SCOPES = ['https://www.googleapis.com/auth/drive']
f = Firebase()
delete = '.done'

def validate_creds():
    creds = None
    if path.exists("credentials/token.json"):
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
    file_path = path.join('model.pt')

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

def evaluate_model(file):
    metrics = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r2_score
    }
    competition = f.get_competition(file['competition'])
    score_func = competition['function']

    model_load = Eval('model.pt', competition)
    res = model_load.eval([(name, metrics[name]) for name in score_func])

    print('Your results:')
    for name in res:
        print(f'{name}: {res[name]:.4f}')

    f.add_submission({
        'name': file['user'],
        'computing_id': file['computing_id'],
        'competition': file['competition'],
        'score': sum(score_func[metric] * res[metric] for metric in score_func),
        'metrics': res
    })
    print('Done')
    remove('model.pt')

while True:
    service = build("drive", "v3", credentials=validate_creds())
    files = list_models(service)
    print(files)
    for file in files:
        if file['name'] == delete:
            continue
        if '.pt' not in file['name']:
            delete_file_from_drive(service, file)
            continue
        file_name = file['name'][:-3].split('_')
        if len(file_name) != 3:
            delete_file_from_drive(service, file)
            continue
        file['user'], file['computing_id'], file['competition'] = file_name
        file['competition'] = int(file['competition'])
        download_file(service, file)
        delete_file_from_drive(service, file)
        evaluate_model(file)
    time.sleep(5)
