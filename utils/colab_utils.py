from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os


class GoogleDrive_Dowonloader(object):
    def __init__(self, use_pydrive):
        self.use_pydrive = use_pydrive

        if self.use_pydrive:
            self.authenticate()

    def authenticate(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def ensure_file_exists(self, file_id, file_dst):
      if not os.path.isfile(file_dst):
        if self.use_pydrive:
            print(f'Downloading {file_dst} ...')
            downloaded = self.drive.CreateFile({'id':file_id})
            downloaded.FetchMetadata(fetch_all=True)
            downloaded.GetContentFile(file_dst)
            print('Finished')
        else:
            from gdown import download as drive_download
            drive_download(f'https://drive.google.com/uc?id={file_id}', file_dst, quiet=False)
      else:
        print(f'{file_dst} exists.')



