import requests

_CHUNK_SIZE = 32768
_GOOGLE_DRIVE_URL = "https://docs.google.com/uc?export=download"

def _save_response_content(response, destination):
    global _CHUNK_SIZE
    with open(destination, "wb") as f:
        for chunk in response.iter_content(_CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def _google_drive_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_google_drive(id, destination):
    """Download a file from Google Drive
    
    Requires the file to be available to view by anyone with the URL.
    
    Parameters
    ----------
    id : string
        Google Drive ID string. You can access this by clicking 'Get Shareable Link',
        which will give a URL of the form
        https://drive.google.com/file/d/**your_file_id**/view?usp=sharing
    destination : string
        Filename to which to save the downloaded file
    """
    global _GOOGLE_DRIVE_URL

    with requests.Session() as session:
        response = session.get(_GOOGLE_DRIVE_URL, params = { 'id' : id }, stream = True)
        token = _google_drive_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        _save_response_content(response, destination)
