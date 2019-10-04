import requests
import zipfile
import tempfile
import os
import urllib.request

_CHUNK_SIZE = 32768
_GOOGLE_DRIVE_URL = "https://docs.google.com/uc?export=download"
_FAKE_HEADERS = [('User-Agent', 'Mozilla/5.0')]


def _save_response_content(response, destination):
    global _CHUNK_SIZE
    if isinstance(destination, str):
        with open(destination, 'wb') as handle:
            _save_response_content(response, handle)
    else:
        for chunk in response.iter_content(_CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                destination.write(chunk)


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
    destination : string or file
        File to which to save the downloaded data
    """
    global _GOOGLE_DRIVE_URL

    with requests.Session() as session:
        response = session.get(_GOOGLE_DRIVE_URL, params = { 'id' : id }, stream = True)
        token = _google_drive_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        _save_response_content(response, destination)


def download_url(url, destination):
    """Download a file from a URL

    Parameters
    ----------
    url : string
        URL of file to be downloaded
    destination : string or file
        File to which to save the downloaded data
    """
    if isinstance(destination, str):
        with open(destination, 'wb') as handle:
            download_url(url, handle)
    else:
        opener = urllib.request.build_opener()
        opener.addheaders = _FAKE_HEADERS
        urllib.request.install_opener(opener)
        with urllib.request.urlopen(url) as handle:
            destination.write(handle.read())


def download_and_extract_zip(url, destination):
    """Download a .zip file from a URL and extract it

    Parameters
    ----------
    url : string
        URL of file to be downloaded
    destination : string
        Directory in which to extract the downloaded zip
    """
    if not os.path.isdir(destination):
        os.mkdir(destination)
    zip_handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    download_url(url, zip_handle)
    zip_handle.close()
    with zipfile.ZipFile(zip_handle.name, 'r') as handle:
        handle.extractall(destination)
    os.unlink(zip_handle.name)
