---
title: 'CDD Vault to DataFrame: Python API and tutorial for querying and downloading
  data'
date: 2022-08-04 00:00:00 Z
layout: post
excerpt: Oftentimes, it is a necessary step to export data out of CDD Vault into Python
  or otherwise onto a local machine. This can be an annoying task, so we want to help
  make the process as simple as possible so you can get to your analysis.
toc: false
author: OCE Communications
featured_image: https://www.collaborativedrug.com/wp-content/uploads/2019/07/visualization-1.gif
---

There’s an easy way to do this, which involves first creating a Saved Search that reflects the exact query and format you’d like to export the data as, and then second calling the Saved Search via CDD’s API.

To use the following code you’ll need three pieces of information: your CDD Vault ID, your CDD API token with read access, and the Saved Search ID.

The below code creates a function, get_dataset_cdd_saved_search(search_id), which given a Saved Search id, returns a pd.DataFrame with the data.
```python
import time
import requests
import pandas as pd
from io import StringIO

# Written by OAM Communications Team, Oloren AI

token = None
vault_id = None

def run_saved_search(search_id):
    base_url = f"https://app.collaborativedrug.com/api/v1/vaults/{vault_id}/"
    headers = {'X-CDD-token':f'{token}'}
    url = base_url + f"searches/{search_id}"

    response = requests.request("GET", url, headers=headers).json()
    return response["id"]

def check_export_status(export_id):
    base_url = f"https://app.collaborativedrug.com/api/v1/vaults/{vault_id}/"
    headers = {'X-CDD-token':f'{token}'}
    url = base_url + f"export_progress/{export_id}"

    response = requests.request("GET", url, headers=headers).json()
    return response["status"]

def get_export(export_id):
    base_url = f"https://app.collaborativedrug.com/api/v1/vaults/{vault_id}/"
    headers = {'X-CDD-token':f'{token}'}
    url = base_url + f"exports/{export_id}"

    response = requests.request("GET", url, headers=headers)
    data_stream = StringIO(response.text)
    return pd.read_csv(data_stream)

def get_dataset_cdd_saved_search(search_id):
    export_id = run_saved_search(search_id)
    i = 0
    status = "new"
    while True:
        print(f"Export status is {status}, checking in {2**i} seconds...")
        time.sleep(2**i)
        status = check_export_status(export_id)
        if status == "finished":
            print("Export ready!")
            break
        i += 1
    return get_export(export_id)
```