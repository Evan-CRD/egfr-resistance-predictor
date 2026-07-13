# GitHub Setup

## 1. Create a repository

On GitHub, create a new public repository named:

`egfr-resistance-predictor`

Do not initialize it with another README because this folder already has one.

## 2. Upload using the GitHub website

For the simplest route:

1. Open the new repository.
2. Choose **Add file → Upload files**.
3. Drag the contents of this project folder into the upload area.
4. Commit the files.

GitHub's browser uploader may have file-size limits. The included files are
kept small enough for ordinary upload.

## 3. Upload using Git

```bash
cd egfr-resistance-predictor
git init
git add .
git commit -m "Initial EGFR resistance predictor"
git branch -M main
git remote add origin <YOUR-REPOSITORY-URL>
git push -u origin main
```

## 4. Deploy

Use Streamlit Community Cloud and set `app.py` as the main file.
