from fastapi import FastAPI, File, UploadFile

from api.helper import load_model

app = FastAPI(title="Ultra Model")

@app.get("/")
def launch_page():
    return {"msg":"Ultra Model !"}

@app.post("/model_predict")
def launch_model(fileimg: UploadFile = File(...)):
    rimage = fileimg.file.read()
    result = load_model(rimage)
    return result