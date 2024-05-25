from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from io import BytesIO
from helper.scanner import get_scanned_document
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
# App instance
app = FastAPI()

# allow cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable static folder for the app
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/image/pdf")
async def scan_image(file: UploadFile = File(...)):
  try:
    print(f'[INFO] Processing image: {file.filename}')
    # Reading image from the request
    image_bytes = await file.read() # Read some bytes from the file
    image_stream = BytesIO(image_bytes) # Buffer the bytes in-memory

    _, file_path = get_scanned_document(image_stream, file.filename)
    if not Path(file_path).is_file():
      raise HTTPException(status_code=500, detail="รูปภาพไม่ชัดเจน หรือ อยู่ในที่มืด")
    return FileResponse(file_path, media_type='application/pdf', filename='scanned_document.pdf')
  except Exception as error:
    raise HTTPException(status_code=500, detail="รูปภาพไม่ชัดเจน หรือ อยู่ในที่มืด")
