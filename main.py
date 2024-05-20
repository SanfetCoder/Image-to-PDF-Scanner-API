from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from io import BytesIO
from helper.scanner import get_scanned_document
from fastapi.staticfiles import StaticFiles

# App instance
app = FastAPI(
  title="image-to-pdf-scanner",
  description="This is an api endpoint to convert image png to pdf",
  version="0.0.1"
)

# Enable static folder for the app
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/image/pdf")
async def scan_image(file: UploadFile = File(...)):
  try:
    # Reading image from the request
    image_bytes = await file.read() # Read some bytes from the file
    image_stream = BytesIO(image_bytes) # Buffer the bytes in-memory
    
    scanned_image = get_scanned_document(image_stream, file.filename)
    
    return Response(content=scanned_image, media_type="image/png")
  except Exception as error:
    print(error)
    raise HTTPException(status_code=500, detail="There was an error while processing your image")
