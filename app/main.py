from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Corner, Video
from app.schemas import CornerResponse, UploadResponse
from app.services import segment_corners, transcribe_video

app = FastAPI(title="Video Corner Analyzer")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

Base.metadata.create_all(bind=engine)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename).suffix
    target = UPLOAD_DIR / f"{uuid4()}{ext}"

    with target.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    video = Video(filename=file.filename)
    db.add(video)
    db.flush()

    transcript = transcribe_video(target)
    corners = segment_corners(transcript)

    db_corners: list[Corner] = []
    for corner in corners:
        row = Corner(
            video_id=video.id,
            start_sec=float(corner["start_sec"]),
            end_sec=float(corner["end_sec"]),
            title=corner["title"],
            summary=corner["summary"],
        )
        db.add(row)
        db_corners.append(row)

    db.commit()

    return UploadResponse(
        video_id=video.id,
        filename=video.filename,
        corners=[
            CornerResponse(
                start_sec=row.start_sec,
                end_sec=row.end_sec,
                title=row.title,
                summary=row.summary,
            )
            for row in db_corners
        ],
    )


@app.get("/api/videos/{video_id}", response_model=UploadResponse)
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="video not found")

    corners = (
        db.query(Corner).filter(Corner.video_id == video_id).order_by(Corner.start_sec.asc()).all()
    )

    return UploadResponse(
        video_id=video.id,
        filename=video.filename,
        corners=[
            CornerResponse(
                start_sec=row.start_sec,
                end_sec=row.end_sec,
                title=row.title,
                summary=row.summary,
            )
            for row in corners
        ],
    )
