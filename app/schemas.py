from pydantic import BaseModel


class CornerResponse(BaseModel):
    start_sec: float
    end_sec: float
    title: str
    summary: str


class UploadResponse(BaseModel):
    video_id: int
    filename: str
    corners: list[CornerResponse]
