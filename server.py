import io
import json
import os
import importlib.util
import shutil
import tempfile
import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
_cors_origins_raw = str(os.environ.get("OCR_CORS_ORIGINS") or "").strip()
_node_env = str(os.environ.get("NODE_ENV") or "").strip().lower()
_allow_all = (_cors_origins_raw == "*") or (not _cors_origins_raw and _node_env != "production")
_cors_origins = ["*"] if _allow_all else [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
if _cors_origins:
  app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
  )


class OcrRequest(BaseModel):
  pdf_url: str
  book_id: Optional[str] = None
  title: Optional[str] = None
  mode: str = "ocr"
  dpi: int = 300
  max_pages: Optional[int] = None
  page_start: Optional[int] = None
  page_end: Optional[int] = None


class OcrSlicesRequest(BaseModel):
  manifest_url: str
  book_id: Optional[str] = None
  title: Optional[str] = None
  max_pages: Optional[int] = None
  page_start: Optional[int] = None
  page_end: Optional[int] = None


class SliceUploadRequest(BaseModel):
  pdf_url: str
  book_id: Optional[str] = None
  title: Optional[str] = None
  dpi: int = 200
  quality: int = 82
  max_pages: Optional[int] = None
  page_start: Optional[int] = None
  page_end: Optional[int] = None
  bucket: str = "library_pages"
  prefix: Optional[str] = None
  supabase_url: Optional[str] = None
  supabase_service_role_key: Optional[str] = None


def _download_pdf(url: str, timeout: int = 60) -> bytes:
  try:
    resp = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
  except Exception as e:
    raise HTTPException(status_code=502, detail=str(e))
  if resp.status_code >= 400:
    raise HTTPException(status_code=502, detail=f"upstream {resp.status_code}")
  buf = io.BytesIO()
  for chunk in resp.iter_content(chunk_size=1024 * 256):
    if chunk:
      buf.write(chunk)
  return buf.getvalue()


def _download_bytes(url: str, timeout: int = 60) -> bytes:
  try:
    resp = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
  except Exception as e:
    raise HTTPException(status_code=502, detail=str(e))
  if resp.status_code >= 400:
    raise HTTPException(status_code=502, detail=f"upstream {resp.status_code}")
  buf = io.BytesIO()
  for chunk in resp.iter_content(chunk_size=1024 * 256):
    if chunk:
      buf.write(chunk)
  return buf.getvalue()


_paddle_ocr: Optional[Any] = None
_paddle_ocr_error: Optional[str] = None


def _get_paddle_ocr() -> Optional[Any]:
  global _paddle_ocr, _paddle_ocr_error
  if _paddle_ocr is not None:
    return _paddle_ocr
  if _paddle_ocr_error is not None:
    return None
  try:
    from paddleocr import PaddleOCR  # type: ignore
  except Exception as e:
    _paddle_ocr_error = str(e)
    return None
  try:
    try:
      _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    except Exception:
      _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch")
  except Exception as e:
    _paddle_ocr_error = str(e)
    return None
  return _paddle_ocr


def _extract_text_with_pypdf(pdf_bytes: bytes, max_pages: Optional[int]) -> List[str]:
  try:
    from pypdf import PdfReader  # type: ignore
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"missing pypdf: {e}")
  reader = PdfReader(io.BytesIO(pdf_bytes))
  pages = reader.pages
  if max_pages and isinstance(max_pages, int) and max_pages > 0:
    pages = pages[:max_pages]
  out: List[str] = []
  for p in pages:
    t = p.extract_text() or ""
    out.append(str(t).strip())
  return out


def _require_true_ocr_deps() -> Dict[str, Any]:
  missing: List[str] = []
  if importlib.util.find_spec("cv2") is None:
    missing.append("opencv-python")
  if importlib.util.find_spec("numpy") is None:
    missing.append("numpy")
  if importlib.util.find_spec("paddleocr") is None:
    missing.append("paddleocr")
  if importlib.util.find_spec("paddle") is None:
    missing.append("paddlepaddle")

  pdf2image_ok = importlib.util.find_spec("pdf2image") is not None
  poppler_ok = bool(shutil.which("pdftoppm") or shutil.which("pdfinfo"))
  pdfium_ok = importlib.util.find_spec("pypdfium2") is not None
  if not ((pdf2image_ok and poppler_ok) or pdfium_ok):
    missing.append("pypdfium2 (或 pdf2image+poppler)")

  if missing:
    hint = {
      "pip": "python3 -m pip install paddlepaddle paddleocr opencv-python pdf2image numpy",
      "macos_poppler": "brew install poppler",
      "env": "export OCR_MODE=ocr",
    }
    raise HTTPException(status_code=500, detail={"error": "真实 OCR 依赖缺失", "missing": missing, "hint": hint})

  return {"ok": True}


def _require_pdf_render_deps() -> Dict[str, Any]:
  pdf2image_ok = importlib.util.find_spec("pdf2image") is not None
  poppler_ok = bool(shutil.which("pdftoppm") or shutil.which("pdfinfo"))
  pdfium_ok = importlib.util.find_spec("pypdfium2") is not None
  if (pdf2image_ok and poppler_ok) or pdfium_ok:
    return {"ok": True}
  raise HTTPException(status_code=500, detail={"error": "缺少 PDF 渲染依赖", "missing": ["pypdfium2 (或 pdf2image+poppler)"]})


def _get_supabase_admin_config() -> Dict[str, str]:
  supabase_url = str(os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
  service_role_key = str(os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
  if not supabase_url or not service_role_key:
    _maybe_load_dev_env()
    supabase_url = str(os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
    service_role_key = str(os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
  if not supabase_url or not service_role_key:
    raise HTTPException(status_code=500, detail="missing SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY")
  return {"supabase_url": supabase_url.rstrip("/"), "service_role_key": service_role_key}


def _get_supabase_url() -> str:
  supabase_url = str(os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
  if not supabase_url:
    _maybe_load_dev_env()
    supabase_url = str(os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
  if not supabase_url:
    raise HTTPException(status_code=500, detail="missing SUPABASE_URL")
  return supabase_url.rstrip("/")


def _get_supabase_storage_config_from_req(req: SliceUploadRequest, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
  headers = headers or {}
  supabase_url = str(req.supabase_url or "").strip().rstrip("/") or _get_supabase_url()
  service_role_key = str(req.supabase_service_role_key or "").strip()
  if service_role_key:
    return {
      "supabase_url": supabase_url,
      "apikey": service_role_key,
      "authorization": f"Bearer {service_role_key}",
    }

  try:
    cfg = _get_supabase_admin_config()
    return {
      "supabase_url": cfg["supabase_url"],
      "apikey": cfg["service_role_key"],
      "authorization": f"Bearer {cfg['service_role_key']}",
    }
  except HTTPException:
    pass

  apikey = str(headers.get("apikey") or headers.get("x-supabase-api-key") or "").strip()
  authorization = str(headers.get("authorization") or "").strip()
  if authorization and not authorization.lower().startswith("bearer "):
    authorization = f"Bearer {authorization}"
  if not apikey or not authorization:
    raise HTTPException(status_code=400, detail="missing apikey/authorization for storage upload")
  return {"supabase_url": supabase_url, "apikey": apikey, "authorization": authorization}


_dev_env_loaded = False


def _load_env_file(path: str) -> None:
  try:
    with open(path, "r", encoding="utf-8") as f:
      lines = f.read().splitlines()
  except Exception:
    return
  for raw in lines:
    line = str(raw or "").strip()
    if not line or line.startswith("#"):
      continue
    if "=" not in line:
      continue
    k, v = line.split("=", 1)
    key = k.strip()
    if not key:
      continue
    if key in os.environ and str(os.environ.get(key) or "").strip():
      continue
    val = v.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
      val = val[1:-1]
    os.environ[key] = val


def _maybe_load_dev_env() -> None:
  global _dev_env_loaded
  if _dev_env_loaded:
    return
  _dev_env_loaded = True
  node_env = str(os.environ.get("NODE_ENV") or os.environ.get("ENV") or "").strip().lower()
  if node_env == "production":
    return
  base_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(base_dir, ".."))
  candidates = [
    os.path.join(repo_root, "web", ".env.local"),
    os.path.join(repo_root, "web", ".env"),
    os.path.join(repo_root, "cms", ".env"),
    os.path.join(repo_root, ".env.local"),
    os.path.join(repo_root, ".env"),
  ]
  for p in candidates:
    if os.path.exists(p):
      _load_env_file(p)


def _encode_storage_path(path: str) -> str:
  parts = [p for p in (path or "").split("/") if p]
  return "/".join(quote(p, safe="") for p in parts)


def _public_object_url(supabase_url: str, bucket: str, path: str) -> str:
  encoded = _encode_storage_path(path)
  return f"{supabase_url}/storage/v1/object/public/{bucket}/{encoded}"


def _upload_storage_object(
  supabase_url: str,
  apikey: str,
  authorization: str,
  bucket: str,
  path: str,
  content: bytes,
  content_type: str,
) -> None:
  encoded_path = _encode_storage_path(path)
  url = f"{supabase_url}/storage/v1/object/{bucket}/{encoded_path}"
  headers = {
    "authorization": authorization,
    "apikey": apikey,
    "content-type": content_type,
    "x-upsert": "true",
  }
  try:
    resp = requests.post(url, headers=headers, data=content, timeout=120)
  except Exception as e:
    raise HTTPException(status_code=502, detail=str(e))
  if resp.status_code >= 400:
    detail = (resp.text or "").strip()
    if len(detail) > 600:
      detail = detail[:600]
    msg = f"storage upload failed: {resp.status_code}"
    if detail:
      msg = f"{msg} - {detail}"
    raise HTTPException(status_code=502, detail=msg)


def _postgrest_request(
  *,
  supabase_url: str,
  service_role_key: str,
  method: str,
  table: str,
  query: str = "",
  body: Optional[Any] = None,
  prefer: Optional[str] = None,
  timeout: int = 120,
) -> Any:
  base = supabase_url.rstrip("/") + "/rest/v1/" + table
  url = base + (query or "")
  headers: Dict[str, str] = {
    "apikey": service_role_key,
    "authorization": f"Bearer {service_role_key}",
    "accept": "application/json",
  }
  if body is not None:
    headers["content-type"] = "application/json"
  if prefer:
    headers["prefer"] = prefer
  try:
    resp = requests.request(method.upper(), url, headers=headers, json=body, timeout=timeout)
  except Exception as e:
    raise HTTPException(status_code=502, detail=str(e))
  if resp.status_code >= 400:
    detail = (resp.text or "").strip()
    if len(detail) > 800:
      detail = detail[:800]
    raise HTTPException(status_code=502, detail=f"postgrest {method.upper()} {table} failed: {resp.status_code} - {detail}")
  if resp.status_code == 204:
    return None
  return resp.json() if (resp.text or "").strip() else None


def _ensure_library_book(
  *,
  supabase_url: str,
  service_role_key: str,
  book_id: Optional[str],
  title: Optional[str],
  author: Optional[str],
  dynasty: Optional[str],
  category: Optional[str],
  status: Optional[str],
  description: Optional[str],
  cover_url: Optional[str],
  create_if_missing: bool,
) -> Dict[str, Any]:
  if book_id:
    rows = _postgrest_request(
      supabase_url=supabase_url,
      service_role_key=service_role_key,
      method="GET",
      table="library_books",
      query=f"?select=*&id=eq.{book_id}&limit=1",
      body=None,
    )
    if isinstance(rows, list) and rows:
      return rows[0]
    if not create_if_missing:
      raise HTTPException(status_code=400, detail="book not found")
    payload: Dict[str, Any] = {
      "id": book_id,
      "title": (title or "").strip() or "未命名",
      "author": (author or "").strip() or None,
      "dynasty": (dynasty or "").strip() or None,
      "category": (category or "").strip() or None,
      "status": (status or "").strip() or None,
      "description": (description or "").strip() or None,
      "cover_url": (cover_url or "").strip() or None,
    }
    created = _postgrest_request(
      supabase_url=supabase_url,
      service_role_key=service_role_key,
      method="POST",
      table="library_books",
      query="",
      body=payload,
      prefer="return=representation",
    )
    if isinstance(created, list) and created:
      return created[0]
    return payload

  if not title or not str(title).strip():
    raise HTTPException(status_code=400, detail="missing title")

  payload = {
    "title": str(title).strip(),
    "author": (author or "").strip() or None,
    "dynasty": (dynasty or "").strip() or None,
    "category": (category or "").strip() or None,
    "status": (status or "").strip() or None,
    "description": (description or "").strip() or None,
    "cover_url": (cover_url or "").strip() or None,
  }
  created = _postgrest_request(
    supabase_url=supabase_url,
    service_role_key=service_role_key,
    method="POST",
    table="library_books",
    query="",
    body=payload,
    prefer="return=representation",
  )
  if isinstance(created, list) and created:
    return created[0]
  raise HTTPException(status_code=502, detail="create book failed")


def _replace_book_contents(
  *,
  supabase_url: str,
  service_role_key: str,
  book_id: str,
  chapters: List[Dict[str, Any]],
) -> None:
  _postgrest_request(
    supabase_url=supabase_url,
    service_role_key=service_role_key,
    method="DELETE",
    table="library_book_contents",
    query=f"?book_id=eq.{book_id}",
    body=None,
    prefer=None,
  )
  rows: List[Dict[str, Any]] = []
  for idx, c in enumerate(chapters):
    rows.append(
      {
        "book_id": book_id,
        "volume_no": c.get("volume_no"),
        "volume_title": c.get("volume_title"),
        "chapter_no": c.get("chapter_no"),
        "chapter_title": c.get("chapter_title"),
        "content": c.get("content"),
        "sort_order": int(c.get("sort_order") or (idx + 1)),
      }
    )
  chunk_size = 200
  for i in range(0, len(rows), chunk_size):
    _postgrest_request(
      supabase_url=supabase_url,
      service_role_key=service_role_key,
      method="POST",
      table="library_book_contents",
      query="",
      body=rows[i : i + chunk_size],
      prefer="return=minimal",
      timeout=300,
    )


def _update_book_source_payload(
  *,
  supabase_url: str,
  service_role_key: str,
  book_id: str,
  next_source_payload: Dict[str, Any],
) -> None:
  _postgrest_request(
    supabase_url=supabase_url,
    service_role_key=service_role_key,
    method="PATCH",
    table="library_books",
    query=f"?id=eq.{book_id}",
    body={"source_payload": next_source_payload},
    prefer="return=minimal",
  )


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
  mode = str(os.environ.get("OCR_MODE", "ocr") or "ocr").strip().lower()
  info: Dict[str, Any] = {"ok": True, "mode": mode}
  try:
    _require_true_ocr_deps()
    info["true_ocr_ready"] = True
  except HTTPException as e:
    info["true_ocr_ready"] = False
    info["detail"] = e.detail
  return info


@app.post("/ocr")
def ocr_pdf(req: OcrRequest) -> Dict[str, Any]:
  pdf_url = (req.pdf_url or "").strip()
  if not pdf_url:
    raise HTTPException(status_code=400, detail="missing pdf_url")

  pdf_bytes = _download_pdf(pdf_url)
  if not pdf_bytes:
    raise HTTPException(status_code=502, detail="empty pdf")

  mode = str((req.mode or "")).strip().lower() or str(os.environ.get("OCR_MODE", "ocr") or "ocr").strip().lower()
  if mode not in ("ocr", "auto", "text"):
    raise HTTPException(status_code=400, detail="invalid mode")

  chapters: List[Dict[str, Any]] = []
  full: List[str] = []

  if mode == "text":
    lines = _extract_text_with_pypdf(pdf_bytes, req.max_pages)
    for idx, page_text in enumerate(lines):
      page_no = idx + 1
      chapters.append(
        {
          "volume_no": 1,
          "volume_title": None,
          "chapter_no": page_no,
          "chapter_title": f"第{page_no}页",
          "content": page_text,
          "sort_order": page_no,
        }
      )
      if page_text:
        full.append(page_text)
    return {"chapters": chapters, "text": "\n\n".join(full).strip()}

  if mode == "ocr":
    _require_true_ocr_deps()

  try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
  except Exception as e:
    if mode == "ocr":
      raise HTTPException(status_code=500, detail={"error": "真实 OCR 运行依赖不可用", "detail": str(e)})
    lines = _extract_text_with_pypdf(pdf_bytes, req.max_pages)
    for idx, page_text in enumerate(lines):
      page_no = idx + 1
      chapters.append(
        {
          "volume_no": 1,
          "volume_title": None,
          "chapter_no": page_no,
          "chapter_title": f"第{page_no}页",
          "content": page_text,
          "sort_order": page_no,
        }
      )
      if page_text:
        full.append(page_text)
    return {"chapters": chapters, "text": "\n\n".join(full).strip()}

  ocr = _get_paddle_ocr()
  if ocr is None:
    if mode == "ocr":
      raise HTTPException(
        status_code=500,
        detail={"error": "PaddleOCR 未就绪", "detail": _paddle_ocr_error or "paddleocr/paddlepaddle 未安装或初始化失败"},
      )
    lines = _extract_text_with_pypdf(pdf_bytes, req.max_pages)
    for idx, page_text in enumerate(lines):
      page_no = idx + 1
      chapters.append(
        {
          "volume_no": 1,
          "volume_title": None,
          "chapter_no": page_no,
          "chapter_title": f"第{page_no}页",
          "content": page_text,
          "sort_order": page_no,
        }
      )
      if page_text:
        full.append(page_text)
    return {"chapters": chapters, "text": "\n\n".join(full).strip()}

  def preprocess_candidates(pil_image: Any) -> List[Any]:
    try:
      if hasattr(pil_image, "convert"):
        pil_image = pil_image.convert("RGB")
    except Exception:
      pass
    img_np = np.array(pil_image)
    if img_np.dtype != np.uint8:
      img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    if len(img_np.shape) == 2:
      img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
      img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
      img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
      img_cv = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
      raise HTTPException(status_code=500, detail=f"unsupported image shape: {tuple(getattr(img_np, 'shape', ()))}")
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
      gray,
      255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY,
      11,
      2,
    )
    return [img_cv, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)]

  def sort_text_lines(res0: Any) -> List[str]:
    def is_box(v: Any) -> bool:
      if hasattr(v, "tolist"):
        try:
          v = v.tolist()
        except Exception:
          return False
      if not isinstance(v, (list, tuple)) or len(v) != 4:
        return False
      for p in v:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
          return False
        if not isinstance(p[0], (int, float)) or not isinstance(p[1], (int, float)):
          return False
      return True

    def box_pos(box: Any) -> Optional[Dict[str, float]]:
      if not is_box(box):
        return None
      xs = [float(p[0]) for p in box]
      ys = [float(p[1]) for p in box]
      return {"x": float(min(xs)), "y": float(min(ys))}

    lines: List[Dict[str, Any]] = []
    items: Any = []
    if isinstance(res0, dict):
      texts = res0.get("rec_texts")
      polys = res0.get("rec_polys") or res0.get("dt_polys") or res0.get("dt_boxes")
      if isinstance(texts, list) and isinstance(polys, list):
        tmp: List[Any] = []
        for t, b in zip(texts, polys):
          if not isinstance(t, str) or not t.strip():
            continue
          if hasattr(b, "tolist"):
            try:
              b = b.tolist()
            except Exception:
              b = None
          tmp.append([b, (t, 1.0)])
        items = tmp
      else:
        items = []
    elif isinstance(res0, list):
      items = res0
    else:
      items = []
    for it in items:
      text: Optional[str] = None
      box: Any = None
      if isinstance(it, dict):
        t = it.get("text")
        if isinstance(t, str) and t.strip():
          text = t.strip()
        b = it.get("box") or it.get("dt_boxes") or it.get("points")
        if is_box(b):
          box = b
      elif isinstance(it, (list, tuple)):
        if len(it) >= 2 and is_box(it[0]):
          box = it[0]
          second = it[1]
          if isinstance(second, (list, tuple)) and len(second) >= 1 and isinstance(second[0], str) and second[0].strip():
            text = second[0].strip()
          elif isinstance(second, str) and second.strip():
            text = second.strip()
          elif len(it) >= 3 and isinstance(it[1], str) and it[1].strip():
            text = it[1].strip()
        elif len(it) >= 1 and isinstance(it[0], str) and it[0].strip():
          text = it[0].strip()
      if not text:
        continue
      pos = box_pos(box)
      if pos:
        lines.append({"text": text, "x": pos["x"], "y": pos["y"]})
      else:
        lines.append({"text": text, "x": float("inf"), "y": float("inf")})

    lines.sort(key=lambda k: (-k["x"], k["y"]))
    return [d["text"] for d in lines]

  def render_images(pdf_path: str) -> List[Any]:
    first_page = req.page_start if isinstance(req.page_start, int) and req.page_start > 0 else None
    last_page = req.page_end if isinstance(req.page_end, int) and req.page_end > 0 else None

    pdf2image_ok = importlib.util.find_spec("pdf2image") is not None
    poppler_ok = bool(shutil.which("pdftoppm") or shutil.which("pdfinfo"))
    if pdf2image_ok and poppler_ok:
      from pdf2image import convert_from_path  # type: ignore

      poppler_path = os.environ.get("POPPLER_PATH") or None
      images = convert_from_path(
        pdf_path,
        dpi=int(req.dpi),
        first_page=first_page,
        last_page=last_page,
        poppler_path=poppler_path,
      )
      return images[: req.max_pages] if req.max_pages and isinstance(req.max_pages, int) and req.max_pages > 0 else images

    try:
      import pypdfium2 as pdfium  # type: ignore
    except Exception as e:
      raise HTTPException(status_code=500, detail={"error": "缺少 PDF 渲染依赖", "detail": str(e)})

    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    start = max(1, int(first_page)) if first_page else 1
    end = min(total, int(last_page)) if last_page else total
    page_indices = list(range(start - 1, end))
    if req.max_pages and isinstance(req.max_pages, int) and req.max_pages > 0:
      page_indices = page_indices[: req.max_pages]

    scale = float(max(int(req.dpi), 72)) / 72.0
    out: List[Any] = []
    for i in page_indices:
      page = pdf.get_page(i)
      try:
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        out.append(pil if isinstance(pil, Image.Image) else pil)
      finally:
        page.close()
    pdf.close()
    return out

  with tempfile.TemporaryDirectory() as td:
    pdf_path = os.path.join(td, "input.pdf")
    with open(pdf_path, "wb") as f:
      f.write(pdf_bytes)

    images = render_images(pdf_path)

    for idx, img in enumerate(images):
      page_no = idx + 1
      page_text = ""
      for processed in preprocess_candidates(img):
        try:
          result = ocr.ocr(processed, cls=True)
        except TypeError:
          result = ocr.ocr(processed)
        if not result or (isinstance(result, list) and result[0] is None):
          continue
        res0 = result[0] if isinstance(result, list) and result else result
        lines = sort_text_lines(res0)
        if lines:
          page_text = "\n".join(lines).strip()
          break

      chapters.append(
        {
          "volume_no": 1,
          "volume_title": None,
          "chapter_no": page_no,
          "chapter_title": f"第{page_no}页",
          "content": page_text,
          "sort_order": page_no,
        }
      )
      if page_text:
        full.append(page_text)

  return {"chapters": chapters, "text": "\n\n".join(full).strip()}


@app.post("/ocr_slices")
def ocr_slices(req: OcrSlicesRequest) -> Dict[str, Any]:
  manifest_url = (req.manifest_url or "").strip()
  if not manifest_url:
    raise HTTPException(status_code=400, detail="missing manifest_url")

  _require_true_ocr_deps()

  try:
    manifest_bytes = _download_bytes(manifest_url, timeout=60)
  except HTTPException:
    raise
  except Exception as e:
    raise HTTPException(status_code=502, detail=str(e))

  try:
    manifest = json.loads(manifest_bytes.decode("utf-8", errors="replace"))
  except Exception as e:
    raise HTTPException(status_code=502, detail=f"invalid manifest json: {e}")

  if not isinstance(manifest, dict):
    raise HTTPException(status_code=502, detail="invalid manifest json")

  raw_pages = manifest.get("pages")
  if not isinstance(raw_pages, list) or not raw_pages:
    raise HTTPException(status_code=502, detail="manifest pages empty")

  pages: List[Dict[str, Any]] = []
  for p in raw_pages:
    if not isinstance(p, dict):
      continue
    no = p.get("no")
    url = p.get("url")
    if not isinstance(no, int) or no <= 0:
      continue
    if not isinstance(url, str) or not url.strip():
      continue
    pages.append({"no": int(no), "url": str(url).strip()})

  if not pages:
    raise HTTPException(status_code=502, detail="manifest pages empty")

  pages.sort(key=lambda x: int(x["no"]))

  first_page = int(req.page_start) if isinstance(req.page_start, int) and req.page_start > 0 else None
  last_page = int(req.page_end) if isinstance(req.page_end, int) and req.page_end > 0 else None
  if first_page is not None:
    pages = [p for p in pages if int(p["no"]) >= first_page]
  if last_page is not None:
    pages = [p for p in pages if int(p["no"]) <= last_page]

  max_pages = int(req.max_pages) if isinstance(req.max_pages, int) and req.max_pages > 0 else None
  if max_pages:
    pages = pages[:max_pages]

  if not pages:
    raise HTTPException(status_code=400, detail="no pages selected")

  try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
  except Exception as e:
    raise HTTPException(status_code=500, detail={"error": "真实 OCR 运行依赖不可用", "detail": str(e)})

  ocr = _get_paddle_ocr()
  if ocr is None:
    raise HTTPException(
      status_code=500,
      detail={"error": "PaddleOCR 未就绪", "detail": _paddle_ocr_error or "paddleocr/paddlepaddle 未安装或初始化失败"},
    )

  def preprocess_candidates(pil_image: Any) -> List[Any]:
    try:
      if hasattr(pil_image, "convert"):
        pil_image = pil_image.convert("RGB")
    except Exception:
      pass
    img_np = np.array(pil_image)
    if img_np.dtype != np.uint8:
      img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    if len(img_np.shape) == 2:
      img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
      img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
      img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
      img_cv = cv2.cvtColor(img_np[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
      raise HTTPException(status_code=500, detail=f"unsupported image shape: {tuple(getattr(img_np, 'shape', ()))}")
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
      gray,
      255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY,
      11,
      2,
    )
    return [img_cv, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)]

  def sort_text_lines(res0: Any) -> List[str]:
    def is_box(v: Any) -> bool:
      if hasattr(v, "tolist"):
        try:
          v = v.tolist()
        except Exception:
          return False
      if not isinstance(v, (list, tuple)) or len(v) != 4:
        return False
      for p in v:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
          return False
        if not isinstance(p[0], (int, float)) or not isinstance(p[1], (int, float)):
          return False
      return True

    def box_pos(box: Any) -> Optional[Dict[str, float]]:
      if not is_box(box):
        return None
      xs = [float(p[0]) for p in box]
      ys = [float(p[1]) for p in box]
      return {"x": float(min(xs)), "y": float(min(ys))}

    lines: List[Dict[str, Any]] = []
    items: Any = []
    if isinstance(res0, dict):
      texts = res0.get("rec_texts")
      polys = res0.get("rec_polys") or res0.get("dt_polys") or res0.get("dt_boxes")
      if isinstance(texts, list) and isinstance(polys, list):
        tmp: List[Any] = []
        for t, b in zip(texts, polys):
          if not isinstance(t, str) or not t.strip():
            continue
          if hasattr(b, "tolist"):
            try:
              b = b.tolist()
            except Exception:
              b = None
          tmp.append([b, (t, 1.0)])
        items = tmp
      else:
        items = []
    elif isinstance(res0, list):
      items = res0
    else:
      items = []
    for it in items:
      text: Optional[str] = None
      box: Any = None
      if isinstance(it, dict):
        t = it.get("text")
        if isinstance(t, str) and t.strip():
          text = t.strip()
        b = it.get("box") or it.get("dt_boxes") or it.get("points")
        if is_box(b):
          box = b
      elif isinstance(it, (list, tuple)):
        if len(it) >= 2 and is_box(it[0]):
          box = it[0]
          second = it[1]
          if isinstance(second, (list, tuple)) and len(second) >= 1 and isinstance(second[0], str) and second[0].strip():
            text = second[0].strip()
          elif isinstance(second, str) and second.strip():
            text = second.strip()
          elif len(it) >= 3 and isinstance(it[1], str) and it[1].strip():
            text = it[1].strip()
        elif len(it) >= 1 and isinstance(it[0], str) and it[0].strip():
          text = it[0].strip()
      if not text:
        continue
      pos = box_pos(box)
      if pos:
        lines.append({"text": text, "x": pos["x"], "y": pos["y"]})
      else:
        lines.append({"text": text, "x": float("inf"), "y": float("inf")})

    lines.sort(key=lambda k: (-k["x"], k["y"]))
    return [d["text"] for d in lines]

  chapters: List[Dict[str, Any]] = []
  full: List[str] = []

  for idx, p in enumerate(pages):
    page_no = int(p["no"])
    url = str(p["url"])
    img_bytes = _download_bytes(url, timeout=120)
    try:
      from PIL import ImageFile  # type: ignore
    except Exception:
      ImageFile = None  # type: ignore
    if ImageFile is not None:
      ImageFile.LOAD_TRUNCATED_IMAGES = True  # type: ignore
    img = Image.open(io.BytesIO(img_bytes))
    page_text = ""
    for processed in preprocess_candidates(img):
      try:
        result = ocr.ocr(processed, cls=True)
      except TypeError:
        result = ocr.ocr(processed)
      if not result or (isinstance(result, list) and result[0] is None):
        continue
      res0 = result[0] if isinstance(result, list) and result else result
      lines = sort_text_lines(res0)
      if lines:
        page_text = "\n".join(lines).strip()
        break

    chapters.append(
      {
        "volume_no": 1,
        "volume_title": None,
        "chapter_no": page_no,
        "chapter_title": f"第{page_no}页",
        "content": page_text,
        "sort_order": idx + 1,
      }
    )
    if page_text:
      full.append(page_text)

  return {"chapters": chapters, "text": "\n\n".join(full).strip()}


@app.post("/slice_upload")
def slice_upload(req: SliceUploadRequest, request: Request) -> Dict[str, Any]:
  pdf_url = (req.pdf_url or "").strip()
  if not pdf_url:
    raise HTTPException(status_code=400, detail="missing pdf_url")

  _require_pdf_render_deps()
  cfg = _get_supabase_storage_config_from_req(req, headers={k.lower(): v for k, v in dict(request.headers).items()})

  bucket = (req.bucket or "library_pages").strip() or "library_pages"
  prefix = (req.prefix or "").strip().strip("/")
  if not prefix:
    if req.book_id and str(req.book_id).strip():
      prefix = f"books/{str(req.book_id).strip()}"
    else:
      prefix = f"books/{abs(hash(pdf_url))}"

  pdf_bytes = _download_pdf(pdf_url)
  if not pdf_bytes:
    raise HTTPException(status_code=502, detail="empty pdf")

  try:
    from PIL import Image  # type: ignore
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"missing pillow: {e}")

  dpi = int(req.dpi) if isinstance(req.dpi, int) and req.dpi > 0 else 200
  quality = int(req.quality) if isinstance(req.quality, int) and 1 <= req.quality <= 100 else 82
  max_pages = int(req.max_pages) if isinstance(req.max_pages, int) and req.max_pages > 0 else None
  first_page = int(req.page_start) if isinstance(req.page_start, int) and req.page_start > 0 else None
  last_page = int(req.page_end) if isinstance(req.page_end, int) and req.page_end > 0 else None

  def render_images(pdf_path: str) -> List[Any]:
    pdf2image_ok = importlib.util.find_spec("pdf2image") is not None
    poppler_ok = bool(shutil.which("pdftoppm") or shutil.which("pdfinfo"))
    if pdf2image_ok and poppler_ok:
      from pdf2image import convert_from_path  # type: ignore

      poppler_path = os.environ.get("POPPLER_PATH") or None
      images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        poppler_path=poppler_path,
      )
      if max_pages:
        images = images[:max_pages]
      return images

    try:
      import pypdfium2 as pdfium  # type: ignore
    except Exception as e:
      raise HTTPException(status_code=500, detail={"error": "缺少 PDF 渲染依赖", "detail": str(e)})

    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    start = max(1, int(first_page)) if first_page else 1
    end = min(total, int(last_page)) if last_page else total
    page_indices = list(range(start - 1, end))
    if max_pages:
      page_indices = page_indices[:max_pages]

    scale = float(max(dpi, 72)) / 72.0
    out: List[Any] = []
    for i in page_indices:
      page = pdf.get_page(i)
      try:
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        out.append(pil if isinstance(pil, Image.Image) else pil)
      finally:
        page.close()
    pdf.close()
    return out

  pages: List[Dict[str, Any]] = []
  with tempfile.TemporaryDirectory() as td:
    pdf_path = os.path.join(td, "input.pdf")
    with open(pdf_path, "wb") as f:
      f.write(pdf_bytes)

    images = render_images(pdf_path)
    for idx, img in enumerate(images):
      page_no = idx + 1
      buf = io.BytesIO()
      img.save(buf, format="WEBP", quality=quality, method=6)
      payload = buf.getvalue()
      page_path = f"{prefix}/p{page_no:04d}.webp"
      _upload_storage_object(
        cfg["supabase_url"],
        cfg["apikey"],
        cfg["authorization"],
        bucket,
        page_path,
        payload,
        "image/webp",
      )
      pages.append(
        {
          "no": page_no,
          "path": page_path,
          "url": _public_object_url(cfg["supabase_url"], bucket, page_path),
          "width": getattr(img, "width", None),
          "height": getattr(img, "height", None),
          "bytes": len(payload),
        }
      )

  manifest = {
    "bucket": bucket,
    "prefix": prefix,
    "page_count": len(pages),
    "pages": pages,
  }
  manifest_path = f"{prefix}/manifest.json"
  _upload_storage_object(
    cfg["supabase_url"],
    cfg["apikey"],
    cfg["authorization"],
    bucket,
    manifest_path,
    json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
    "application/json",
  )
  manifest_url = _public_object_url(cfg["supabase_url"], bucket, manifest_path)

  return {
    "bucket": bucket,
    "prefix": prefix,
    "page_count": len(pages),
    "manifest_path": manifest_path,
    "manifest_url": manifest_url,
  }


def _slice_upload_bytes(
  *,
  pdf_bytes: bytes,
  book_id: Optional[str],
  pdf_url: Optional[str],
  title: Optional[str],
  dpi: int,
  quality: int,
  max_pages: Optional[int],
  page_start: Optional[int],
  page_end: Optional[int],
  bucket: str,
  prefix: Optional[str],
  supabase_url: Optional[str],
  supabase_service_role_key: Optional[str],
  request_headers: Optional[Dict[str, str]],
) -> Dict[str, Any]:
  _require_pdf_render_deps()
  cfg = _get_supabase_storage_config_from_req(
    SliceUploadRequest(
      pdf_url=pdf_url or "file://upload.pdf",
      book_id=book_id,
      title=title,
      dpi=dpi,
      quality=quality,
      max_pages=max_pages,
      page_start=page_start,
      page_end=page_end,
      bucket=bucket,
      prefix=prefix,
      supabase_url=supabase_url,
      supabase_service_role_key=supabase_service_role_key,
    )
    ,
    headers=request_headers,
  )

  bucket = (bucket or "library_pages").strip() or "library_pages"
  prefix = (prefix or "").strip().strip("/")
  if not prefix:
    if book_id and str(book_id).strip():
      prefix = f"books/{str(book_id).strip()}"
    else:
      seed = (pdf_url or "") if pdf_url else str(len(pdf_bytes))
      prefix = f"books/{abs(hash(seed))}"

  try:
    from PIL import Image  # type: ignore
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"missing pillow: {e}")

  dpi = int(dpi) if isinstance(dpi, int) and dpi > 0 else 200
  quality = int(quality) if isinstance(quality, int) and 1 <= quality <= 100 else 82
  max_pages = int(max_pages) if isinstance(max_pages, int) and max_pages > 0 else None
  first_page = int(page_start) if isinstance(page_start, int) and page_start > 0 else None
  last_page = int(page_end) if isinstance(page_end, int) and page_end > 0 else None

  def render_images(pdf_path: str) -> List[Any]:
    pdf2image_ok = importlib.util.find_spec("pdf2image") is not None
    poppler_ok = bool(shutil.which("pdftoppm") or shutil.which("pdfinfo"))
    if pdf2image_ok and poppler_ok:
      from pdf2image import convert_from_path  # type: ignore

      poppler_path = os.environ.get("POPPLER_PATH") or None
      images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        poppler_path=poppler_path,
      )
      if max_pages:
        images = images[:max_pages]
      return images

    try:
      import pypdfium2 as pdfium  # type: ignore
    except Exception as e:
      raise HTTPException(status_code=500, detail={"error": "缺少 PDF 渲染依赖", "detail": str(e)})

    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    start = max(1, int(first_page)) if first_page else 1
    end = min(total, int(last_page)) if last_page else total
    page_indices = list(range(start - 1, end))
    if max_pages:
      page_indices = page_indices[:max_pages]

    scale = float(max(dpi, 72)) / 72.0
    out: List[Any] = []
    for i in page_indices:
      page = pdf.get_page(i)
      try:
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        out.append(pil if isinstance(pil, Image.Image) else pil)
      finally:
        page.close()
    pdf.close()
    return out

  pages: List[Dict[str, Any]] = []
  with tempfile.TemporaryDirectory() as td:
    pdf_path = os.path.join(td, "input.pdf")
    with open(pdf_path, "wb") as f:
      f.write(pdf_bytes)

    images = render_images(pdf_path)
    for idx, img in enumerate(images):
      page_no = idx + 1
      buf = io.BytesIO()
      img.save(buf, format="WEBP", quality=quality, method=6)
      payload = buf.getvalue()
      page_path = f"{prefix}/p{page_no:04d}.webp"
      _upload_storage_object(
        cfg["supabase_url"],
        cfg["apikey"],
        cfg["authorization"],
        bucket,
        page_path,
        payload,
        "image/webp",
      )
      pages.append(
        {
          "no": page_no,
          "path": page_path,
          "url": _public_object_url(cfg["supabase_url"], bucket, page_path),
          "width": getattr(img, "width", None),
          "height": getattr(img, "height", None),
          "bytes": len(payload),
        }
      )

  manifest = {
    "bucket": bucket,
    "prefix": prefix,
    "page_count": len(pages),
    "pages": pages,
  }
  manifest_path = f"{prefix}/manifest.json"
  _upload_storage_object(
    cfg["supabase_url"],
    cfg["apikey"],
    cfg["authorization"],
    bucket,
    manifest_path,
    json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
    "application/json",
  )
  manifest_url = _public_object_url(cfg["supabase_url"], bucket, manifest_path)

  return {
    "bucket": bucket,
    "prefix": prefix,
    "page_count": len(pages),
    "manifest_path": manifest_path,
    "manifest_url": manifest_url,
  }


@app.post("/slice_upload_file")
async def slice_upload_file(
  request: Request,
  book_id: Optional[str] = None,
  title: Optional[str] = None,
  dpi: int = 200,
  quality: int = 82,
  max_pages: Optional[int] = None,
  page_start: Optional[int] = None,
  page_end: Optional[int] = None,
  bucket: str = "library_pages",
  prefix: Optional[str] = None,
  supabase_url: Optional[str] = None,
  supabase_service_role_key: Optional[str] = None,
) -> Dict[str, Any]:
  content_type = str(request.headers.get("content-type") or "").lower()
  if content_type and "application/pdf" not in content_type:
    raise HTTPException(status_code=400, detail="only supports application/pdf")
  pdf_bytes = await request.body()
  if not pdf_bytes:
    raise HTTPException(status_code=400, detail="empty pdf")
  return _slice_upload_bytes(
    pdf_bytes=pdf_bytes,
    book_id=book_id,
    pdf_url=None,
    title=title,
    dpi=dpi,
    quality=quality,
    max_pages=max_pages,
    page_start=page_start,
    page_end=page_end,
    bucket=bucket,
    prefix=prefix,
    supabase_url=supabase_url,
    supabase_service_role_key=supabase_service_role_key,
    request_headers={k.lower(): v for k, v in dict(request.headers).items()},
  )


if __name__ == "__main__":
  import argparse
  import sys
  import uvicorn

  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="cmd")

  serve = sub.add_parser("serve")
  serve.add_argument("--host", default=os.environ.get("OCR_HOST", "0.0.0.0"))
  serve.add_argument("--port", type=int, default=int(os.environ.get("OCR_PORT", "8008")))

  local = sub.add_parser("local")
  local.add_argument("pdf", type=str)
  local.add_argument("--book-id", type=str, default="")
  local.add_argument("--title", type=str, default="")
  local.add_argument("--author", type=str, default="")
  local.add_argument("--dynasty", type=str, default="")
  local.add_argument("--category", type=str, default="")
  local.add_argument("--status", type=str, default="")
  local.add_argument("--description", type=str, default="")
  local.add_argument("--cover-url", type=str, default="")
  local.add_argument("--dpi", type=int, default=200)
  local.add_argument("--quality", type=int, default=82)
  local.add_argument("--max-pages", type=int, default=0)
  local.add_argument("--page-start", type=int, default=0)
  local.add_argument("--page-end", type=int, default=0)
  local.add_argument("--bucket", type=str, default="library_pages")
  local.add_argument("--prefix", type=str, default="")
  local.add_argument("--supabase-url", type=str, default="")
  local.add_argument("--service-role-key", type=str, default="")
  local.add_argument("--create-book", action="store_true")
  local.add_argument("--skip-db", action="store_true")
  local.add_argument("--skip-ocr", action="store_true")

  args = parser.parse_args()

  cmd = getattr(args, "cmd", None)
  if cmd in (None, "serve"):
    host = getattr(args, "host", os.environ.get("OCR_HOST", "0.0.0.0"))
    port = int(getattr(args, "port", int(os.environ.get("OCR_PORT", "8008"))))
    uvicorn.run(app, host=host, port=port)
    raise SystemExit(0)

  if cmd != "local":
    raise SystemExit(2)

  supabase_url = str(getattr(args, "supabase_url", "") or "").strip().rstrip("/") or _get_supabase_url()
  service_role_key = str(getattr(args, "service_role_key", "") or "").strip() or str(os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
  if not service_role_key:
    raise SystemExit("missing service role key")

  pdf_path = str(getattr(args, "pdf", "") or "").strip()
  if not pdf_path:
    raise SystemExit("missing pdf path")
  try:
    with open(pdf_path, "rb") as f:
      pdf_bytes = f.read()
  except Exception as e:
    raise SystemExit(str(e))
  if not pdf_bytes:
    raise SystemExit("empty pdf")

  book = _ensure_library_book(
    supabase_url=supabase_url,
    service_role_key=service_role_key,
    book_id=str(getattr(args, "book_id", "") or "").strip() or None,
    title=str(getattr(args, "title", "") or "").strip() or None,
    author=str(getattr(args, "author", "") or "").strip() or None,
    dynasty=str(getattr(args, "dynasty", "") or "").strip() or None,
    category=str(getattr(args, "category", "") or "").strip() or None,
    status=str(getattr(args, "status", "") or "").strip() or None,
    description=str(getattr(args, "description", "") or "").strip() or None,
    cover_url=str(getattr(args, "cover_url", "") or "").strip() or None,
    create_if_missing=bool(getattr(args, "create_book", False)),
  )
  book_id = str(book.get("id") or "").strip()
  if not book_id:
    raise SystemExit("book id missing after create")

  dpi = int(getattr(args, "dpi", 200) or 200)
  quality = int(getattr(args, "quality", 82) or 82)
  max_pages = int(getattr(args, "max_pages", 0) or 0) or None
  page_start = int(getattr(args, "page_start", 0) or 0) or None
  page_end = int(getattr(args, "page_end", 0) or 0) or None
  bucket = str(getattr(args, "bucket", "library_pages") or "library_pages").strip() or "library_pages"
  prefix = str(getattr(args, "prefix", "") or "").strip().strip("/") or f"books/{book_id}"

  sliced = _slice_upload_bytes(
    pdf_bytes=pdf_bytes,
    book_id=book_id,
    pdf_url=None,
    title=str(getattr(args, "title", "") or "").strip() or str(book.get("title") or "").strip() or None,
    dpi=dpi,
    quality=quality,
    max_pages=max_pages,
    page_start=page_start,
    page_end=page_end,
    bucket=bucket,
    prefix=prefix,
    supabase_url=supabase_url,
    supabase_service_role_key=service_role_key,
    request_headers=None,
  )
  manifest_url = str(sliced.get("manifest_url") or "").strip()
  if not manifest_url:
    raise SystemExit("missing manifest_url")

  chapters: List[Dict[str, Any]] = []
  text = ""
  ocr_json_url: Optional[str] = None
  if not bool(getattr(args, "skip_ocr", False)):
    ocr_result = ocr_slices(
      OcrSlicesRequest(
        manifest_url=manifest_url,
        book_id=book_id,
        title=str(getattr(args, "title", "") or "").strip() or None,
        max_pages=max_pages,
        page_start=page_start,
        page_end=page_end,
      )
    )
    chapters = list(ocr_result.get("chapters") or [])
    text = str(ocr_result.get("text") or "")

    ocr_payload = {
      "book_id": book_id,
      "manifest_url": manifest_url,
      "chapters": chapters,
      "text": text,
    }
    ocr_path = f"{prefix}/ocr.json"
    _upload_storage_object(
      supabase_url,
      service_role_key,
      f"Bearer {service_role_key}",
      bucket,
      ocr_path,
      json.dumps(ocr_payload, ensure_ascii=False).encode("utf-8"),
      "application/json",
    )
    ocr_json_url = _public_object_url(supabase_url, bucket, ocr_path)

  if not bool(getattr(args, "skip_db", False)):
    src_payload = book.get("source_payload")
    if isinstance(src_payload, dict):
      merged: Dict[str, Any] = dict(src_payload)
    else:
      merged = {}
    merged["slices"] = {
      "manifest_url": manifest_url,
      "bucket": bucket,
      "prefix": prefix,
      "page_count": sliced.get("page_count"),
    }
    if ocr_json_url is not None:
      merged["ocr"] = {
        "json_url": ocr_json_url,
        "generated_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
      }
    _update_book_source_payload(supabase_url=supabase_url, service_role_key=service_role_key, book_id=book_id, next_source_payload=merged)
    if chapters:
      _replace_book_contents(supabase_url=supabase_url, service_role_key=service_role_key, book_id=book_id, chapters=chapters)

  sys.stdout.write(json.dumps({"ok": True, "book_id": book_id, "manifest_url": manifest_url, "ocr_json_url": ocr_json_url}, ensure_ascii=False) + "\n")
