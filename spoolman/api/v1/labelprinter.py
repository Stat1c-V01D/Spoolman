"""Server-side label printing on a Brother QL label printer (printserver fork addition).

The new web client renders labels with the regular label designer (same PNG raster as
"Save as PNG") and POSTs them here; the server rasterizes them for the Brother QL and
sends them over the network - no browser print dialog, no client-side driver.

Configuration (environment):
    SPOOLMAN_LABEL_PRINTER        tcp://<ip>:9100   (unset = feature disabled)
    SPOOLMAN_LABEL_PRINTER_MODEL  brother_ql model, default QL-820NWB
    SPOOLMAN_LABEL_PRINTER_LABEL  brother_ql label of the inserted roll, default 62x29
"""

import asyncio
import base64
import binascii
import io
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/labelprinter",
    tags=["labelprinter"],
)

# ruff: noqa: D103

PRINTER = os.getenv("SPOOLMAN_LABEL_PRINTER") or None
MODEL = os.getenv("SPOOLMAN_LABEL_PRINTER_MODEL", "QL-820NWB")
LABEL = os.getenv("SPOOLMAN_LABEL_PRINTER_LABEL", "62x29")

_print_lock = asyncio.Lock()


class LabelPrinterInfo(BaseModel):
    enabled: bool = Field(description="Whether a label printer is configured on the server.")
    printer: str | None = Field(description="Printer address, e.g. tcp://192.168.1.50:9100.")
    model: str = Field(description="brother_ql printer model.")
    label: str = Field(description="brother_ql label identifier of the inserted roll.")


class LabelImage(BaseModel):
    name: str = Field(default="label", max_length=200)
    png_base64: str = Field(max_length=20_000_000, description="PNG raster of one label, base64.")


class PrintRequest(BaseModel):
    labels: list[LabelImage] = Field(min_length=1, max_length=200)
    copies: int = Field(default=1, ge=1, le=50)


class PrintResult(BaseModel):
    printed: int


@router.get("", name="Get label printer")
async def get_label_printer() -> LabelPrinterInfo:
    return LabelPrinterInfo(enabled=PRINTER is not None, printer=PRINTER, model=MODEL, label=LABEL)


def _label_dots() -> tuple[int, int]:
    from brother_ql.labels import LabelsManager  # noqa: PLC0415 - optional dependency, loaded on use

    for lbl in LabelsManager().iter_elements():
        if lbl.identifier == LABEL:
            return lbl.dots_printable
    msg = f"Unknown label '{LABEL}' (SPOOLMAN_LABEL_PRINTER_LABEL)"
    raise ValueError(msg)


def _fit(png: bytes, dots: tuple[int, int]):  # noqa: ANN202
    """Fit a label raster to the printable area: exact box for die-cut, width for endless."""
    from PIL import Image  # noqa: PLC0415

    tw, th = dots
    im = Image.open(io.BytesIO(png)).convert("L")
    w, h = im.size
    if th:
        if (w >= h) != (tw >= th):
            im = im.rotate(90, expand=True)
            w, h = im.size
        scale = min(tw / w, th / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        im = im.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("L", (tw, th), 255)
        canvas.paste(im, ((tw - nw) // 2, (th - nh) // 2))
        return canvas
    return im.resize((tw, max(1, round(h * tw / w))), Image.LANCZOS)


def _print_blocking(images: list[bytes], copies: int) -> int:
    from brother_ql.backends.helpers import send  # noqa: PLC0415
    from brother_ql.conversion import convert  # noqa: PLC0415
    from brother_ql.raster import BrotherQLRaster  # noqa: PLC0415

    dots = _label_dots()
    fitted = [_fit(png, dots) for png in images]
    jobs = [im for im in fitted for _ in range(copies)]
    qlr = BrotherQLRaster(MODEL)
    qlr.exception_on_warning = True
    instructions = convert(
        qlr=qlr,
        images=jobs,
        label=LABEL,
        rotate="0",
        threshold=70.0,
        dither=False,
        compress=False,
        red=LABEL.endswith("red"),
        dpi_600=False,
        hq=True,
        cut=True,
    )
    send(instructions=instructions, printer_identifier=PRINTER, backend_identifier="network", blocking=True)
    return len(jobs)


@router.post(
    "/print",
    name="Print labels on the server label printer",
    responses={503: {"description": "No label printer configured"}, 502: {"description": "Printing failed"}},
)
async def print_labels(body: PrintRequest) -> PrintResult:
    if PRINTER is None:
        raise HTTPException(status_code=503, detail="No label printer configured (SPOOLMAN_LABEL_PRINTER).")
    try:
        images = [base64.b64decode(lbl.png_base64, validate=True) for lbl in body.labels]
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}") from e

    async with _print_lock:
        try:
            printed = await asyncio.to_thread(_print_blocking, images, body.copies)
        except Exception as e:
            logger.exception("Label printing failed")
            raise HTTPException(status_code=502, detail=f"Label printing failed: {e}") from e
    logger.info("Printed %d label(s) on %s (%s, %s)", printed, PRINTER, MODEL, LABEL)
    return PrintResult(printed=printed)
