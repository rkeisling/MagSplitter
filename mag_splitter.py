#!/usr/bin/env python3
"""Utilities for converting CBZ/CBR/PDF magazine issues into page images and layouts."""

from __future__ import annotations

import argparse
import dataclasses
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Set, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat, UnidentifiedImageError

CANVAS_SIZE = (1024, 600)
CANVAS_BACKGROUND = (8, 8, 8)
INFO_PANEL_BACKGROUND = (20, 20, 20)
INFO_PANEL_TEXT_COLOR = (107, 225, 5)
IMAGE_BORDER_COLOR = (40, 40, 40)
IMAGE_BORDER_WIDTH = 4
INFO_PANEL_WIDTH = 440
PANEL_MARGIN = 20
IMAGE_LEFT_OFFSET = 60
IMAGE_PANEL_GAP = 18
IMAGE_TOP_MARGIN = 4
IMAGE_BOTTOM_MARGIN = 4
MAX_FONT_SIZE = 48
MIN_FONT_SIZE = 16
FONT_SIZE_REDUCTION = 10
LINE_SPACING = 18
SEPARATOR_COLOR = INFO_PANEL_TEXT_COLOR
SEPARATOR_OFFSET = 20
SEPARATOR_WIDTH = 1
RANDOM_NAME_LENGTH = 16
ZIP_EXTENSIONS = {".cbz", ".zip"}
RAR_EXTENSIONS = {".cbr", ".rar"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = ZIP_EXTENSIONS | RAR_EXTENSIONS | PDF_EXTENSIONS
PDF_RENDER_DPI = 200
DEFAULT_INTEREST_THRESHOLD = 0.0
DEFAULT_PLAIN_THRESHOLD = 0.30
DEFAULT_MIN_ASPECT_RATIO = 0.55
DEFAULT_MAX_ASPECT_RATIO = 0.85
EDGE_ENHANCE_SIZE = (256, 256)
TILE_GRID_SIZE = (4, 4)
MARGIN_CROP_PERCENT = 0.05
EMPTY_TILE_THRESHOLD = 0.02
CONTENT_TILE_MIN_STD = 15
DEFAULT_SPLIT_SPREADS = True
DEFAULT_SPREAD_THRESHOLD = 1.2


@dataclasses.dataclass
class IssueMetadata:
    title: str
    issue_number: str | None
    month: str | None
    year: str | None

    @property
    def display_lines(self) -> List[str]:
        lines: List[str] = [self.title]
        if self.issue_number:
            lines.append(f"Issue {self.issue_number}")
        if self.month and self.year:
            lines.append(f"{self.month} {self.year}")
        elif self.year:
            lines.append(self.year)
        return [line for line in lines if line.strip()]


@dataclasses.dataclass
class WrappedLine:
    text: str
    group_index: int


def find_publication_files(sources: Iterable[str]) -> Iterator[Path]:
    for src in sources:
        path = Path(src)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield candidate


def numeric_key_from_string(value: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", value)
    number = int(match.group(1)) if match else 10**9
    return (number, value.lower())


def page_sort_key(path: Path) -> Tuple[int, str]:
    return numeric_key_from_string(path.stem)


def sanitise_issue_name(name: str) -> str:
    return re.sub(r"[\\/]+", "-", name).strip()


def is_double_page_spread(image: Image.Image, threshold: float) -> bool:
    """Check if an image is a double-page spread based on aspect ratio."""
    width, height = image.size
    if height == 0:
        return False
    aspect_ratio = width / height
    return aspect_ratio > threshold


def find_center_seam(image: Image.Image) -> int:
    """
    Find the center seam (binding/fold) in a double-page spread.
    Returns the x-coordinate of the center seam.
    Uses horizontal gradient analysis to detect the darkest vertical line.
    """
    try:
        import cv2
        import numpy as np

        img_array = np.array(image.convert("L"))
        sobel_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        column_sums = np.abs(sobel_x).sum(axis=0)

        width = image.size[0]
        center_region_start = int(width * 0.4)
        center_region_end = int(width * 0.6)
        center_region = column_sums[center_region_start:center_region_end]

        local_max = np.argmax(center_region)
        seam_x = center_region_start + local_max

        return seam_x
    except ImportError:
        return image.size[0] // 2


def split_double_page(image: Image.Image, split_spreads: bool, spread_threshold: float) -> List[Image.Image]:
    """
    Split a double-page spread into two single pages if needed.
    Returns a list containing either [original_image] or [left_page, right_page].
    """
    if not split_spreads or not is_double_page_spread(image, spread_threshold):
        return [image]

    seam_x = find_center_seam(image)
    width, height = image.size

    left_page = image.crop((0, 0, seam_x, height))
    right_page = image.crop((seam_x, 0, width, height))

    return [left_page, right_page]


def split_zip_archive(archive_path: Path, output_root: Path, split_spreads: bool = True, spread_threshold: float = 1.2) -> List[Path]:
    import zipfile

    issue_name = sanitise_issue_name(archive_path.stem)
    issue_dir = output_root / issue_name
    issue_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths: List[Path] = []
    page_index = 1
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.namelist() if not m.endswith("/")]
        members.sort(key=numeric_key_from_string)
        for member in members:
            with archive.open(member) as member_file:
                with Image.open(member_file) as img:
                    img = img.convert("RGB")
                    pages = split_double_page(img, split_spreads, spread_threshold)
                    for page in pages:
                        output_path = issue_dir / f"page{page_index:03d}.jpg"
                        page.save(output_path, format="JPEG", quality=95, optimize=True)
                        extracted_paths.append(output_path)
                        page_index += 1
    return extracted_paths


ExtractorCommandBuilder = Callable[[Path, Path], List[str]]

RAR_TOOL_BUILDERS: Tuple[Tuple[str, ExtractorCommandBuilder], ...] = (
    ("unrar", lambda archive, dest: ["unrar", "x", "-inul", str(archive), str(dest)]),
    ("7z", lambda archive, dest: ["7z", "x", "-y", f"-o{dest}", str(archive)]),
    ("bsdtar", lambda archive, dest: ["bsdtar", "-xf", str(archive), "-C", str(dest)]),
)


def detect_rar_extractor() -> Tuple[str, ExtractorCommandBuilder] | None:
    for tool, builder in RAR_TOOL_BUILDERS:
        if shutil.which(tool):
            return tool, builder
    return None


def split_rar_archive(archive_path: Path, output_root: Path, split_spreads: bool = True, spread_threshold: float = 1.2) -> List[Path]:
    extractor = detect_rar_extractor()
    if extractor is None:
        raise SystemExit(
            "Processing CBR files requires one of: unrar, 7z, bsdtar. Install a compatible extractor and retry."
        )
    tool, command_builder = extractor

    issue_name = sanitise_issue_name(archive_path.stem)
    issue_dir = output_root / issue_name
    issue_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths: List[Path] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        command = command_builder(archive_path, temp_path)
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError as exc:
            raise SystemExit(f"Failed to execute {tool} for extracting {archive_path}: {exc}") from exc
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="ignore").strip()
            raise SystemExit(f"Failed to extract {archive_path} using {tool}: {stderr or 'unknown error'}")

        page_index = 1
        for candidate in sorted((p for p in temp_path.rglob("*") if p.is_file()), key=page_sort_key):
            try:
                with Image.open(candidate) as img:
                    img = img.convert("RGB")
                    pages = split_double_page(img, split_spreads, spread_threshold)
                    for page in pages:
                        output_path = issue_dir / f"page{page_index:03d}.jpg"
                        page.save(output_path, format="JPEG", quality=95, optimize=True)
                        extracted_paths.append(output_path)
                        page_index += 1
            except UnidentifiedImageError:
                continue

    if not extracted_paths:
        raise SystemExit(f"No images found within {archive_path}.")
    return extracted_paths


def split_pdf_document(pdf_path: Path, output_root: Path, split_spreads: bool = True, spread_threshold: float = 1.2) -> List[Path]:
    if shutil.which("pdftoppm") is None:
        raise SystemExit(
            "Processing PDF files requires pdftoppm (Poppler). Install poppler-utils or an equivalent package."
        )

    issue_name = sanitise_issue_name(pdf_path.stem)
    issue_dir = output_root / issue_name
    issue_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths: List[Path] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        prefix = temp_root / "page"
        command = [
            "pdftoppm",
            "-jpeg",
            "-r",
            str(PDF_RENDER_DPI),
            str(pdf_path),
            str(prefix),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="ignore").strip()
            raise SystemExit(f"Failed to render {pdf_path} using pdftoppm: {stderr or 'unknown error'}")

        page_index = 1
        generated = sorted(temp_root.glob("page-*.jpg"), key=page_sort_key)
        if not generated:
            raise SystemExit(f"No images were produced when converting {pdf_path}.")

        for candidate in generated:
            with Image.open(candidate) as img:
                img = img.convert("RGB")
                pages = split_double_page(img, split_spreads, spread_threshold)
                for page in pages:
                    output_path = issue_dir / f"page{page_index:03d}.jpg"
                    page.save(output_path, format="JPEG", quality=95, optimize=True)
                    extracted_paths.append(output_path)
                    page_index += 1

    return extracted_paths


def generate_random_name(used: Set[str]) -> str:
    while True:
        token = uuid.uuid4().hex[:RANDOM_NAME_LENGTH]
        if token in used:
            continue
        used.add(token)
        return token


def split_publication(source_path: Path, output_root: Path, split_spreads: bool = True, spread_threshold: float = 1.2) -> List[Path]:
    suffix = source_path.suffix.lower()
    if suffix in ZIP_EXTENSIONS:
        return split_zip_archive(source_path, output_root, split_spreads, spread_threshold)
    if suffix in RAR_EXTENSIONS:
        return split_rar_archive(source_path, output_root, split_spreads, spread_threshold)
    if suffix in PDF_EXTENSIONS:
        return split_pdf_document(source_path, output_root, split_spreads, spread_threshold)
    raise SystemExit(f"Unsupported file extension {suffix} for {source_path}.")


def parse_issue_metadata(issue_dir: Path) -> IssueMetadata:
    name = issue_dir.name.strip()

    patterns = [
        re.compile(
            r"^(?P<title>.+?)\s+(?P<year>\d{4})\s+(?P<number>\d{1,2})$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?P<title>.+?)\s*(?:Issue\s*(?P<number>\d+))?\s*(?:\((?P<month>[A-Za-z]+)\s*(?P<year>\d{4})\))?$",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.match(name)
        if not match:
            continue
        title = match.group("title").strip()
        issue_number = match.groupdict().get("number")
        month = match.groupdict().get("month")
        year = match.groupdict().get("year")
        if pattern is patterns[0] and issue_number:
            try:
                number_value = int(issue_number)
            except ValueError:
                continue
            if not 1 <= number_value <= 12:
                continue
        if issue_number and month is None:
            month = derive_month_from_issue(issue_number)
        return IssueMetadata(
            title=title,
            issue_number=issue_number,
            month=month,
            year=year,
        )

    return IssueMetadata(title=name, issue_number=None, month=None, year=None)


MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def derive_month_from_issue(issue_number: str | None) -> str | None:
    if not issue_number:
        return None
    try:
        index = int(issue_number)
    except ValueError:
        return None
    if 1 <= index <= 12:
        return MONTH_NAMES[index - 1]
    return None


@dataclasses.dataclass
class PageMetrics:
    path: Path
    aspect_ratio: float
    aspect_valid: bool
    interest_score: float
    colour_std: float
    saturation_mean: float
    edge_density: float
    white_ratio: float
    plain_ratio: float
    empty_tile_ratio: float
    margin_empty_ratio: float
    text_coverage: float
    has_swt_text: bool


def crop_margins(image: Image.Image, margin_percent: float) -> Image.Image:
    """Crop margins from an image to focus on content area."""
    width, height = image.size
    margin_x = int(width * margin_percent)
    margin_y = int(height * margin_percent)
    return image.crop((margin_x, margin_y, width - margin_x, height - margin_y))


def analyze_tile_content(tile: Image.Image) -> Tuple[float, float, float]:
    """
    Analyze a single tile to determine if it contains content.
    Returns: (std_dev, edge_density, dark_pixel_ratio)
    """
    grayscale = tile.convert("L")
    stat = ImageStat.Stat(grayscale)
    std_dev = stat.stddev[0]

    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_density = ImageStat.Stat(edges).mean[0] / 255

    histogram = grayscale.histogram()
    pixel_count = tile.size[0] * tile.size[1]
    dark_pixels = sum(histogram[:200])
    dark_ratio = dark_pixels / pixel_count if pixel_count > 0 else 0

    return std_dev, edge_density, dark_ratio


def is_tile_empty(std_dev: float, edge_density: float, dark_ratio: float) -> bool:
    """Determine if a tile is empty based on its metrics."""
    return std_dev < CONTENT_TILE_MIN_STD and edge_density < EMPTY_TILE_THRESHOLD and dark_ratio < 0.1


def analyze_page_tiles(image: Image.Image, grid_size: Tuple[int, int]) -> Tuple[float, float]:
    """
    Divide image into tiles and analyze content distribution.
    Returns: (empty_tile_ratio, margin_empty_ratio)
    """
    width, height = image.size
    rows, cols = grid_size
    tile_width = width // cols
    tile_height = height // rows

    empty_count = 0
    margin_empty_count = 0
    margin_tile_count = 0
    total_tiles = rows * cols

    for row in range(rows):
        for col in range(cols):
            is_margin = (row == 0 or row == rows - 1 or col == 0 or col == cols - 1)
            if is_margin:
                margin_tile_count += 1

            left = col * tile_width
            top = row * tile_height
            right = min(left + tile_width, width)
            bottom = min(top + tile_height, height)

            tile = image.crop((left, top, right, bottom))
            std_dev, edge_density, dark_ratio = analyze_tile_content(tile)

            if is_tile_empty(std_dev, edge_density, dark_ratio):
                empty_count += 1
                if is_margin:
                    margin_empty_count += 1

    empty_tile_ratio = empty_count / total_tiles if total_tiles > 0 else 0
    margin_empty_ratio = margin_empty_count / margin_tile_count if margin_tile_count > 0 else 0

    return empty_tile_ratio, margin_empty_ratio


def detect_text_with_swt(image: Image.Image) -> Tuple[float, bool]:
    """
    Detect text using Stroke Width Transform or fallback method.
    Returns: (text_coverage_ratio, has_text)
    """
    try:
        import cv2
        import numpy as np

        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        if hasattr(cv2, 'text') and hasattr(cv2.text, 'detectTextSWT'):
            swt_result = cv2.text.detectTextSWT(gray, dark_on_light=True)
            if swt_result is not None and len(swt_result) > 0:
                text_mask = swt_result[1] if len(swt_result) > 1 else swt_result[0]
                text_pixels = np.count_nonzero(text_mask)
                total_pixels = gray.size
                text_coverage = text_pixels / total_pixels if total_pixels > 0 else 0
                has_text = text_coverage > 0.01
                return text_coverage, has_text

        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_like_pixels = 0
        total_pixels = gray.size

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 0.1 < aspect_ratio < 10 and 10 < w < gray.shape[1] * 0.8 and 5 < h < gray.shape[0] * 0.3:
                text_like_pixels += w * h

        text_coverage = text_like_pixels / total_pixels if total_pixels > 0 else 0
        has_text = text_coverage > 0.05
        return text_coverage, has_text

    except ImportError:
        return 0.0, False


def evaluate_page_image(
    page_path: Path,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> PageMetrics:
    with Image.open(page_path) as img:
        width, height = img.size
        if height == 0:
            aspect_ratio = 0.0
        else:
            aspect_ratio = width / height
        aspect_valid = min_aspect_ratio <= aspect_ratio <= max_aspect_ratio

        rgb = img.convert("RGB")
        small = rgb.resize(EDGE_ENHANCE_SIZE, Image.BILINEAR)

        stat = ImageStat.Stat(small)
        colour_std = sum(stat.stddev) / (len(stat.stddev) * 255)

        grayscale = small.convert("L")
        histogram = grayscale.histogram()
        pixel_count = EDGE_ENHANCE_SIZE[0] * EDGE_ENHANCE_SIZE[1]
        white_pixels = sum(histogram[220:])
        white_ratio = white_pixels / pixel_count

        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        edge_density = ImageStat.Stat(edges).mean[0] / 255

        hsv = small.convert("HSV")
        h_band, s_band, v_band = hsv.split()
        saturation_mean = ImageStat.Stat(s_band).mean[0] / 255
        s_data = list(s_band.getdata())
        v_data = list(v_band.getdata())
        plain_pixels = sum(1 for s_val, v_val in zip(s_data, v_data) if s_val < 40 and v_val > 200)
        plain_ratio = plain_pixels / pixel_count

        interest_score = (
            colour_std * 0.4
            + (1 - white_ratio) * 0.25
            + edge_density * 0.2
            + saturation_mean * 0.15
        )

        cropped = crop_margins(rgb, MARGIN_CROP_PERCENT)
        empty_tile_ratio, margin_empty_ratio = analyze_page_tiles(cropped, TILE_GRID_SIZE)

        text_coverage, has_swt_text = detect_text_with_swt(rgb)

    return PageMetrics(
        path=page_path,
        aspect_ratio=aspect_ratio,
        aspect_valid=aspect_valid,
        interest_score=interest_score,
        colour_std=colour_std,
        saturation_mean=saturation_mean,
        edge_density=edge_density,
        white_ratio=white_ratio,
        plain_ratio=plain_ratio,
        empty_tile_ratio=empty_tile_ratio,
        margin_empty_ratio=margin_empty_ratio,
        text_coverage=text_coverage,
        has_swt_text=has_swt_text,
    )


def filter_pages(
    pages_root: Path,
    threshold: float,
    plain_threshold: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    delete: bool,
    verbose: bool,
) -> Tuple[int, int]:
    removed = 0
    kept = 0
    for issue_dir in sorted([p for p in pages_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        page_files: List[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            page_files.extend(sorted(issue_dir.glob(pattern), key=page_sort_key))
        for page_path in page_files:
            metrics = evaluate_page_image(page_path, min_aspect_ratio, max_aspect_ratio)
            remove_reason = None

            if not metrics.aspect_valid:
                remove_reason = f"aspect ratio {metrics.aspect_ratio:.2f} outside [{min_aspect_ratio}, {max_aspect_ratio}]"
            elif metrics.empty_tile_ratio >= plain_threshold:
                remove_reason = f"empty tile ratio {metrics.empty_tile_ratio:.2f} ≥ {plain_threshold}"
            elif metrics.plain_ratio >= plain_threshold:
                remove_reason = f"plain/text ratio {metrics.plain_ratio:.2f} ≥ {plain_threshold} (undecorated)"
            elif threshold > 0 and metrics.interest_score < threshold:
                remove_reason = f"interest score {metrics.interest_score:.3f} < {threshold}"

            if remove_reason:
                removed += 1
                if verbose:
                    print(f"Removing {page_path}: {remove_reason}")
                if delete:
                    try:
                        page_path.unlink()
                    except OSError as exc:
                        print(f"Failed to delete {page_path}: {exc}")
            else:
                kept += 1
                if verbose:
                    print(
                        f"Keeping {page_path}: empty_tiles {metrics.empty_tile_ratio:.2f}, "
                        f"text {metrics.text_coverage:.2f}, plain {metrics.plain_ratio:.2f}, "
                        f"interest {metrics.interest_score:.3f}, aspect {metrics.aspect_ratio:.2f}"
                    )
    return kept, removed


def ensure_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError as exc:
        raise SystemExit(f"Failed to load font at {font_path}: {exc}") from exc


def wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    raw_lines: List[str],
    max_width: int,
) -> List[WrappedLine]:
    wrapped: List[WrappedLine] = []
    for group_index, raw in enumerate(raw_lines):
        words = raw.split()
        if not words:
            wrapped.append(WrappedLine("", group_index))
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                wrapped.append(WrappedLine(current, group_index))
                current = word
        wrapped.append(WrappedLine(current, group_index))
    return wrapped


def create_canvas() -> Image.Image:
    return Image.new("RGB", CANVAS_SIZE, CANVAS_BACKGROUND)


def draw_panel_background(draw: ImageDraw.ImageDraw, panel_box: Tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(panel_box, radius=12, fill=INFO_PANEL_BACKGROUND)


def draw_image_with_border(canvas: Image.Image, image: Image.Image, position: Tuple[int, int]) -> None:
    x, y = position
    width, height = image.size
    border_box = (x - IMAGE_BORDER_WIDTH, y - IMAGE_BORDER_WIDTH, x + width + IMAGE_BORDER_WIDTH, y + height + IMAGE_BORDER_WIDTH)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(border_box, radius=12, fill=IMAGE_BORDER_COLOR)
    canvas.paste(image, (x, y))


def fit_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    scale = min(target_width / image.width, target_height / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def build_page_label(page_index: int, total_pages: int) -> str:
    digits = max(3, len(str(total_pages)))
    return f"Page {page_index:0{digits}d}/{total_pages:0{digits}d}"


def render_info_panel(
    canvas: Image.Image,
    metadata: IssueMetadata,
    page_label: str,
    font_path: Path,
) -> None:
    panel_left = CANVAS_SIZE[0] - PANEL_MARGIN - INFO_PANEL_WIDTH
    panel_box = (
        panel_left,
        PANEL_MARGIN,
        panel_left + INFO_PANEL_WIDTH,
        CANVAS_SIZE[1] - PANEL_MARGIN,
    )
    draw = ImageDraw.Draw(canvas)
    draw_panel_background(draw, panel_box)

    base_lines = metadata.display_lines + [page_label]
    max_width = INFO_PANEL_WIDTH - 2 * PANEL_MARGIN
    panel_height = panel_box[3] - panel_box[1]
    max_height = panel_height - 2 * PANEL_MARGIN
    base_target = min(max_height, MAX_FONT_SIZE)
    font_size = max(MIN_FONT_SIZE, base_target - FONT_SIZE_REDUCTION)

    wrapped_lines: List[WrappedLine] = []
    line_heights: List[int] = []
    total_height = 0
    max_line_width = 0

    while font_size >= MIN_FONT_SIZE:
        font = ensure_font(font_path, size=font_size)
        wrapped_lines = wrap_text_lines(draw, font, base_lines, max_width)
        line_heights = []
        total_height = 0
        max_line_width = 0
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line.text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            line_heights.append(height)
            total_height += height
            if width > max_line_width:
                max_line_width = width
        min_total_height = total_height + LINE_SPACING * max(0, len(wrapped_lines) - 1)
        if max_line_width <= max_width and min_total_height <= max_height:
            break
        font_size -= 2
    else:
        font_size = MIN_FONT_SIZE
        font = ensure_font(font_path, size=font_size)
        wrapped_lines = wrap_text_lines(draw, font, base_lines, max_width)
        line_heights = []
        total_height = 0
        max_line_width = 0
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line.text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            line_heights.append(height)
            total_height += height
            if width > max_line_width:
                max_line_width = width
        min_total_height = total_height + LINE_SPACING * max(0, len(wrapped_lines) - 1)

    available_vertical = max_height
    gaps = max(0, len(wrapped_lines) - 1)
    baseline_total = total_height + LINE_SPACING * gaps
    extra_space = max(0, available_vertical - baseline_total)
    separator_indices = [
        index
        for index in range(len(wrapped_lines) - 1)
        if wrapped_lines[index + 1].group_index != wrapped_lines[index].group_index
    ]
    separator_count = len(separator_indices)
    if separator_count > 0:
        extra_per_separator = extra_space // separator_count
        extra_remainder = extra_space % separator_count
    else:
        extra_per_separator = 0
        extra_remainder = 0

    text_x = panel_box[0] + PANEL_MARGIN
    current_y = panel_box[1] + PANEL_MARGIN
    last_index = len(wrapped_lines) - 1
    for index, line in enumerate(wrapped_lines):
        draw.text((text_x, current_y), line.text, font=font, fill=INFO_PANEL_TEXT_COLOR)
        line_bottom = current_y + line_heights[index]
        if index < last_index:
            next_line = wrapped_lines[index + 1]
            gap = LINE_SPACING
            if next_line.group_index != line.group_index:
                additional = extra_per_separator
                if extra_remainder > 0:
                    additional += 1
                    extra_remainder -= 1
                gap += additional
                separator_y = int(round(line_bottom + gap / 2))
                draw.line(
                    (
                        panel_box[0] + SEPARATOR_OFFSET,
                        separator_y,
                        panel_box[2] - SEPARATOR_OFFSET,
                        separator_y,
                    ),
                    fill=SEPARATOR_COLOR,
                    width=SEPARATOR_WIDTH,
                )
            current_y = line_bottom + gap
        else:
            current_y = line_bottom


def layout_single(page_path: Path, metadata: IssueMetadata, page_label: str, font_path: Path, output_path: Path) -> None:
    with Image.open(page_path) as original:
        original = original.convert("RGB")
        canvas = create_canvas()
        render_info_panel(canvas, metadata, page_label, font_path)

        panel_left = CANVAS_SIZE[0] - PANEL_MARGIN - INFO_PANEL_WIDTH
        available_width = panel_left - IMAGE_LEFT_OFFSET - IMAGE_PANEL_GAP
        available_height = CANVAS_SIZE[1] - IMAGE_TOP_MARGIN - IMAGE_BOTTOM_MARGIN
        fitted = fit_image(original, available_width, available_height)

        x = IMAGE_LEFT_OFFSET
        y = IMAGE_TOP_MARGIN
        draw_image_with_border(canvas, fitted, (x, y))
        canvas.save(output_path, format="JPEG", quality=95, optimize=True)


def generate_layouts(
    pages_root: Path,
    output_root: Path,
    font_path: Path,
    flatten: bool,
) -> List[Path]:
    created: List[Path] = []
    issue_dirs = sorted([p for p in pages_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    used_names: Set[str] = set()

    for issue_dir in issue_dirs:
        page_files: List[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            page_files.extend(sorted(issue_dir.glob(pattern), key=page_sort_key))
        if not page_files:
            continue
        total_pages = len(page_files)
        metadata = parse_issue_metadata(issue_dir)
        for index, page_path in enumerate(page_files, start=1):
            page_label = build_page_label(index, total_pages)
            random_name = generate_random_name(used_names)
            if flatten:
                dest_dir = output_root
            else:
                dest_dir = output_root / issue_dir.name / "single"
            dest = dest_dir / f"{random_name}.jpg"
            dest_dir.mkdir(parents=True, exist_ok=True)
            layout_single(page_path, metadata, page_label, font_path, dest)
            created.append(dest)
    return created


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split CBZ/CBR/PDF issues into page images and craft 1024x600 layouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Extract page images from CBZ/CBR/PDF sources.")
    split_parser.add_argument("sources", nargs="+", help="Files or directories containing CBZ/CBR/PDF issues.")
    split_parser.add_argument("--output-dir", default="split", help="Directory to store extracted pages.")
    split_parser.add_argument(
        "--split-spreads",
        action="store_true",
        default=DEFAULT_SPLIT_SPREADS,
        help="Automatically detect and split double-page spreads into two single pages.",
    )
    split_parser.add_argument(
        "--no-split-spreads",
        dest="split_spreads",
        action="store_false",
        help="Disable automatic splitting of double-page spreads.",
    )
    split_parser.add_argument(
        "--spread-threshold",
        type=float,
        default=DEFAULT_SPREAD_THRESHOLD,
        help="Aspect ratio threshold to detect double-page spreads (width/height). Pages with aspect ratio above this will be split.",
    )

    filter_parser = subparsers.add_parser("filter", help="Score page images and remove uninteresting ones.")
    filter_parser.add_argument("--input-dir", default="split", help="Directory containing extracted page images.")
    filter_parser.add_argument(
        "--plain-threshold",
        type=float,
        default=DEFAULT_PLAIN_THRESHOLD,
        help="Maximum allowed fraction of the page considered plain (white, undecorated space).",
    )
    filter_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_INTEREST_THRESHOLD,
        help="Minimum interest score required to keep a page.",
    )
    filter_parser.add_argument(
        "--min-aspect",
        type=float,
        default=DEFAULT_MIN_ASPECT_RATIO,
        help="Minimum allowed width/height ratio for a page (filters out tall skinny scans).",
    )
    filter_parser.add_argument(
        "--max-aspect",
        type=float,
        default=DEFAULT_MAX_ASPECT_RATIO,
        help="Maximum allowed width/height ratio for a page (filters out double-page spreads).",
    )
    filter_parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete pages that fail the filter. Otherwise only report counts.",
    )
    filter_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-page decisions while filtering.",
    )

    layout_parser = subparsers.add_parser("layout", help="Create 1024x600 single-page layouts from extracted images.")
    layout_parser.add_argument("--input-dir", default="split", help="Directory containing extracted page images.")
    layout_parser.add_argument("--output-dir", default="layouts", help="Directory to store generated layouts.")
    layout_parser.add_argument("--font", default="Fonts/lucasarts-scumm-solid.ttf", help="Path to the TrueType font file.")
    layout_parser.add_argument(
        "--flatten",
        action="store_true",
        help="Write all generated layouts into a single directory instead of per-issue folders.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "split":
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        source_files = list(find_publication_files(args.sources))
        if not source_files:
            parser.error("No CBZ/CBR/PDF files found in the provided sources.")
        for source_file in source_files:
            print(f"Extracting pages from {source_file}...")
            split_publication(source_file, output_root, args.split_spreads, args.spread_threshold)
        print(f"Pages written under {output_root}")

    elif args.command == "filter":
        pages_root = Path(args.input_dir)
        if not pages_root.exists():
            parser.error(f"Input directory {pages_root} does not exist.")
        kept, removed = filter_pages(
            pages_root,
            threshold=args.threshold,
            plain_threshold=args.plain_threshold,
            min_aspect_ratio=args.min_aspect,
            max_aspect_ratio=args.max_aspect,
            delete=args.delete,
            verbose=args.verbose,
        )
        action = "Deleted" if args.delete else "Flagged"
        print(f"{action} {removed} pages; kept {kept}.")
        if not args.delete:
            print("Re-run with --delete to remove the flagged pages from disk.")

    elif args.command == "layout":
        pages_root = Path(args.input_dir)
        output_root = Path(args.output_dir)
        font_path = Path(args.font)
        if not pages_root.exists():
            parser.error(f"Input directory {pages_root} does not exist.")
        if not font_path.exists():
            parser.error(f"Font file {font_path} does not exist.")
        created = generate_layouts(pages_root, output_root, font_path, args.flatten)
        if not created:
            print("No page images found. Run the split command first.")
        else:
            print(f"Created {len(created)} layouts under {output_root}")


if __name__ == "__main__":
    main()
