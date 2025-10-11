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

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

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


def split_zip_archive(archive_path: Path, output_root: Path) -> List[Path]:
    import zipfile

    issue_name = sanitise_issue_name(archive_path.stem)
    issue_dir = output_root / issue_name
    issue_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths: List[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.namelist() if not m.endswith("/")]
        members.sort(key=numeric_key_from_string)
        for index, member in enumerate(members, start=1):
            with archive.open(member) as member_file:
                with Image.open(member_file) as img:
                    img = img.convert("RGB")
                    output_path = issue_dir / f"page{index:03d}.jpg"
                    img.save(output_path, format="JPEG", quality=95, optimize=True)
                    extracted_paths.append(output_path)
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


def split_rar_archive(archive_path: Path, output_root: Path) -> List[Path]:
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
                    output_path = issue_dir / f"page{page_index:03d}.jpg"
                    img.save(output_path, format="JPEG", quality=95, optimize=True)
                    extracted_paths.append(output_path)
                    page_index += 1
            except UnidentifiedImageError:
                continue

    if not extracted_paths:
        raise SystemExit(f"No images found within {archive_path}.")
    return extracted_paths


def split_pdf_document(pdf_path: Path, output_root: Path) -> List[Path]:
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
                output_path = issue_dir / f"page{page_index:03d}.jpg"
                img.save(output_path, format="JPEG", quality=95, optimize=True)
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


def split_publication(source_path: Path, output_root: Path) -> List[Path]:
    suffix = source_path.suffix.lower()
    if suffix in ZIP_EXTENSIONS:
        return split_zip_archive(source_path, output_root)
    if suffix in RAR_EXTENSIONS:
        return split_rar_archive(source_path, output_root)
    if suffix in PDF_EXTENSIONS:
        return split_pdf_document(source_path, output_root)
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
            split_publication(source_file, output_root)
        print(f"Pages written under {output_root}")

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
