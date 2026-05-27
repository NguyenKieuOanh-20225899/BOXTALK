from __future__ import annotations

from typing import Callable, Iterable, TypeVar


T = TypeVar("T")
BBox = tuple[float, float, float, float]


def sort_in_reading_order(
    items: Iterable[T],
    *,
    bbox_getter: Callable[[T], BBox],
    page_width: float,
    page_height: float,
) -> list[T]:
    ordered = list(items)
    if len(ordered) < 4:
        return _sort_by_horizontal_bands(ordered, bbox_getter=bbox_getter)

    page_width = max(float(page_width), 1.0)
    page_height = max(float(page_height), 1.0)
    if not _looks_two_column(ordered, bbox_getter=bbox_getter, page_width=page_width):
        return _sort_by_horizontal_bands(ordered, bbox_getter=bbox_getter)

    full_width: list[T] = []
    column_items: list[T] = []
    for item in ordered:
        bbox = bbox_getter(item)
        width_ratio = max(0.0, bbox[2] - bbox[0]) / page_width
        spans_mid = bbox[0] < page_width * 0.38 and bbox[2] > page_width * 0.62
        if width_ratio >= 0.62 or spans_mid:
            full_width.append(item)
        else:
            column_items.append(item)

    full_width = sorted(full_width, key=lambda item: _simple_key(bbox_getter(item)))
    result: list[T] = []
    band_top = 0.0
    emitted_ids: set[int] = set()

    for separator in full_width:
        sep_box = bbox_getter(separator)
        band_items = [
            item
            for item in column_items
            if id(item) not in emitted_ids
            and bbox_getter(item)[1] >= band_top - page_height * 0.01
            and bbox_getter(item)[1] < sep_box[1] - page_height * 0.01
        ]
        result.extend(_sort_column_band(band_items, bbox_getter=bbox_getter, page_width=page_width))
        emitted_ids.update(id(item) for item in band_items)
        result.append(separator)
        emitted_ids.add(id(separator))
        band_top = max(band_top, sep_box[3])

    remaining = [
        item
        for item in column_items
        if id(item) not in emitted_ids
    ]
    result.extend(_sort_column_band(remaining, bbox_getter=bbox_getter, page_width=page_width))

    trailing_full_width = [
        item
        for item in full_width
        if id(item) not in emitted_ids
    ]
    result.extend(trailing_full_width)
    return result


def _looks_two_column(
    items: list[T],
    *,
    bbox_getter: Callable[[T], BBox],
    page_width: float,
) -> bool:
    narrow = []
    for item in items:
        bbox = bbox_getter(item)
        width_ratio = max(0.0, bbox[2] - bbox[0]) / page_width
        if width_ratio <= 0.55:
            narrow.append(bbox)
    if len(narrow) < 6:
        return False

    left = [bbox for bbox in narrow if _center_x(bbox) < page_width * 0.48]
    right = [bbox for bbox in narrow if _center_x(bbox) > page_width * 0.52]
    if len(left) < 2 or len(right) < 2:
        return False

    left_right_edge = median_value([bbox[2] for bbox in left])
    right_left_edge = median_value([bbox[0] for bbox in right])
    gutter = right_left_edge - left_right_edge
    return gutter >= page_width * 0.06


def _sort_column_band(
    items: list[T],
    *,
    bbox_getter: Callable[[T], BBox],
    page_width: float,
) -> list[T]:
    midpoint = page_width * 0.5
    return sorted(
        items,
        key=lambda item: (
            0 if _center_x(bbox_getter(item)) < midpoint else 1,
            bbox_getter(item)[1],
            bbox_getter(item)[0],
        ),
    )


def _sort_by_horizontal_bands(
    items: list[T],
    *,
    bbox_getter: Callable[[T], BBox],
) -> list[T]:
    if not items:
        return []

    ordered = sorted(items, key=lambda item: _simple_key(bbox_getter(item)))
    heights = [max(1.0, bbox_getter(item)[3] - bbox_getter(item)[1]) for item in ordered]
    y_tolerance = max(3.0, median_value(heights) * 0.45)

    bands: list[dict[str, object]] = []
    for item in ordered:
        bbox = bbox_getter(item)
        center_y = _center_y(bbox)
        matched_index = None
        for idx, band in enumerate(bands):
            band_center = float(band["center_y"])
            band_top = float(band["top"])
            band_bottom = float(band["bottom"])
            same_visual_line = abs(center_y - band_center) <= y_tolerance
            overlaps_band = min(bbox[3], band_bottom) - max(bbox[1], band_top) > 0
            if same_visual_line and overlaps_band:
                matched_index = idx
                break
        if matched_index is None:
            bands.append(
                {
                    "items": [item],
                    "center_y": center_y,
                    "top": bbox[1],
                    "bottom": bbox[3],
                }
            )
            continue
        band = bands[matched_index]
        band_items = band["items"]
        assert isinstance(band_items, list)
        band_items.append(item)
        band["center_y"] = sum(_center_y(bbox_getter(entry)) for entry in band_items) / len(band_items)
        band["top"] = min(float(band["top"]), bbox[1])
        band["bottom"] = max(float(band["bottom"]), bbox[3])

    result: list[T] = []
    for band in sorted(bands, key=lambda item: (float(item["top"]), float(item["center_y"]))):
        band_items = band["items"]
        assert isinstance(band_items, list)
        result.extend(sorted(band_items, key=lambda item: (bbox_getter(item)[0], bbox_getter(item)[1])))
    return result


def median_value(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _center_x(bbox: BBox) -> float:
    return (bbox[0] + bbox[2]) / 2.0


def _center_y(bbox: BBox) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _simple_key(bbox: BBox) -> tuple[float, float]:
    return (bbox[1], bbox[0])
