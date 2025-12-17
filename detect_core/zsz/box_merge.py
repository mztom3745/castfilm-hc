from typing import List
from detect_core.castfilm_detection.processing.defects import (
    BoundingBox,
)
def _boxes_overlap(a, b) -> bool:
    # 注意：bottom/right 在你这里是 max_row/max_col（像素索引）
    return not (a.right < b.left or b.right < a.left or a.bottom < b.top or b.bottom < a.top)

def merge_overlapping_boxes(boxes: List[BoundingBox]) -> List[BoundingBox]:
    if not boxes:
        return []

    merged: List[BoundingBox] = []

    for box in boxes:
        cur = box
        changed = True

        # 可能出现“链式重叠”（A overlap B, B overlap C），所以用循环把能并的都并完
        while changed:
            changed = False
            i = 0
            while i < len(merged):
                if _boxes_overlap(cur, merged[i]):
                    other = merged.pop(i)

                    # ✅ 合并：框取并集，pixels/dark_pixels 相加
                    top = min(cur.top, other.top)
                    left = min(cur.left, other.left)
                    bottom = max(cur.bottom, other.bottom)
                    right = max(cur.right, other.right)

                    pixels = cur.pixels + other.pixels
                    dark_pixels = getattr(cur, "dark_pixels", 0) + getattr(other, "dark_pixels", 0)
                    dark_ratio = (dark_pixels / pixels) if pixels > 0 else 0.0

                    cur = BoundingBox(
                        top=top, left=left, bottom=bottom, right=right,
                        pixels=pixels,
                        dark_pixels=dark_pixels,
                        dark_ratio=dark_ratio,
                    )
                    changed = True
                else:
                    i += 1

        merged.append(cur)

    return merged
