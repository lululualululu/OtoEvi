"""Existing Chinese report rules, used only as evaluation proxies."""

from __future__ import annotations

import re
import unicodedata

BINARY_RULES = {
    "perforation": {
        "terms": ("穿孔", "孔洞"),
        "negative": (
            "无(?:明显)?穿孔",
            "未见(?:明显)?穿孔",
            "没有(?:出现)?(?:明显)?穿孔",
            "没有孔洞",
        ),
    },
    "effusion": {
        "terms": ("积液", "液体积聚", "液平(?:征)?", "鼓室内?湿润"),
        "negative": (
            "无(?:明显)?积液",
            "未见(?:明显)?积液",
            "没有(?:发现)?(?:明显)?积液",
            "鼓室内?(?:也是)?正常",
        ),
    },
    "retraction": {
        "terms": ("凹陷",),
        "negative": ("无(?:明显)?凹陷", "未见(?:明显)?凹陷", "没有(?:明显)?凹陷"),
    },
    "secretion": {
        "terms": ("分泌物", "鼓室内?湿润"),
        "negative": (
            "未见(?:明显|异常)?分泌物",
            "无(?:明显|异常)?分泌物",
            "没有(?:粘附的|异常|明显)?分泌物",
        ),
    },
    "foreign_body": {
        "terms": ("异物",),
        "negative": (
            "未见(?:明显|异常)?异物",
            "未见(?:异常)?分泌物及(?:明显)?异物",
            "无(?:明显|异常)?异物",
            "没有发现任何异物",
            "没有(?:明显)?异物",
        ),
    },
    "hyperemia": {
        "terms": ("充血", "发红", "红肿"),
        "negative": ("皮肤(?:状况|状态|情况)?(?:无异常|正常|良好)",),
    },
    "bulging": {
        "terms": ("膨隆", "鼓膜鼓起", "鼓膜鼓起"),
        "negative": (
            "无(?:明显)?膨隆",
            "未见(?:明显)?膨隆",
            "鼓膜形态正常",
            "鼓膜形状正常",
        ),
    },
    "thinning": {
        "terms": ("菲薄", "厚度较薄", "变薄", "很薄"),
        "negative": ("无菲薄", "鼓膜形态正常", "鼓膜形状正常"),
    },
    "clouding": {
        "terms": ("鼓膜.{0,5}(?:浑浊|混浊)", "(?:浑浊|混浊)鼓膜"),
        "negative": ("鼓膜形态正常", "鼓膜形状正常"),
    },
    "cerumen": {
        "terms": ("耵聍", "耳垢"),
        "negative": ("未见(?:明显)?耵聍", "无(?:明显)?耵聍", "没有(?:明显)?耵聍"),
    },
}
CLINICAL_VARIANT_TRANSLATION = str.maketrans(
    {
        "見": "见",
        "異": "异",
        "無": "无",
        "內": "内",
        "側": "侧",
        "錐": "锥",
        "積": "积",
        "渾": "浑",
        "濁": "浊",
        "著": "着",
    }
)


def normalize_clinical_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(CLINICAL_VARIANT_TRANSLATION)


def binary_state(text: str, rule: dict) -> str:
    negative_matches = []
    cleaned = text
    for pattern in rule["negative"]:
        matches = list(re.finditer(pattern, cleaned))
        negative_matches.extend(matches)
        cleaned = re.sub(pattern, "", cleaned)
    positive = any((re.search(pattern, cleaned) for pattern in rule["terms"]))
    negative = bool(negative_matches)
    if positive and negative:
        return "conflict"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "unknown"


def laterality(text: str) -> str:
    left = bool(re.search("左(?:侧|耳|外耳道)", text))
    right = bool(re.search("右(?:侧|耳|外耳道)", text))
    if left and right:
        return "conflict"
    if left:
        return "left"
    if right:
        return "right"
    return "unknown"


def light_cone_state(text: str) -> str:
    if re.search("光锥.{0,5}(?:消失|不可见|无法看到)", text):
        return "absent"
    if re.search("光锥.{0,6}(?:模糊|边界不清|轮廓模糊)", text):
        return "blurred"
    if re.search("光锥.{0,6}(?:清晰|可见)", text):
        return "clear"
    return "unknown"


def extract_states(text: str) -> dict[str, str]:
    text = normalize_clinical_text(text)
    states = {name: binary_state(text, rule) for name, rule in BINARY_RULES.items()}
    states["laterality"] = laterality(text)
    states["light_cone"] = light_cone_state(text)
    return states
