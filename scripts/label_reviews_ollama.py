#!/usr/bin/env python3
"""Label Wongnai reviews with a local Ollama model.

This script is intended for pilot labeling before building retrieval indexes.
It reads the Wongnai review CSV, keeps the gold `rating` from the raw file,
asks Ollama to extract restaurant facets, normalizes the response to a fixed
Thai taxonomy, and writes JSONL output plus a summary file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("wongnai-review-dataset/review_dataset/w_review_train.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/ollama_labels")

FOOD_NATIONALITY = [
    "อาหารไทย",
    "อาหารจีน",
    "อาหารญี่ปุ่น",
    "อาหารอินเดีย",
    "อาหารอิตาลี",
    "อาหารฟิวชั่น",
    "อาหารเกาหลี",
    "อาหารเวียดนาม",
    "อาหารอเมริกัน",
]

FOOD_TYPE = [
    "อาหารทะเล",
    "พิซซ่า",
    "เบเกอรี่",
    "ขนมและเครื่องดื่ม",
    "ไอศกรีม",
    "ข้าวแกง",
    "ก๋วยเตี๋ยว",
    "ตามสั่ง",
    "สุขภาพ",
]

AMBIENCE = [
    "หรูหรา",
    "ติดแอร์",
    "ร้านเปิด",
    "ร้านข้างทาง",
    "บรรยากาศสงบ",
    "ริมน้ำ",
    "คาเฟ่",
]

PRICE_LEVEL = ["ราคาแพง", "ราคาย่อมเยา"]
VENUE_TYPE = ["ร้านอาหาร", "คาเฟ่", "unknown"]

ENGLISH_TO_THAI = {
    "thai": "อาหารไทย",
    "chinese": "อาหารจีน",
    "japanese": "อาหารญี่ปุ่น",
    "indian": "อาหารอินเดีย",
    "italian": "อาหารอิตาลี",
    "fusion": "อาหารฟิวชั่น",
    "korean": "อาหารเกาหลี",
    "vietnamese": "อาหารเวียดนาม",
    "american": "อาหารอเมริกัน",
    "seafood": "อาหารทะเล",
    "pizza": "พิซซ่า",
    "bakery": "เบเกอรี่",
    "dessert and drinks": "ขนมและเครื่องดื่ม",
    "desserts and drinks": "ขนมและเครื่องดื่ม",
    "dessert": "ขนมและเครื่องดื่ม",
    "drinks": "ขนมและเครื่องดื่ม",
    "ice cream": "ไอศกรีม",
    "curry rice": "ข้าวแกง",
    "noodles": "ก๋วยเตี๋ยว",
    "made to order": "ตามสั่ง",
    "healthy": "สุขภาพ",
    "luxury": "หรูหรา",
    "air conditioned": "ติดแอร์",
    "open air": "ร้านเปิด",
    "street food": "ร้านข้างทาง",
    "quiet": "บรรยากาศสงบ",
    "riverside": "ริมน้ำ",
    "cafe": "คาเฟ่",
    "restaurant": "ร้านอาหาร",
    "expensive": "ราคาแพง",
    "affordable": "ราคาย่อมเยา",
    "budget": "ราคาย่อมเยา",
    "low": "ราคาย่อมเยา",
    "mid-range": None,
}

FOOD_NATIONALITY_KEYWORDS = {
    "อาหารไทย": ["อาหารไทย", "ขนมไทย", "ลาบ", "พะแนง", "แกง", "ส้มตำ", "ผัดไทย", "กะเพรา", "ข้าวแกง", "น้ำพริก", "ต้มยำ", "แกงส้ม", "ยำ", "ข้าวมันไก่", "ขนมจีน", "หมูสะเต๊ะ"],
    "อาหารจีน": ["อาหารจีน", "จีนแคะ", "ติ่มซำ", "ซาลาเปา", "เกี๊ยว", "บะหมี่", "ข้าวผัด", "ภัตตาคารจีน", "ฮะเก๋า", "เคาหยก", "หม้อไฟ", "ขาหมูหมั่นโถว", "ปาท่องโก๋"],
    "อาหารญี่ปุ่น": ["อาหารญี่ปุ่น", "ซูชิ", "ซาชิมิ", "ราเมง", "ด้ง", "เทมปุระ", "อุด้ง", "โซบะ", "ยากิโซบะ", "ทงคัตสึ", "ข้าวหน้า", "โอซาก้า", "ฮอกไกโด"],
    "อาหารอินเดีย": ["อาหารอินเดีย", "แกงกะหรี่", "นาน", "บัตเตอร์ชิกเก้น", "biryani", "biryani", "tandoori", "มัสมั่น", "โรตี"],
    "อาหารอิตาลี": ["อาหารอิตาลี", "พิซซ่า", "สปาเก็ตตี้", "พาสต้า", "ลาซานญ่า", "carbonara", "คาโบนารา", "risotto", "ริซอตโต้", "parma ham", "truffle"],
    "อาหารฟิวชั่น": ["ฟิวชั่น", "fusion", "ประยุกต์", "creative", "modern", "ร่วมสมัย"],
    "อาหารเกาหลี": ["อาหารเกาหลี", "กิมจิ", "บิบิมบับ", "ต๊อกบกกี", "ปิ้งย่างเกาหลี", "คิมบับ", "ซุปกิมจิ", "bulgogi", "บูลโกกิ"],
    "อาหารเวียดนาม": ["อาหารเวียดนาม", "เฝอ", "แหนมเนือง", "ปอเปี๊ยะสด", "บั๋นหมี่", "banh mi", "bun cha", "ก๋วยจั๊บญวน"],
    "อาหารอเมริกัน": ["อาหารอเมริกัน", "เบอร์เกอร์", "สเต๊ก", "hotdog", "บาร์บีคิว", "bbq", "buffalo wing", "เฟรนช์ฟรายส์", "แพนเค้ก"],
}

FOOD_TYPE_KEYWORDS = {
    "อาหารทะเล": ["ซีฟู้ด", "อาหารทะเล", "กุ้ง", "ปู", "ปลา", "หอย", "ปลาหมึก", "กุ้งมังกร", "ปลาเก๋า", "ปูม้า", "กุ้งแม่น้ำ", "ทะเลเผา"],
    "พิซซ่า": ["พิซซ่า", "pizza", "พิซซ่าพารม่า", "pizza parma ham"],
    "เบเกอรี่": ["เบเกอรี่", "เค้ก", "คัพเค้ก", "ครัวซองต์", "ขนมปัง", "บราวนี่", "พาย", "แพนเค้ก", "waffle", "วาฟเฟิล", "มาการอง", "คัสตาร์ด"],
    "ขนมและเครื่องดื่ม": ["ชา", "กาแฟ", "โกโก้", "น้ำผึ้ง", "เครื่องดื่ม", "ของหวาน", "ขนมหวาน", "cupcake", "lemonade", "smoothie", "น้ำเต้าหู้", "ชานม", "ลาเต้", "มอคค่า", "เฟรปปูชิโน่", "ปั่น", "สมูทตี้", "ชานมไข่มุก"],
    "ไอศกรีม": ["ไอศกรีม", "ไอติม", "gelato", "sundae", "ซันเดย์", "ice cream"],
    "ข้าวแกง": ["ข้าวแกง", "แกง", "กับข้าว", "ข้าวราดแกง", "แกงกะหรี่", "ข้าวหน้าแกง"],
    "ก๋วยเตี๋ยว": ["ก๋วยเตี๋ยว", "บะหมี่", "ราเมง", "เย็นตาโฟ", "เกาเหลา", "ราดหน้า", "เส้นหมี่", "เส้นเล็ก", "ก๋วยจั๊บ", "หมี่", "โจ๊ก", "ต้มเลือดหมู"],
    "ตามสั่ง": ["ตามสั่ง", "จานเดียว", "ข้าวผัด", "กะเพรา", "ผัดซีอิ๊ว", "ข้าวหน้าไก่", "ผัดกระเพรา", "ผัดไทย", "ข้าวยำ"],
    "สุขภาพ": ["สุขภาพ", "คลีน", "สลัด", "ไรซ์เบอร์รี่", "low fat", "ออร์แกนิก", "อโวคาโด", "avocado", "vegan", "มังสวิรัติ"],
}

AMBIENCE_KEYWORDS = {
    "หรูหรา": ["หรู", "หรูหรา", "พรีเมียม", "ภัตตาคาร", "fine dining", "elegant", "วิวดี", "ตกแต่งสวย"],
    "ติดแอร์": ["ติดแอร์", "แอร์", "ห้องแอร์", "ในห้าง", "digital gateway", "ในห้างสรรพสินค้า", "ห้าง"],
    "ร้านเปิด": ["open air", "เปิดโล่ง", "ลมโกรก", "outdoor", "ริมทาง", "ข้างนอก", "โอเพ่นแอร์"],
    "ร้านข้างทาง": ["ข้างทาง", "รถเข็น", "ริมถนน", "เพิง", "สะพานเหลือง", "ตลาด", "สตรีทฟู้ด", "ฟุตบาท"],
    "บรรยากาศสงบ": ["สงบ", "เงียบ", "ชิล", "ผ่อนคลาย", "สบายๆ", "บรรยากาศดี", "นั่งชิล"],
    "ริมน้ำ": ["ริมน้ำ", "ติดทะเล", "ชายหาด", "ริมทะเล", "วิวทะเล", "ริมคลอง", "ริมแม่น้ำ"],
    "คาเฟ่": ["คาเฟ่", "cafe", "coffee", "cupcake", "กาแฟ", "coffee shop", "ร้านน่ารัก", "ร้านสีชมพู"],
}

PRICE_KEYWORDS = {
    "ราคาแพง": ["แพง", "แพงเวอร์", "ราคาแรง", "เกินไป", "รับไม่ไหว", "สูงไป"],
    "ราคาย่อมเยา": ["ราคาย่อมเยา", "ไม่แพง", "ถูก", "คุ้ม", "คุ้มค่า", "30 บาท", "59 บาท"],
}

VENUE_TYPE_KEYWORDS = {
    "คาเฟ่": ["คาเฟ่", "cafe", "coffee", "กาแฟ", "cupcake", "เบเกอรี่", "coffee shop", "ร้านกาแฟ"],
    "ร้านอาหาร": ["ร้านอาหาร", "ภัตตาคาร", "อาหาร", "เมนู", "กับข้าว", "ร้านข้าว", "บุฟเฟ่ต์", "ซีฟู้ด", "ก๋วยเตี๋ยว"],
}

GENERIC_RESTAURANT_NAMES = {
    "ร้านนี้",
    "ที่นี่",
    "ร้านอาหาร",
    "คาเฟ่",
    "ปอเปี๊ยะสด",
    "ก๋วยเตี๋ยว",
    "ข้าวแกง",
    "ร้านกาแฟ",
}

GENERIC_LOCATION_TERMS = {
    "ทางซ้ายมือ",
    "ในเมืองไทย",
    "ร้านนี้",
    "ที่นี่",
    "ห้องน้ำ",
    "ชั้น 1",
    "ชั้น 2",
    "ชั้น 3",
    "ชั้นสาม",
    "ชั้นสอง",
    "ชั้นบน",
    "ข้างใน",
    "ข้างนอก",
    "ถึงโต๊ะ",
    "โต๊ะ",
    "หน้าร้าน",
}

NAME_STOPWORDS = {
    "เมนู",
    "อาหาร",
    "อร่อย",
    "ชอบ",
    "เดินทาง",
    "หาง่าย",
    "รสชาติ",
    "ราคา",
    "ร้านอยู่",
}

LOCATION_STOP_MARKERS = [" เมื่อ", " ร้าน", " อาหาร", " มีคน", " เพราะ", " เลย", " ค่ะ", " ครับ", "\n", "("]
LOW_SIGNAL_KEYWORDS = {
    "อาหารไทย": {"แกง"},
    "ขนมและเครื่องดื่ม": {"ชา"},
    "ร้านอาหาร": {"อาหาร"},
    "หรูหรา": {"วิวดี"},
}

PROMPT_TEMPLATE = """สกัดข้อมูลจากรีวิวร้านอาหารต่อไปนี้ และตอบเป็น JSON เท่านั้น

กติกาเข้มงวด:
- ใช้เฉพาะข้อมูลที่มีหลักฐานตรงในข้อความรีวิว
- ถ้าไม่พบหลักฐานชัดเจน ให้ตอบ null หรือ []
- ห้ามเดาชื่อร้าน
- ค่าหมวดหมู่ต้องเป็นภาษาไทยเท่านั้น
- venue_type เลือกได้เฉพาะ: ร้านอาหาร, คาเฟ่, unknown
- food_nationality เลือกได้จาก: {food_nationality}
- food_type เลือกได้จาก: {food_type}
- ambience เลือกได้จาก: {ambience}
- price_level เลือกได้เฉพาะ: ราคาแพง, ราคาย่อมเยา, null
- location_mentions ให้ดึงเฉพาะสถานที่ที่มีในข้อความเป็นคำสั้น ๆ
- restaurant_name ให้ใส่เฉพาะเมื่อข้อความระบุชัดว่าเป็นชื่อร้าน

JSON schema:
{{
  "restaurant_name": "string|null",
  "venue_type": "ร้านอาหาร|คาเฟ่|unknown|null",
  "food_nationality": ["string"],
  "food_type": ["string"],
  "ambience": ["string"],
  "price_level": "string|null",
  "location_mentions": ["string"],
  "evidence": {{
    "restaurant_name": "string|null",
    "food_nationality": ["string"],
    "food_type": ["string"],
    "ambience": ["string"],
    "price_level": ["string"],
    "location_mentions": ["string"]
  }}
}}

รีวิว:
{review}
"""


@dataclass
class ReviewRow:
    row_id: int
    review_text: str
    rating: int


@dataclass
class RuleExtraction:
    restaurant_name: str | None
    venue_type: str
    food_nationality: list[str]
    food_type: list[str]
    ambience: list[str]
    price_level: str | None
    location_mentions: list[str]
    evidence: dict[str, list[str] | str | None]
    confidence: dict[str, float]


def env_or_default(name: str, default: Any, cast: Any | None = None) -> Any:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if cast is None:
        return value
    return cast(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot label Wongnai reviews via Ollama.")
    parser.add_argument("--input", type=Path, default=Path(env_or_default("LABEL_INPUT", str(DEFAULT_INPUT))), help="Input CSV path.")
    parser.add_argument("--output-dir", type=Path, default=Path(env_or_default("LABEL_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))), help="Directory for JSONL outputs.")
    parser.add_argument("--model", default=env_or_default("OLLAMA_MODEL", "scb10x/typhoon2.5-qwen3-4b"), help="Ollama model name.")
    parser.add_argument("--sample-size", type=int, default=env_or_default("LABEL_SAMPLE_SIZE", 20, int), help="Number of rows to label.")
    parser.add_argument("--seed", type=int, default=env_or_default("LABEL_SEED", 42, int), help="Random seed for sampling.")
    parser.add_argument("--parallel", type=int, default=env_or_default("LABEL_PARALLEL", 2, int), help="Parallel request count.")
    parser.add_argument("--temperature", type=float, default=env_or_default("OLLAMA_TEMPERATURE", 0.0, float), help="Generation temperature.")
    parser.add_argument("--num-ctx", type=int, default=env_or_default("OLLAMA_NUM_CTX", 2048, int), help="Context window per request.")
    parser.add_argument("--timeout", type=int, default=env_or_default("OLLAMA_TIMEOUT", 600, int), help="Per-request timeout in seconds.")
    parser.add_argument("--keep-alive", default=env_or_default("OLLAMA_KEEP_ALIVE", "30m"), help="Ollama keep_alive duration, for example 30m or 1h.")
    parser.add_argument("--limit-rows", type=int, default=env_or_default("LABEL_LIMIT_ROWS", None, int) if os.getenv("LABEL_LIMIT_ROWS") else None, help="Optional hard cap before sampling.")
    parser.add_argument("--start-row", type=int, default=env_or_default("LABEL_START_ROW", 0, int), help="Optional start row before sampling.")
    parser.add_argument("--base-url", default=env_or_default("OLLAMA_BASE_URL", "http://localhost:11434/api/chat"), help="Ollama chat endpoint.")
    parser.add_argument("--resume-dir", type=Path, default=None, help="Resume an existing run directory.")
    parser.add_argument("--retries", type=int, default=env_or_default("LABEL_RETRIES", 2, int), help="Retries per row on transient or parse failures.")
    parser.add_argument("--heartbeat-sec", type=int, default=env_or_default("LABEL_HEARTBEAT_SEC", 30, int), help="Heartbeat interval in seconds.")
    parser.add_argument("--max-review-chars", type=int, default=env_or_default("LABEL_MAX_REVIEW_CHARS", 1200, int), help="Truncate review text before prompting; 0 disables truncation.")
    parser.add_argument("--fast-mode", action="store_true", help="Skip heavy debug fields such as raw model content.")
    return parser.parse_args()


def load_reviews(path: Path, start_row: int = 0, limit_rows: int | None = None) -> list[ReviewRow]:
    rows: list[ReviewRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row_id, parts in enumerate(reader):
            if row_id < start_row:
                continue
            if not parts:
                continue
            if len(parts) < 2:
                continue
            review_text = ";".join(parts[:-1]).strip()
            rating_raw = parts[-1].strip()
            if not review_text:
                continue
            try:
                rating = int(rating_raw)
            except ValueError:
                continue
            rows.append(ReviewRow(row_id=row_id, review_text=review_text, rating=rating))
            if limit_rows is not None and len(rows) >= limit_rows:
                break
    return rows


def build_prompt(review_text: str, rule_labels: RuleExtraction) -> str:
    rule_summary = {
        "restaurant_name": rule_labels.restaurant_name,
        "venue_type": rule_labels.venue_type,
        "food_nationality": rule_labels.food_nationality,
        "food_type": rule_labels.food_type,
        "ambience": rule_labels.ambience,
        "price_level": rule_labels.price_level,
        "location_mentions": rule_labels.location_mentions,
    }
    return PROMPT_TEMPLATE.format(
        food_nationality=", ".join(FOOD_NATIONALITY),
        food_type=", ".join(FOOD_TYPE),
        ambience=", ".join(AMBIENCE),
        review=(
            "ผลจาก rule-based extractor (อาจไม่ครบและอาจผิดบางส่วน):\n"
            f"{json.dumps(rule_summary, ensure_ascii=False)}\n\n"
            "ให้ตรวจทานกับรีวิวจริง แล้วตอบ JSON ตาม schema โดยแก้เฉพาะจากหลักฐานในรีวิว:\n"
            f"{review_text}"
        ),
    )


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def clean_text_for_output(value: Any) -> str:
    text = normalize_text(value)
    return text or ""


def join_list(values: list[str]) -> str:
    return " | ".join(values)


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def truncate_review_text(review_text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(review_text) <= max_chars:
        return review_text
    truncated = review_text[:max_chars].rstrip()
    return truncated + "\n\n[TRUNCATED]"


def normalize_list(values: Any, allowed: list[str] | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        lowered = text.lower()
        mapped = ENGLISH_TO_THAI.get(lowered, text)
        if mapped is None:
            continue
        if allowed is not None and mapped not in allowed:
            continue
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def matched_keywords(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    matches = [keyword for keyword in keywords if keyword.lower() in lowered]
    return unique_keep_order(matches)


def detect_supported_categories(review_text: str, mapping: dict[str, list[str]]) -> set[str]:
    supported: set[str] = set()
    for category, keywords in mapping.items():
        if contains_any_keyword(review_text, keywords):
            supported.add(category)
    return supported


def detect_categories_with_evidence(review_text: str, mapping: dict[str, list[str]]) -> tuple[list[str], dict[str, list[str]]]:
    categories: list[str] = []
    evidence: dict[str, list[str]] = {}
    for category, keywords in mapping.items():
        matches = matched_keywords(review_text, keywords)
        low_signal = LOW_SIGNAL_KEYWORDS.get(category, set())
        matches = [match for match in matches if match not in low_signal or len(matches) > 1]
        if matches:
            categories.append(category)
            evidence[category] = matches
    return categories, evidence


def filter_predicted_categories(predicted: list[str], supported: set[str]) -> list[str]:
    if not predicted:
        return sorted(supported)
    filtered = [item for item in predicted if item in supported]
    if filtered:
        return filtered
    return []


def normalize_price(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    mapped = ENGLISH_TO_THAI.get(text.lower(), text)
    if mapped in PRICE_LEVEL:
        return mapped
    return None


def normalize_venue_type(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return "unknown"
    mapped = ENGLISH_TO_THAI.get(text.lower(), text)
    if mapped in VENUE_TYPE:
        return mapped
    return "unknown"


def normalize_restaurant_name(value: Any, review_text: str) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    if text in GENERIC_RESTAURANT_NAMES:
        return None
    if text not in review_text:
        return None
    return text


def normalize_location(values: Any) -> list[str]:
    items = normalize_list(values, allowed=None)
    cleaned: list[str] = []
    for item in items:
        for marker in LOCATION_STOP_MARKERS:
            if marker in item:
                item = item.split(marker, 1)[0].strip()
        if item in GENERIC_LOCATION_TERMS:
            continue
        if re.fullmatch(r"ชั้น\s*\d+", item):
            continue
        if re.fullmatch(r"\d+\s*บาท", item):
            continue
        if len(item) < 3:
            continue
        if item not in cleaned:
            cleaned.append(item)
    return cleaned


def sanitize_name_candidate(value: str) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    text = re.split(r"[\n,.()]", text, maxsplit=1)[0].strip()
    if len(text) > 40:
        return None
    tokens = text.split()
    if len(tokens) > 4:
        return None
    if any(stopword in text for stopword in NAME_STOPWORDS):
        return None
    return text


def extract_price_from_numeric(review_text: str) -> tuple[str | None, list[str]]:
    matches = re.findall(r"(\d{2,4})\s*บาท", review_text)
    if not matches:
        return None, []
    prices = [int(match) for match in matches]
    evidence = [f"{price} บาท" for price in prices]
    if min(prices) <= 80 and max(prices) <= 120:
        return "ราคาย่อมเยา", evidence
    if max(prices) >= 250:
        return "ราคาแพง", evidence
    return None, evidence


def extract_location_candidates(review_text: str) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"(เชียงใหม่|พัทยา|ภูเก็ต|หัวหิน|ระยอง|สงขลา|นครปฐม|อยุธยา|ศรีราชา|เขาใหญ่|กระบี่|สุโขทัย|ชลบุรี|บางนา|ราชเทวี|ประชาชื่น|ลาดพร้าว|พระราม3|อ่อนนุช|สยาม|สะพานเหลือง)",
        r"(BTS\s*[A-Za-zก-๙0-9]+|BTS[ก-๙A-Za-z0-9]+)",
        r"(Digital gateway|digital gateway|Central [A-Za-zก-๙]+|เดอะมอลล์[ก-๙A-Za-z0-9]*)",
        r"(ถนน[ก-๙A-Za-z0-9 .-]+|ถ\.[ก-๙A-Za-z0-9 .-]+)",
        r"(พุทธมณฑลสาย\s*\d+)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, review_text, flags=re.IGNORECASE):
            text = normalize_text(match)
            if text:
                candidates.append(text)
    return normalize_location(unique_keep_order(candidates))


def extract_restaurant_name_from_text(review_text: str) -> tuple[str | None, list[str]]:
    patterns = [
        r"ร้าน\s+([A-Za-z0-9][A-Za-z0-9 '&._-]{2,40})",
        r"ร้าน\s+([ก-๙A-Za-z0-9][ก-๙A-Za-z0-9 '&._-]{2,40})",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, review_text):
            candidate = sanitize_name_candidate(match)
            if not candidate:
                continue
            if candidate in GENERIC_RESTAURANT_NAMES:
                continue
            return candidate, [candidate]
    return None, []


def extract_rule_labels(row: ReviewRow) -> RuleExtraction:
    review_text = row.review_text
    restaurant_name, restaurant_evidence = extract_restaurant_name_from_text(review_text)

    food_nationality, food_nationality_evidence = detect_categories_with_evidence(review_text, FOOD_NATIONALITY_KEYWORDS)
    food_type, food_type_evidence = detect_categories_with_evidence(review_text, FOOD_TYPE_KEYWORDS)
    ambience, ambience_evidence = detect_categories_with_evidence(review_text, AMBIENCE_KEYWORDS)
    venue_type_matches, venue_type_evidence = detect_categories_with_evidence(review_text, VENUE_TYPE_KEYWORDS)
    price_matches, price_evidence_keywords = detect_categories_with_evidence(review_text, PRICE_KEYWORDS)
    numeric_price, numeric_price_evidence = extract_price_from_numeric(review_text)
    location_mentions = extract_location_candidates(review_text)

    venue_type = "unknown"
    if "คาเฟ่" in venue_type_matches and "ร้านอาหาร" not in venue_type_matches:
        venue_type = "คาเฟ่"
    elif "ร้านอาหาร" in venue_type_matches:
        venue_type = "ร้านอาหาร"

    price_level = None
    if "ราคาแพง" in price_matches and "ราคาย่อมเยา" not in price_matches:
        price_level = "ราคาแพง"
    elif "ราคาย่อมเยา" in price_matches and "ราคาแพง" not in price_matches:
        price_level = "ราคาย่อมเยา"
    elif numeric_price:
        price_level = numeric_price

    evidence = {
        "restaurant_name": restaurant_evidence[0] if restaurant_evidence else None,
        "food_nationality": [f"{label}: {', '.join(food_nationality_evidence[label])}" for label in food_nationality],
        "food_type": [f"{label}: {', '.join(food_type_evidence[label])}" for label in food_type],
        "ambience": [f"{label}: {', '.join(ambience_evidence[label])}" for label in ambience],
        "price_level": [f"{label}: {', '.join(price_evidence_keywords[label])}" for label in price_matches] + numeric_price_evidence,
        "location_mentions": location_mentions,
    }

    confidence = {
        "restaurant_name": 0.95 if restaurant_name else 0.0,
        "venue_type": 0.9 if venue_type != "unknown" else 0.0,
        "food_nationality": min(1.0, 0.55 + 0.15 * len(food_nationality)) if food_nationality else 0.0,
        "food_type": min(1.0, 0.5 + 0.1 * len(food_type)) if food_type else 0.0,
        "ambience": min(1.0, 0.55 + 0.15 * len(ambience)) if ambience else 0.0,
        "price_level": 0.9 if price_level else 0.0,
        "location_mentions": min(1.0, 0.5 + 0.1 * len(location_mentions)) if location_mentions else 0.0,
    }

    return RuleExtraction(
        restaurant_name=restaurant_name,
        venue_type=venue_type,
        food_nationality=food_nationality,
        food_type=food_type,
        ambience=ambience,
        price_level=price_level,
        location_mentions=location_mentions,
        evidence=evidence,
        confidence=confidence,
    )


def infer_venue_type(review_text: str, predicted: str) -> str:
    supported = detect_supported_categories(review_text, VENUE_TYPE_KEYWORDS)
    if predicted in supported:
        return predicted
    if "คาเฟ่" in supported and "ร้านอาหาร" not in supported:
        return "คาเฟ่"
    if "ร้านอาหาร" in supported:
        return "ร้านอาหาร"
    return predicted if predicted in VENUE_TYPE else "unknown"


def infer_price_level(review_text: str, predicted: str | None) -> str | None:
    supported = detect_supported_categories(review_text, PRICE_KEYWORDS)
    if predicted in supported:
        return predicted
    if predicted and predicted not in supported:
        return None
    if "ราคาแพง" in supported and "ราคาย่อมเยา" not in supported:
        return "ราคาแพง"
    if "ราคาย่อมเยา" in supported and "ราคาแพง" not in supported:
        return "ราคาย่อมเยา"
    return None


def choose_scalar(rule_value: str | None, llm_value: str | None, rule_conf: float, default: str | None = None) -> tuple[str | None, str]:
    if rule_value and rule_conf >= 0.8:
        return rule_value, "rule"
    if llm_value:
        return llm_value, "llm"
    if rule_value:
        return rule_value, "rule"
    return default, "default"


def choose_list(rule_values: list[str], llm_values: list[str], rule_conf: float) -> tuple[list[str], str]:
    if rule_values and rule_conf >= 0.75:
        if llm_values:
            return unique_keep_order(rule_values + llm_values), "rule+llm"
        return rule_values, "rule"
    if llm_values:
        if rule_values:
            return unique_keep_order(rule_values + llm_values), "rule+llm"
        return llm_values, "llm"
    if rule_values:
        return rule_values, "rule"
    return [], "default"


def parse_model_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def ns_to_sec(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1_000_000_000, 6)
    except (TypeError, ValueError):
        return None


def safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator in (None, 0) or denominator in (None, 0):
        return None
    try:
        return round(float(numerator) / float(denominator), 6)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_runtime_metrics(raw_response: dict[str, Any], wall_latency_sec: float) -> dict[str, Any]:
    prompt_tokens = raw_response.get("prompt_eval_count")
    completion_tokens = raw_response.get("eval_count")
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    total_duration_sec = ns_to_sec(raw_response.get("total_duration"))
    load_duration_sec = ns_to_sec(raw_response.get("load_duration"))
    prompt_eval_duration_sec = ns_to_sec(raw_response.get("prompt_eval_duration"))
    eval_duration_sec = ns_to_sec(raw_response.get("eval_duration"))

    return {
        "timing": {
            "wall_latency_sec": round(wall_latency_sec, 3),
            "total_duration_sec": total_duration_sec,
            "load_duration_sec": load_duration_sec,
            "prompt_eval_duration_sec": prompt_eval_duration_sec,
            "eval_duration_sec": eval_duration_sec,
        },
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens if total_tokens else None,
        },
        "throughput": {
            "completion_tokens_per_sec": safe_div(completion_tokens, eval_duration_sec),
            "total_tokens_per_sec": safe_div(total_tokens, total_duration_sec),
            "requests_per_sec_wall": safe_div(1.0, wall_latency_sec),
        },
    }


def build_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
        },
    }


def call_ollama(args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], dict[str, Any], float]:
    payload = build_payload(args, prompt)
    request = urllib.request.Request(
        args.base_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    elapsed = time.perf_counter() - start
    outer = json.loads(raw)
    model_content = outer["message"]["content"]
    parsed = parse_model_json(model_content)
    return parsed, outer, elapsed


def post_process(parsed: dict[str, Any], row: ReviewRow, rule_labels: RuleExtraction) -> dict[str, Any]:
    evidence = parsed.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    food_nationality_supported = detect_supported_categories(row.review_text, FOOD_NATIONALITY_KEYWORDS)
    food_type_supported = detect_supported_categories(row.review_text, FOOD_TYPE_KEYWORDS)
    ambience_supported = detect_supported_categories(row.review_text, AMBIENCE_KEYWORDS)

    llm_food_nationality = filter_predicted_categories(
        normalize_list(parsed.get("food_nationality"), allowed=FOOD_NATIONALITY),
        food_nationality_supported,
    )
    llm_food_type = filter_predicted_categories(
        normalize_list(parsed.get("food_type"), allowed=FOOD_TYPE),
        food_type_supported,
    )
    llm_ambience = filter_predicted_categories(
        normalize_list(parsed.get("ambience"), allowed=AMBIENCE),
        ambience_supported,
    )
    llm_price_level = infer_price_level(row.review_text, normalize_price(parsed.get("price_level")))
    llm_venue_type = infer_venue_type(row.review_text, normalize_venue_type(parsed.get("venue_type")))
    llm_restaurant_name = normalize_restaurant_name(parsed.get("restaurant_name"), row.review_text)
    llm_location_mentions = normalize_location(parsed.get("location_mentions"))

    restaurant_name, restaurant_name_source = choose_scalar(
        rule_labels.restaurant_name,
        llm_restaurant_name,
        rule_labels.confidence["restaurant_name"],
    )
    venue_type, venue_type_source = choose_scalar(
        None if rule_labels.venue_type == "unknown" else rule_labels.venue_type,
        llm_venue_type if llm_venue_type != "unknown" else None,
        rule_labels.confidence["venue_type"],
        default="unknown",
    )
    food_nationality, food_nationality_source = choose_list(
        rule_labels.food_nationality,
        llm_food_nationality,
        rule_labels.confidence["food_nationality"],
    )
    food_type, food_type_source = choose_list(
        rule_labels.food_type,
        llm_food_type,
        rule_labels.confidence["food_type"],
    )
    ambience, ambience_source = choose_list(
        rule_labels.ambience,
        llm_ambience,
        rule_labels.confidence["ambience"],
    )
    price_level, price_level_source = choose_scalar(
        rule_labels.price_level,
        llm_price_level,
        rule_labels.confidence["price_level"],
    )
    location_mentions, location_source = choose_list(
        rule_labels.location_mentions,
        llm_location_mentions,
        rule_labels.confidence["location_mentions"],
    )

    record = {
        "doc_id": f"rev_{row.row_id:06d}",
        "row_id": row.row_id,
        "review_text": row.review_text,
        "review_text_clean": clean_text_for_output(row.review_text),
        "review_rating": row.rating,
        "restaurant_name": restaurant_name,
        "venue_type": venue_type,
        "food_nationality": food_nationality,
        "food_type": food_type,
        "ambience": ambience,
        "price_level": price_level,
        "location_mentions": location_mentions,
        "evidence": {
            "restaurant_name": normalize_text(evidence.get("restaurant_name")) or rule_labels.evidence["restaurant_name"],
            "food_nationality": unique_keep_order(rule_labels.evidence["food_nationality"] + normalize_list(evidence.get("food_nationality"))),
            "food_type": unique_keep_order(rule_labels.evidence["food_type"] + normalize_list(evidence.get("food_type"))),
            "ambience": unique_keep_order(rule_labels.evidence["ambience"] + normalize_list(evidence.get("ambience"))),
            "price_level": unique_keep_order(rule_labels.evidence["price_level"] + normalize_list(evidence.get("price_level"))),
            "location_mentions": unique_keep_order(rule_labels.evidence["location_mentions"] + normalize_location(evidence.get("location_mentions"))),
        },
        "rule_labels": {
            "restaurant_name": rule_labels.restaurant_name,
            "venue_type": rule_labels.venue_type,
            "food_nationality": rule_labels.food_nationality,
            "food_type": rule_labels.food_type,
            "ambience": rule_labels.ambience,
            "price_level": rule_labels.price_level,
            "location_mentions": rule_labels.location_mentions,
        },
        "llm_labels": {
            "restaurant_name": llm_restaurant_name,
            "venue_type": llm_venue_type,
            "food_nationality": llm_food_nationality,
            "food_type": llm_food_type,
            "ambience": llm_ambience,
            "price_level": llm_price_level,
            "location_mentions": llm_location_mentions,
        },
        "label_source": {
            "restaurant_name": restaurant_name_source,
            "venue_type": venue_type_source,
            "food_nationality": food_nationality_source,
            "food_type": food_type_source,
            "ambience": ambience_source,
            "price_level": price_level_source,
            "location_mentions": location_source,
            "review_rating": "raw_csv",
        },
        "confidence": {
            "restaurant_name": rule_labels.confidence["restaurant_name"],
            "venue_type": max(rule_labels.confidence["venue_type"], 0.6 if llm_venue_type != "unknown" else 0.0),
            "food_nationality": max(rule_labels.confidence["food_nationality"], 0.6 if llm_food_nationality else 0.0),
            "food_type": max(rule_labels.confidence["food_type"], 0.6 if llm_food_type else 0.0),
            "ambience": max(rule_labels.confidence["ambience"], 0.6 if llm_ambience else 0.0),
            "price_level": max(rule_labels.confidence["price_level"], 0.6 if llm_price_level else 0.0),
            "location_mentions": max(rule_labels.confidence["location_mentions"], 0.6 if llm_location_mentions else 0.0),
            "review_rating": 1.0,
        },
    }
    return record


def process_row(args: argparse.Namespace, row: ReviewRow) -> dict[str, Any]:
    prompt_review_text = truncate_review_text(row.review_text, args.max_review_chars)
    prompt_row = ReviewRow(row_id=row.row_id, review_text=prompt_review_text, rating=row.rating)
    rule_labels = extract_rule_labels(prompt_row)
    prompt = build_prompt(prompt_review_text, rule_labels)
    parsed, raw_response, elapsed = call_ollama(args, prompt)
    record = post_process(parsed, row, rule_labels)
    runtime_metrics = extract_runtime_metrics(raw_response, elapsed)
    record["latency_sec"] = runtime_metrics["timing"]["wall_latency_sec"]
    record["timing"] = runtime_metrics["timing"]
    record["token_usage"] = runtime_metrics["token_usage"]
    record["throughput"] = runtime_metrics["throughput"]
    record["prompt_review_chars"] = len(prompt_review_text)
    record["review_truncated"] = len(prompt_review_text) != len(row.review_text)
    record["model"] = args.model
    record["prompt_version"] = "v2_rule_first_hybrid"
    if not args.fast_mode:
        record["raw_model_content"] = raw_response["message"]["content"]
    record["done_reason"] = raw_response.get("done_reason")
    record["eval_notes"] = []
    return record


def build_error_record(row: ReviewRow, exc: Exception) -> dict[str, Any]:
    return {
        "doc_id": f"rev_{row.row_id:06d}",
        "row_id": row.row_id,
        "review_text": row.review_text,
        "review_rating": row.rating,
        "error": f"{type(exc).__name__}: {exc}",
    }


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_row_with_retries(args: argparse.Namespace, row: ReviewRow) -> dict[str, Any]:
    attempts = args.retries + 1
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            record = process_row(args, row)
            record["attempt"] = attempt
            return record
        except (json.JSONDecodeError, KeyError, urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(min(2 * attempt, 10))
    assert last_exc is not None
    raise last_exc


def sample_rows(rows: list[ReviewRow], sample_size: int, seed: int) -> list[ReviewRow]:
    if sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_preview_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "row_id",
        "review_rating",
        "restaurant_name",
        "venue_type",
        "food_nationality",
        "food_type",
        "ambience",
        "price_level",
        "location_mentions",
        "source_restaurant_name",
        "source_venue_type",
        "source_food_nationality",
        "source_food_type",
        "source_ambience",
        "source_price_level",
        "source_location_mentions",
        "conf_restaurant_name",
        "conf_venue_type",
        "conf_food_nationality",
        "conf_food_type",
        "conf_ambience",
        "conf_price_level",
        "conf_location_mentions",
        "rule_food_nationality",
        "rule_food_type",
        "rule_ambience",
        "rule_price_level",
        "rule_location_mentions",
        "llm_food_nationality",
        "llm_food_type",
        "llm_ambience",
        "llm_price_level",
        "llm_location_mentions",
        "review_text_clean",
        "evidence_food_nationality",
        "evidence_food_type",
        "evidence_ambience",
        "evidence_price_level",
        "evidence_location_mentions",
        "latency_sec",
        "total_duration_sec",
        "load_duration_sec",
        "prompt_eval_duration_sec",
        "eval_duration_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_per_sec",
        "total_tokens_per_sec",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "doc_id": record["doc_id"],
                    "row_id": record["row_id"],
                    "review_rating": record["review_rating"],
                    "restaurant_name": record["restaurant_name"] or "",
                    "venue_type": record["venue_type"],
                    "food_nationality": join_list(record["food_nationality"]),
                    "food_type": join_list(record["food_type"]),
                    "ambience": join_list(record["ambience"]),
                    "price_level": record["price_level"] or "",
                    "location_mentions": join_list(record["location_mentions"]),
                    "source_restaurant_name": record["label_source"]["restaurant_name"],
                    "source_venue_type": record["label_source"]["venue_type"],
                    "source_food_nationality": record["label_source"]["food_nationality"],
                    "source_food_type": record["label_source"]["food_type"],
                    "source_ambience": record["label_source"]["ambience"],
                    "source_price_level": record["label_source"]["price_level"],
                    "source_location_mentions": record["label_source"]["location_mentions"],
                    "conf_restaurant_name": record["confidence"]["restaurant_name"],
                    "conf_venue_type": record["confidence"]["venue_type"],
                    "conf_food_nationality": record["confidence"]["food_nationality"],
                    "conf_food_type": record["confidence"]["food_type"],
                    "conf_ambience": record["confidence"]["ambience"],
                    "conf_price_level": record["confidence"]["price_level"],
                    "conf_location_mentions": record["confidence"]["location_mentions"],
                    "rule_food_nationality": join_list(record["rule_labels"]["food_nationality"]),
                    "rule_food_type": join_list(record["rule_labels"]["food_type"]),
                    "rule_ambience": join_list(record["rule_labels"]["ambience"]),
                    "rule_price_level": record["rule_labels"]["price_level"] or "",
                    "rule_location_mentions": join_list(record["rule_labels"]["location_mentions"]),
                    "llm_food_nationality": join_list(record["llm_labels"]["food_nationality"]),
                    "llm_food_type": join_list(record["llm_labels"]["food_type"]),
                    "llm_ambience": join_list(record["llm_labels"]["ambience"]),
                    "llm_price_level": record["llm_labels"]["price_level"] or "",
                    "llm_location_mentions": join_list(record["llm_labels"]["location_mentions"]),
                    "review_text_clean": record.get("review_text_clean", ""),
                    "evidence_food_nationality": join_list(record["evidence"]["food_nationality"]),
                    "evidence_food_type": join_list(record["evidence"]["food_type"]),
                    "evidence_ambience": join_list(record["evidence"]["ambience"]),
                    "evidence_price_level": join_list(record["evidence"]["price_level"]),
                    "evidence_location_mentions": join_list(record["evidence"]["location_mentions"]),
                    "latency_sec": record.get("latency_sec", ""),
                    "total_duration_sec": record.get("timing", {}).get("total_duration_sec", ""),
                    "load_duration_sec": record.get("timing", {}).get("load_duration_sec", ""),
                    "prompt_eval_duration_sec": record.get("timing", {}).get("prompt_eval_duration_sec", ""),
                    "eval_duration_sec": record.get("timing", {}).get("eval_duration_sec", ""),
                    "prompt_tokens": record.get("token_usage", {}).get("prompt_tokens", ""),
                    "completion_tokens": record.get("token_usage", {}).get("completion_tokens", ""),
                    "total_tokens": record.get("token_usage", {}).get("total_tokens", ""),
                    "completion_tokens_per_sec": record.get("throughput", {}).get("completion_tokens_per_sec", ""),
                    "total_tokens_per_sec": record.get("throughput", {}).get("total_tokens_per_sec", ""),
                }
            )


def summarize(records: list[dict[str, Any]], errors: list[dict[str, Any]], args: argparse.Namespace, requested_rows: int | None = None) -> dict[str, Any]:
    latencies = [record["latency_sec"] for record in records if "latency_sec" in record]
    prompt_tokens = [record.get("token_usage", {}).get("prompt_tokens") for record in records if record.get("token_usage", {}).get("prompt_tokens") is not None]
    completion_tokens = [record.get("token_usage", {}).get("completion_tokens") for record in records if record.get("token_usage", {}).get("completion_tokens") is not None]
    total_tokens = [record.get("token_usage", {}).get("total_tokens") for record in records if record.get("token_usage", {}).get("total_tokens") is not None]
    total_duration = [record.get("timing", {}).get("total_duration_sec") for record in records if record.get("timing", {}).get("total_duration_sec") is not None]
    eval_duration = [record.get("timing", {}).get("eval_duration_sec") for record in records if record.get("timing", {}).get("eval_duration_sec") is not None]
    source_fields = ["restaurant_name", "venue_type", "food_nationality", "food_type", "ambience", "price_level", "location_mentions"]
    summary = {
        "model": args.model,
        "input": str(args.input),
        "base_url": args.base_url,
        "requested_rows": requested_rows if requested_rows is not None else len(records) + len(errors),
        "processed_rows": len(records) + len(errors),
        "success_count": len(records),
        "error_count": len(errors),
        "parallel": args.parallel,
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
        "keep_alive": args.keep_alive,
        "latency_avg_sec": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_median_sec": round(statistics.median(latencies), 3) if latencies else None,
        "latency_max_sec": round(max(latencies), 3) if latencies else None,
        "prompt_tokens_total": sum(prompt_tokens) if prompt_tokens else None,
        "completion_tokens_total": sum(completion_tokens) if completion_tokens else None,
        "all_tokens_total": sum(total_tokens) if total_tokens else None,
        "prompt_tokens_avg": round(statistics.mean(prompt_tokens), 3) if prompt_tokens else None,
        "completion_tokens_avg": round(statistics.mean(completion_tokens), 3) if completion_tokens else None,
        "all_tokens_avg": round(statistics.mean(total_tokens), 3) if total_tokens else None,
        "total_duration_sum_sec": round(sum(total_duration), 3) if total_duration else None,
        "eval_duration_sum_sec": round(sum(eval_duration), 3) if eval_duration else None,
        "completion_tokens_per_sec_run": safe_div(sum(completion_tokens), sum(eval_duration)) if completion_tokens and eval_duration else None,
        "all_tokens_per_sec_run": safe_div(sum(total_tokens), sum(total_duration)) if total_tokens and total_duration else None,
        "rows_with_restaurant_name": sum(1 for record in records if record["restaurant_name"]),
        "rows_with_food_nationality": sum(1 for record in records if record["food_nationality"]),
        "rows_with_food_type": sum(1 for record in records if record["food_type"]),
        "rows_with_ambience": sum(1 for record in records if record["ambience"]),
        "rows_with_price_level": sum(1 for record in records if record["price_level"]),
        "rows_with_location_mentions": sum(1 for record in records if record["location_mentions"]),
        "label_source_counts": {
            field: {
                source: sum(1 for record in records if record["label_source"][field] == source)
                for source in ["rule", "llm", "rule+llm", "default"]
            }
            for field in source_fields
        },
    }
    return summary


def build_perf_report(records: list[dict[str, Any]], errors: list[dict[str, Any]], args: argparse.Namespace, run_started_at: float, run_finished_at: float) -> dict[str, Any]:
    wall_time_sec = round(run_finished_at - run_started_at, 3)
    prompt_tokens_total = sum(record.get("token_usage", {}).get("prompt_tokens") or 0 for record in records)
    completion_tokens_total = sum(record.get("token_usage", {}).get("completion_tokens") or 0 for record in records)
    total_tokens_total = sum(record.get("token_usage", {}).get("total_tokens") or 0 for record in records)
    return {
        "run": {
            "started_at_epoch": round(run_started_at, 3),
            "finished_at_epoch": round(run_finished_at, 3),
            "wall_time_sec": wall_time_sec,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "model": args.model,
            "base_url": args.base_url,
            "parallel": args.parallel,
            "temperature": args.temperature,
            "num_ctx": args.num_ctx,
            "keep_alive": args.keep_alive,
        },
        "counts": {
            "requested_rows": len(records) + len(errors),
            "success_rows": len(records),
            "error_rows": len(errors),
        },
        "tokens": {
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens_total": completion_tokens_total,
            "all_tokens_total": total_tokens_total,
        },
        "throughput": {
            "successful_rows_per_sec_wall": safe_div(len(records), wall_time_sec),
            "all_tokens_per_sec_wall": safe_div(total_tokens_total, wall_time_sec),
            "completion_tokens_per_sec_wall": safe_div(completion_tokens_total, wall_time_sec),
        },
    }


def write_perf_rows_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "row_id",
        "review_rating",
        "latency_sec",
        "total_duration_sec",
        "load_duration_sec",
        "prompt_eval_duration_sec",
        "eval_duration_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_per_sec",
        "total_tokens_per_sec",
        "done_reason",
        "model",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "doc_id": record["doc_id"],
                    "row_id": record["row_id"],
                    "review_rating": record["review_rating"],
                    "latency_sec": record.get("timing", {}).get("wall_latency_sec", ""),
                    "total_duration_sec": record.get("timing", {}).get("total_duration_sec", ""),
                    "load_duration_sec": record.get("timing", {}).get("load_duration_sec", ""),
                    "prompt_eval_duration_sec": record.get("timing", {}).get("prompt_eval_duration_sec", ""),
                    "eval_duration_sec": record.get("timing", {}).get("eval_duration_sec", ""),
                    "prompt_tokens": record.get("token_usage", {}).get("prompt_tokens", ""),
                    "completion_tokens": record.get("token_usage", {}).get("completion_tokens", ""),
                    "total_tokens": record.get("token_usage", {}).get("total_tokens", ""),
                    "completion_tokens_per_sec": record.get("throughput", {}).get("completion_tokens_per_sec", ""),
                    "total_tokens_per_sec": record.get("throughput", {}).get("total_tokens_per_sec", ""),
                    "done_reason": record.get("done_reason", ""),
                    "model": record.get("model", ""),
                }
            )


def build_progress_snapshot(
    total_rows: int,
    successes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    inflight_count: int,
    run_started_at: float,
) -> dict[str, Any]:
    completed = len(successes) + len(errors)
    elapsed = max(time.time() - run_started_at, 0.001)
    rows_per_sec = completed / elapsed if completed else 0.0
    remaining = max(total_rows - completed, 0)
    eta_sec = round(remaining / rows_per_sec, 3) if rows_per_sec > 0 else None
    return {
        "completed": completed,
        "total": total_rows,
        "success_count": len(successes),
        "error_count": len(errors),
        "inflight_count": inflight_count,
        "rows_per_sec_wall": round(rows_per_sec, 6),
        "elapsed_sec": round(elapsed, 3),
        "eta_sec": eta_sec,
        "updated_at_epoch": round(time.time(), 3),
    }


def write_progress_json(path: Path, snapshot: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    if args.parallel < 1:
        print("--parallel must be >= 1", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    rows = load_reviews(args.input, start_row=args.start_row, limit_rows=args.limit_rows)
    if not rows:
        print("No reviews loaded from input.", file=sys.stderr)
        return 2

    sampled = sample_rows(rows, args.sample_size, args.seed)
    if args.resume_dir is not None:
        run_dir = args.resume_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = args.output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = time.time()

    labels_path = run_dir / "labels.jsonl"
    errors_path = run_dir / "errors.jsonl"
    summary_path = run_dir / "summary.json"
    perf_report_path = run_dir / "perf_report.json"
    perf_rows_path = run_dir / "perf_rows.csv"
    config_path = run_dir / "run_config.json"
    preview_path = run_dir / "preview.csv"
    progress_path = run_dir / "progress.json"

    successes = load_jsonl_records(labels_path)
    errors = load_jsonl_records(errors_path)
    processed_ids = {record.get("row_id") for record in successes + errors if record.get("row_id") is not None}
    pending_rows = [row for row in sampled if row.row_id not in processed_ids]

    lock = threading.Lock()
    inflight: dict[int, float] = {}
    stop_heartbeat = threading.Event()

    print(f"Loaded {len(rows)} candidate rows; labeling {len(sampled)} rows with {args.model}")
    print(f"Output directory: {run_dir}")
    if processed_ids:
        print(f"Resuming run with {len(processed_ids)} rows already processed; {len(pending_rows)} rows remaining")

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(args.heartbeat_sec):
            with lock:
                snapshot = build_progress_snapshot(len(sampled), successes, errors, len(inflight), run_started_at)
                write_progress_json(progress_path, snapshot)
            eta_text = f"{snapshot['eta_sec']}s" if snapshot["eta_sec"] is not None else "unknown"
            print(
                f"[heartbeat] completed={snapshot['completed']}/{snapshot['total']} "
                f"ok={snapshot['success_count']} err={snapshot['error_count']} "
                f"inflight={snapshot['inflight_count']} rps={snapshot['rows_per_sec_wall']} eta={eta_text}"
            )

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_map = {}
            for row in pending_rows:
                inflight[row.row_id] = time.time()
                future_map[executor.submit(process_row_with_retries, args, row)] = row
            for index, future in enumerate(as_completed(future_map), start=1):
                row = future_map[future]
                try:
                    result = future.result()
                except (json.JSONDecodeError, KeyError, urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
                    error_record = build_error_record(row, exc)
                    with lock:
                        errors.append(error_record)
                        inflight.pop(row.row_id, None)
                        append_jsonl_record(errors_path, error_record)
                        write_progress_json(progress_path, build_progress_snapshot(len(sampled), successes, errors, len(inflight), run_started_at))
                    print(f"[{index}/{len(sampled)}] row={row.row_id} ERROR {error_record['error']}")
                    continue
                with lock:
                    successes.append(result)
                    inflight.pop(row.row_id, None)
                    append_jsonl_record(labels_path, result)
                    write_progress_json(progress_path, build_progress_snapshot(len(sampled), successes, errors, len(inflight), run_started_at))
                print(
                    f"[{index}/{len(sampled)}] row={row.row_id} "
                    f"ok latency={result['latency_sec']:.3f}s "
                    f"rating={result['review_rating']}"
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Progress has been checkpointed; resume with --resume-dir.")

    stop_heartbeat.set()
    heartbeat_thread.join(timeout=1)

    successes.sort(key=lambda item: item["row_id"])
    errors.sort(key=lambda item: item["row_id"])
    write_preview_csv(preview_path, successes)
    write_perf_rows_csv(perf_rows_path, successes)

    summary = summarize(successes, errors, args, requested_rows=len(sampled))
    perf_report = build_perf_report(successes, errors, args, run_started_at, time.time())
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with perf_report_path.open("w", encoding="utf-8") as handle:
        json.dump(perf_report, handle, ensure_ascii=False, indent=2)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "host_runtime": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    write_progress_json(progress_path, build_progress_snapshot(len(sampled), successes, errors, 0, run_started_at))

    print("")
    print("Run summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"labels:  {labels_path}")
    print(f"preview: {preview_path}")
    print(f"errors:  {errors_path}")
    print(f"summary: {summary_path}")
    print(f"progress:{progress_path}")
    print(f"perf:    {perf_report_path}")
    print(f"perfrow: {perf_rows_path}")
    if interrupted:
        return 130
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
