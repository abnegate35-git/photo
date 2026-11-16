"""GUI 專用的查詢與小型寫入。介面程式不直接寫 SQL，全部經過這裡。

讀取函式可以在任何執行緒呼叫（`DBManager` 會給每條執行緒自己的唯讀連線）；GUI 一律在背景
執行緒呼叫它們，再把結果交回主執行緒顯示。寫入函式經 `DBManager` 的 writer 執行緒，
回傳時已 commit。

導航節點以字串表示（也就是左側樹狀檢視的 item id），`list_photos()` 依它決定要列哪些照片::

    all                    全部照片
    events                 已歸入任何事件的照片        event:<id> / event:none
    burst:<group_id>       一組連拍
    persons                有已知人物的照片            person:<id>
    tags                   有任何標籤的照片            tag:<id>
    locations              有地點的照片                loc:<國碼> / city:<國碼>|<城市>
    staging                待清理隔離區的全部          staging:blurry / staging:burst / staging:screenshot

隔離區三類的判定與 `core.export_manager.ExportManager._categorize()` 相同（同時符合多項時
依 螢幕截圖 → 模糊/廢片 → 連拍次佳 的順序取第一個），所以這裡看到的數量就是匯出時會進
`99_Review_Staging` 各子資料夾的數量（未指定 `blur_threshold` 時）。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..utils.clustering import person_display_name
from .db_manager import DBManager

#: 隔離區各類別的 SQL 條件（別名 p = photos）
STAGING_SQL: dict[str, str] = {
    "screenshot": "p.is_screenshot = 1",
    "blurry": "p.is_screenshot = 0 AND p.is_rejected = 1 AND COALESCE(p.reject_reason, '') != 'duplicate'",
    "burst": "p.is_screenshot = 0 AND ((p.is_rejected = 1 AND p.reject_reason = 'duplicate') "
             "OR (p.is_rejected = 0 AND p.is_burst_duplicate = 1))",
}
STAGING_LABELS: dict[str, str] = {
    "blurry": "疑似模糊廢片",
    "burst": "連拍次佳照片",
    "screenshot": "螢幕截圖",
}

_ITEM_COLS = ("p.id, p.path, p.file_name, p.sha256, p.taken_at, p.is_best_shot, p.is_burst_duplicate, "
              "p.burst_group_id, p.is_rejected, p.is_screenshot, p.is_favorite")


# --------------------------------------------------------------------------- #
# 資料結構
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PhotoItem:
    """照片牆上的一格。"""

    id: int
    path: str
    file_name: str
    sha256: bytes | None
    taken_at: str | None
    is_best_shot: bool
    is_burst_duplicate: bool
    burst_group_id: int | None
    is_rejected: bool
    is_screenshot: bool
    is_favorite: bool


@dataclass(slots=True)
class BurstNode:
    group_id: int
    count: int
    best_file: str | None                # 被選為 Best Shot 的檔名


@dataclass(slots=True)
class EventNode:
    id: int | None                       # None = 未歸入事件
    name: str
    count: int
    bursts: list[BurstNode] = field(default_factory=list)


@dataclass(slots=True)
class PersonNode:
    id: int
    display_name: str
    is_locked: bool
    is_named: bool
    count: int                           # 有這個人的照片數


@dataclass(slots=True)
class TagNode:
    id: int
    name: str
    display_name: str
    count: int


@dataclass(slots=True)
class CountryNode:
    code: str
    count: int
    cities: list[tuple[str, int]] = field(default_factory=list)


@dataclass(slots=True)
class SidebarData:
    total: int = 0
    events: list[EventNode] = field(default_factory=list)
    persons: list[PersonNode] = field(default_factory=list)
    tags: list[TagNode] = field(default_factory=list)
    locations: list[CountryNode] = field(default_factory=list)
    staging: dict[str, int] = field(default_factory=dict)

    @property
    def in_events(self) -> int:
        return sum(e.count for e in self.events if e.id is not None)

    @property
    def burst_groups(self) -> int:
        return sum(len(e.bursts) for e in self.events)


@dataclass(slots=True)
class FaceInfo:
    id: int
    bbox: tuple[float, float, float, float]      # (x, y, w, h)，原圖正規化 0..1
    det_score: float
    person_id: int | None
    person_name: str                             # 顯示名稱；未指派為「未知」
    is_locked: bool


@dataclass(slots=True)
class PhotoDetail:
    row: dict[str, Any]                          # photos 的整列
    event_name: str | None
    tags: list[tuple[str, float | None, str]]    # (顯示名稱, 信心度, 'auto' | 'manual')
    faces: list[FaceInfo]

    @property
    def short_hash(self) -> str:
        sha = self.row.get("sha256")
        return bytes(sha).hex()[:6] if sha else ""


@dataclass(slots=True)
class ExportBatchInfo:
    id: int
    mode: str
    dest_root: str
    layout: str                                  # export_batches.template
    state: str
    started_at: int
    total: int
    counts: dict[str, dict[str, int]]            # action -> state -> 筆數（取自 export_logs）

    def describe_counts(self) -> str:
        """例如 `copy: done 8｜hardlink: done 10, failed 1`。"""
        parts = []
        for action in sorted(self.counts):
            states = ", ".join(f"{s} {n}" for s, n in sorted(self.counts[action].items()))
            parts.append(f"{action}: {states}")
        return "｜".join(parts) or "（沒有紀錄）"

    @property
    def can_rollback(self) -> bool:
        return any(states.get("done", 0) for states in self.counts.values())


# --------------------------------------------------------------------------- #
# 讀取
# --------------------------------------------------------------------------- #
def load_sidebar(db: DBManager) -> SidebarData:
    """左側導航樹需要的所有計數，一次查完。遺失的檔案（`is_missing = 1`）不計入。"""
    data = SidebarData()
    data.total = int(db.query_value("SELECT COUNT(*) FROM photos WHERE is_missing = 0", default=0))

    # ---- 事件與連拍 ----
    bursts: dict[int | None, list[BurstNode]] = {}
    for r in db.query(
        "SELECT burst_group_id, COUNT(*) AS n, MIN(event_id) AS event_id, "
        "MAX(CASE WHEN is_best_shot = 1 THEN file_name END) AS best "
        "FROM photos WHERE burst_group_id IS NOT NULL AND is_missing = 0 "
        "GROUP BY burst_group_id ORDER BY MIN(taken_ts), burst_group_id"
    ):
        bursts.setdefault(r["event_id"], []).append(BurstNode(r["burst_group_id"], r["n"], r["best"]))
    for r in db.query(
        "SELECT e.id, e.name, e.start_ts, e.location_label, "
        "(SELECT COUNT(*) FROM photos p WHERE p.event_id = e.id AND p.is_missing = 0) AS n "
        "FROM events e ORDER BY e.start_ts DESC, e.id DESC"
    ):
        name = r["name"] or _event_fallback_name(r["start_ts"], r["location_label"])
        data.events.append(EventNode(r["id"], name, r["n"], bursts.get(r["id"], [])))
    orphan = int(db.query_value(
        "SELECT COUNT(*) FROM photos WHERE event_id IS NULL AND is_missing = 0", default=0))
    data.events.append(EventNode(None, "未歸入事件", orphan, bursts.get(None, [])))

    # ---- 人物 ----
    for r in db.query(
        "SELECT pe.id, pe.name, pe.is_locked, COUNT(DISTINCT f.photo_id) AS n "
        "FROM persons pe LEFT JOIN faces f ON f.person_id = pe.id AND f.is_ignored = 0 "
        "AND f.photo_id IN (SELECT id FROM photos WHERE is_missing = 0) "
        "WHERE pe.is_hidden = 0 GROUP BY pe.id "
        "ORDER BY (pe.name IS NULL OR pe.name = ''), pe.name COLLATE NOCASE, n DESC, pe.id"
    ):
        data.persons.append(PersonNode(
            r["id"], person_display_name(r["id"], r["name"]), bool(r["is_locked"]), bool(r["name"]), r["n"]))

    # ---- 標籤 ----
    for r in db.query(
        "SELECT t.id, t.name, t.display_name, COUNT(*) AS n FROM photo_tags pt "
        "JOIN tags t ON t.id = pt.tag_id JOIN photos p ON p.id = pt.photo_id "
        "WHERE pt.is_rejected = 0 AND p.is_missing = 0 GROUP BY t.id ORDER BY n DESC, t.name"
    ):
        data.tags.append(TagNode(r["id"], r["name"], r["display_name"] or r["name"], r["n"]))

    # ---- 地點 ----
    countries: dict[str, CountryNode] = {}
    for r in db.query(
        "SELECT country_code, COALESCE(NULLIF(city, ''), '（未知城市）') AS city, COUNT(*) AS n FROM photos "
        "WHERE country_code IS NOT NULL AND country_code != '' AND is_missing = 0 "
        "GROUP BY country_code, city ORDER BY country_code, n DESC, city"
    ):
        node = countries.setdefault(r["country_code"], CountryNode(r["country_code"], 0))
        node.count += r["n"]
        node.cities.append((r["city"], r["n"]))
    data.locations = sorted(countries.values(), key=lambda c: (-c.count, c.code))

    # ---- 隔離區 ----
    for key, cond in STAGING_SQL.items():
        data.staging[key] = int(db.query_value(
            f"SELECT COUNT(*) FROM photos p WHERE p.is_missing = 0 AND {cond}", default=0))
    return data


def _event_fallback_name(start_ts: int, location: str | None) -> str:
    import time

    day = time.strftime("%Y-%m-%d", time.gmtime(start_ts))
    return f"{day} {location}" if location else day


def _node_filter(node: str) -> tuple[str, tuple[Any, ...]]:
    """導航節點 -> (WHERE 條件, 參數)。不認得的節點丟 ValueError。"""
    kind, _, arg = node.partition(":")
    if node == "all":
        return "1", ()
    if node == "events":
        return "p.event_id IS NOT NULL", ()
    if kind == "event":
        return ("p.event_id IS NULL", ()) if arg == "none" else ("p.event_id = ?", (int(arg),))
    if kind == "burst":
        return "p.burst_group_id = ?", (int(arg),)
    if node == "persons":
        return ("EXISTS (SELECT 1 FROM faces f JOIN persons pe ON pe.id = f.person_id "
                "WHERE f.photo_id = p.id AND f.is_ignored = 0 AND pe.is_hidden = 0)", ())
    if kind == "person":
        return ("EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id AND f.person_id = ? AND f.is_ignored = 0)",
                (int(arg),))
    if node == "tags":
        return "EXISTS (SELECT 1 FROM photo_tags pt WHERE pt.photo_id = p.id AND pt.is_rejected = 0)", ()
    if kind == "tag":
        return ("EXISTS (SELECT 1 FROM photo_tags pt WHERE pt.photo_id = p.id AND pt.tag_id = ? "
                "AND pt.is_rejected = 0)", (int(arg),))
    if node == "locations":
        return "p.country_code IS NOT NULL AND p.country_code != ''", ()
    if kind == "loc":
        return "p.country_code = ?", (arg,)
    if kind == "city":
        code, _, city = arg.partition("|")
        if city == "（未知城市）":
            return "p.country_code = ? AND (p.city IS NULL OR p.city = '')", (code,)
        return "p.country_code = ? AND p.city = ?", (code, city)
    if node == "staging":
        return "(" + " OR ".join(f"({c})" for c in STAGING_SQL.values()) + ")", ()
    if kind == "staging" and arg in STAGING_SQL:
        return STAGING_SQL[arg], ()
    raise ValueError(f"未知的導航節點：{node!r}")


def list_photos(db: DBManager, node: str) -> list[PhotoItem]:
    """某個導航節點底下的照片，依拍攝時間排序。連拍群組內 Best Shot 排最前面。"""
    where, params = _node_filter(node)
    order = "p.is_best_shot DESC, p.taken_ts, p.id" if node.startswith("burst:") else "p.taken_ts, p.id"
    rows = db.query(
        f"SELECT {_ITEM_COLS} FROM photos p WHERE p.is_missing = 0 AND {where} ORDER BY {order}", params)
    return [
        PhotoItem(r["id"], r["path"], r["file_name"], r["sha256"], r["taken_at"], bool(r["is_best_shot"]),
                  bool(r["is_burst_duplicate"]), r["burst_group_id"], bool(r["is_rejected"]),
                  bool(r["is_screenshot"]), bool(r["is_favorite"]))
        for r in rows
    ]


def photo_detail(db: DBManager, photo_id: int) -> PhotoDetail | None:
    """右側屬性面板需要的全部資料。照片不存在回傳 None。"""
    row = db.query_one("SELECT * FROM photos WHERE id = ?", (photo_id,))
    if row is None:
        return None
    event = None
    if row["event_id"] is not None:
        e = db.query_one("SELECT name, start_ts, location_label FROM events WHERE id = ?", (row["event_id"],))
        if e is not None:
            event = e["name"] or _event_fallback_name(e["start_ts"], e["location_label"])
    tags = [
        (r["display_name"] or r["name"], r["score"], r["source"])
        for r in db.query(
            "SELECT t.name, t.display_name, pt.score, pt.source FROM photo_tags pt JOIN tags t ON t.id = pt.tag_id "
            "WHERE pt.photo_id = ? AND pt.is_rejected = 0 ORDER BY (pt.source = 'manual') DESC, pt.score DESC, t.name",
            (photo_id,))
    ]
    faces = [
        FaceInfo(
            id=r["id"], bbox=(r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]), det_score=r["det_score"],
            person_id=r["person_id"],
            person_name="未知" if r["person_id"] is None else person_display_name(r["person_id"], r["name"]),
            is_locked=bool(r["is_locked"]),
        )
        for r in db.query(
            "SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.det_score, f.person_id, pe.name, "
            "COALESCE(pe.is_locked, 0) AS is_locked FROM faces f LEFT JOIN persons pe ON pe.id = f.person_id "
            "WHERE f.photo_id = ? AND f.is_ignored = 0 ORDER BY f.bbox_x, f.id",
            (photo_id,))
    ]
    return PhotoDetail(row=dict(row), event_name=event, tags=tags, faces=faces)


def export_history(db: DBManager, limit: int = 200) -> list[ExportBatchInfo]:
    """匯出批次清單（最新的在前），連同 `export_logs` 依 `action` 與 `state` 的統計。"""
    counts: dict[int, dict[str, dict[str, int]]] = {}
    for r in db.query("SELECT batch_id, action, state, COUNT(*) AS n FROM export_logs GROUP BY batch_id, action, state"):
        counts.setdefault(r["batch_id"], {}).setdefault(r["action"], {})[r["state"]] = r["n"]
    return [
        ExportBatchInfo(r["id"], r["mode"], r["dest_root"], r["template"], r["state"], r["started_at"],
                        r["total"], counts.get(r["id"], {}))
        for r in db.query(
            "SELECT id, mode, dest_root, template, state, started_at, total FROM export_batches "
            "ORDER BY id DESC LIMIT ?", (limit,))
    ]


# --------------------------------------------------------------------------- #
# 寫入（使用者在介面上的決定）
# --------------------------------------------------------------------------- #
def rename_person(db: DBManager, person_id: int, name: str) -> None:
    """替人物命名，並同時鎖定（`is_locked = 1`）：使用者確認過的群組，重新分群不可覆蓋或拆散。

    Raises:
        ValueError: 名稱是空的，或人物不存在。
    """
    name = name.strip()
    if not name:
        raise ValueError("人物名稱不可以是空的")

    def op(conn: sqlite3.Connection) -> int:
        return conn.execute(
            "UPDATE persons SET name = ?, is_locked = 1, updated_at = unixepoch() WHERE id = ?", (name, person_id)
        ).rowcount

    if db.write(op) != 1:
        raise ValueError(f"找不到人物 {person_id}")


def name_face(db: DBManager, face_id: int, name: str) -> int:
    """替一張臉命名。回傳它所屬的 person id。

    * 臉已屬於某個人物 → 等同替那個人物改名並鎖定。
    * 臉還沒有歸屬 → 已有同名人物就歸入，否則建立新人物；這張臉標成手動指派
      （`assign_src = 'manual'`），重新分群不會動它。
    """
    name = name.strip()
    if not name:
        raise ValueError("人物名稱不可以是空的")

    def op(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT person_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if row is None:
            raise ValueError(f"找不到人臉 {face_id}")
        if row["person_id"] is not None:
            person_id = int(row["person_id"])
            conn.execute("UPDATE persons SET name = ?, is_locked = 1, updated_at = unixepoch() WHERE id = ?",
                         (name, person_id))
            return person_id
        existing = conn.execute("SELECT id FROM persons WHERE name = ? ORDER BY id LIMIT 1", (name,)).fetchone()
        if existing is not None:
            person_id = int(existing["id"])
            conn.execute("UPDATE persons SET is_locked = 1, updated_at = unixepoch() WHERE id = ?", (person_id,))
        else:
            person_id = int(conn.execute(
                "INSERT INTO persons(name, is_locked, cover_face_id) VALUES (?, 1, ?)", (name, face_id)).lastrowid)
        conn.execute("UPDATE faces SET person_id = ?, assign_src = 'manual' WHERE id = ?", (person_id, face_id))
        conn.execute("UPDATE persons SET face_count = (SELECT COUNT(*) FROM faces WHERE person_id = persons.id) "
                     "WHERE id = ?", (person_id,))
        return person_id

    return db.write(op)


def set_person_locked(db: DBManager, person_id: int, locked: bool) -> None:
    """鎖定或解鎖人物群組（不改名稱）。"""
    db.execute("UPDATE persons SET is_locked = ?, updated_at = unixepoch() WHERE id = ?",
               (int(locked), person_id)).result()
