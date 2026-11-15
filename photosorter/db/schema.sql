-- ============================================================================
-- photosorter schema  (PRAGMA user_version = 1)
--
-- 約定
--   * 時間戳一律 INTEGER epoch 秒。
--   * 向量一律 BLOB = numpy float32 little-endian 的 tobytes()，已 L2 正規化。
--     這正好是 sqlite-vec 的原生向量格式，載入擴充後可直接
--     `vec_distance_cosine(vector, :q)`，不必轉檔。
--   * 座標（人臉框）是「套用 EXIF 方向後的原圖」上的正規化值 0..1。
--   * 本檔只在全新資料庫執行一次；之後的結構變更寫在 db_manager.MIGRATIONS。
--   * 連線層級的 PRAGMA（WAL、synchronous、cache…）不在這裡，見 db_manager._apply_pragmas。
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 模型登錄：向量是哪個模型算的。換模型時舊向量不會被誤用。
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,                 -- 'siglip-base-patch16-224' / 'buffalo_l'
    kind        TEXT    NOT NULL CHECK (kind IN ('image_embed','text_embed','face_detect','face_embed')),
    dim         INTEGER,
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    UNIQUE (name, kind)
);

-- ---------------------------------------------------------------------------
-- 掃描工作階段：給 UI 顯示歷史與統計。續掃不靠它，靠 photos.process_status。
-- ---------------------------------------------------------------------------
CREATE TABLE scan_sessions (
    id           INTEGER PRIMARY KEY,
    roots        TEXT    NOT NULL,                -- JSON 陣列
    state        TEXT    NOT NULL DEFAULT 'running'
                         CHECK (state IN ('running','finished','stopped','failed')),
    started_at   INTEGER NOT NULL DEFAULT (unixepoch()),
    finished_at  INTEGER,
    stats        TEXT                             -- JSON：PipelineStats 快照
);

-- ---------------------------------------------------------------------------
-- 事件（自動時間/地點聚類）
-- ---------------------------------------------------------------------------
CREATE TABLE events (
    id              INTEGER PRIMARY KEY,
    name            TEXT,                         -- 使用者命名；NULL 時 UI 以日期+地點顯示
    start_ts        INTEGER NOT NULL,
    end_ts          INTEGER NOT NULL,
    photo_count     INTEGER NOT NULL DEFAULT 0,
    center_lat      REAL,
    center_lon      REAL,
    location_label  TEXT,                         -- 例如 'Kyoto, JP'
    cover_photo_id  INTEGER REFERENCES photos(id) ON DELETE SET NULL,
    is_user_edited  INTEGER NOT NULL DEFAULT 0,   -- 1 = 使用者改過，重新聚類時不可覆蓋
    created_at      INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at      INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX idx_events_start ON events(start_ts);

-- ---------------------------------------------------------------------------
-- 照片：一個路徑一列。內容相同的兩個檔案是兩列（sha256 相同）。
-- 這張表刻意不放任何 BLOB 向量，讓瀏覽/篩選的全表掃描保持又小又快。
-- ---------------------------------------------------------------------------
CREATE TABLE photos (
    id              INTEGER PRIMARY KEY,

    -- 檔案
    path            TEXT    NOT NULL UNIQUE,      -- 絕對路徑
    dir_path        TEXT    NOT NULL,
    file_name       TEXT    NOT NULL,
    ext             TEXT    NOT NULL,             -- 小寫、不含點
    file_size       INTEGER NOT NULL,
    mtime_ns        INTEGER NOT NULL,
    sha256          BLOB,                         -- 32 bytes

    -- 影像
    width           INTEGER,                      -- 套用方向後
    height          INTEGER,
    orientation     INTEGER,                      -- EXIF 原始值 1..8

    -- EXIF
    taken_at        TEXT,                         -- 'YYYY-MM-DDTHH:MM:SS' 相機當地時間
    taken_ts        INTEGER,                      -- 排序/間隔用（把當地時間當 UTC 換算）
    taken_src       TEXT    CHECK (taken_src IN ('exif','filename','mtime')),
    camera_make     TEXT,
    camera_model    TEXT,
    lens_model      TEXT,
    iso             INTEGER,
    f_number        REAL,
    exposure_time   REAL,
    focal_length    REAL,
    gps_lat         REAL,
    gps_lon         REAL,
    gps_alt         REAL,
    country_code    TEXT,
    admin1          TEXT,
    city            TEXT,

    -- 品質
    blur_score      REAL,                         -- Variance of Laplacian，越小越糊
    dhash           INTEGER,                      -- 有號 64-bit
    phash           INTEGER,                      -- 選配（imagehash）
    is_screenshot   INTEGER NOT NULL DEFAULT 0,

    -- 使用者整理（非破壞性：只記在這裡，不動原檔）
    is_rejected     INTEGER NOT NULL DEFAULT 0,   -- 廢片
    reject_reason   TEXT,                         -- 'blur' | 'duplicate' | 'manual' …
    is_favorite     INTEGER NOT NULL DEFAULT 0,
    rating          INTEGER,                      -- 0..5
    dup_group_id    INTEGER,                      -- 近似重複群（dHash 漢明距離）
    event_id        INTEGER REFERENCES events(id) ON DELETE SET NULL,

    -- 管線狀態
    process_status  INTEGER NOT NULL DEFAULT 0,   -- core.status.ProcessStatus 位元
    error_count     INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    is_missing      INTEGER NOT NULL DEFAULT 0,   -- 掃描時檔案已不在（保留紀錄，不刪）
    last_scan_id    INTEGER REFERENCES scan_sessions(id) ON DELETE SET NULL,
    created_at      INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at      INTEGER NOT NULL DEFAULT (unixepoch())
);
-- Stage 0：以內容找搬移/重複
CREATE INDEX idx_photos_sha256   ON photos(sha256);
-- 時間軸、事件聚類
CREATE INDEX idx_photos_taken    ON photos(taken_ts);
CREATE INDEX idx_photos_event    ON photos(event_id, taken_ts);
-- 資料夾瀏覽與 Stage 0 依根目錄載入索引（path 的 UNIQUE 索引可做前綴範圍查詢）
CREATE INDEX idx_photos_dir      ON photos(dir_path);
-- 近似重複
CREATE INDEX idx_photos_dhash    ON photos(dhash) WHERE dhash IS NOT NULL;
-- 地點瀏覽
CREATE INDEX idx_photos_place    ON photos(country_code, city) WHERE country_code IS NOT NULL;
-- 待 geocode
CREATE INDEX idx_photos_gps_todo ON photos(id) WHERE gps_lat IS NOT NULL AND country_code IS NULL;
-- 常用篩選（部分索引：命中的列很少，索引極小）
CREATE INDEX idx_photos_rejected ON photos(id) WHERE is_rejected = 1;
CREATE INDEX idx_photos_shot     ON photos(id) WHERE is_screenshot = 1;
CREATE INDEX idx_photos_missing  ON photos(id) WHERE is_missing = 1;
CREATE INDEX idx_photos_camera   ON photos(camera_model) WHERE camera_model IS NOT NULL;

CREATE TRIGGER trg_photos_touch AFTER UPDATE ON photos
WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE photos SET updated_at = unixepoch() WHERE id = NEW.id;
END;

-- ---------------------------------------------------------------------------
-- 影像語意向量（SigLIP 768d）。獨立一張表：photos 保持精簡，換模型可並存。
-- ---------------------------------------------------------------------------
CREATE TABLE embeddings (
    id          INTEGER PRIMARY KEY,
    photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    model_id    INTEGER NOT NULL REFERENCES models(id),
    dim         INTEGER NOT NULL,
    vector      BLOB    NOT NULL,                 -- float32 LE x dim，L2 正規化
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    UNIQUE (photo_id, model_id)
);
CREATE INDEX idx_embeddings_model ON embeddings(model_id, photo_id);

-- ---------------------------------------------------------------------------
-- 標籤。prompt 非 NULL 的標籤由 SigLIP 自動判定；文字向量快取在 text_vector。
-- ---------------------------------------------------------------------------
CREATE TABLE tags (
    id             INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    display_name   TEXT,                          -- 中文顯示名稱
    prompt         TEXT,                          -- 餵 text encoder 的英文句子；NULL = 純手動標籤
    threshold      REAL,                          -- NULL = 用 siglip.default_threshold
    gate_role      TEXT    NOT NULL DEFAULT 'none'
                           CHECK (gate_role IN ('none','skip_face','has_person')),
    parent_id      INTEGER REFERENCES tags(id) ON DELETE SET NULL,
    is_enabled     INTEGER NOT NULL DEFAULT 1,
    text_vector    BLOB,                          -- float32 LE，L2 正規化
    text_model_id  INTEGER REFERENCES models(id), -- text_vector 是哪個模型算的
    created_at     INTEGER NOT NULL DEFAULT (unixepoch())
);
-- prompt 一改，快取的文字向量就作廢（下次啟動會重新編碼）
CREATE TRIGGER trg_tags_prompt_changed AFTER UPDATE OF prompt ON tags
WHEN OLD.prompt IS NOT NEW.prompt
BEGIN
    UPDATE tags SET text_vector = NULL, text_model_id = NULL WHERE id = NEW.id;
END;

CREATE TABLE photo_tags (
    photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES tags(id)   ON DELETE CASCADE,
    score       REAL,                             -- sigmoid 機率；手動標籤為 NULL
    source      TEXT    NOT NULL DEFAULT 'auto' CHECK (source IN ('auto','manual')),
    is_rejected INTEGER NOT NULL DEFAULT 0,       -- 1 = 使用者否決了這個標籤，自動重標不可再加回
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    PRIMARY KEY (photo_id, tag_id)
) WITHOUT ROWID;
-- 「某標籤的所有照片，依分數排序」
CREATE INDEX idx_photo_tags_tag ON photo_tags(tag_id, score DESC) WHERE is_rejected = 0;

-- ---------------------------------------------------------------------------
-- 人物與人臉
-- ---------------------------------------------------------------------------
CREATE TABLE persons (
    id             INTEGER PRIMARY KEY,
    name           TEXT,                          -- NULL = 尚未命名的群
    cover_face_id  INTEGER REFERENCES faces(id) ON DELETE SET NULL,   -- 代表頭像
    centroid       BLOB,                          -- 該群特徵的平均（再正規化），增量指派用
    face_count     INTEGER NOT NULL DEFAULT 0,
    is_hidden      INTEGER NOT NULL DEFAULT 0,    -- 路人，UI 不顯示
    created_at     INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at     INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX idx_persons_name ON persons(name) WHERE name IS NOT NULL;

CREATE TABLE faces (
    id          INTEGER PRIMARY KEY,
    photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    bbox_x      REAL    NOT NULL,                 -- 0..1
    bbox_y      REAL    NOT NULL,
    bbox_w      REAL    NOT NULL,
    bbox_h      REAL    NOT NULL,
    det_score   REAL    NOT NULL,
    landmarks   BLOB,                             -- float32 LE x 10：5 點 (x, y)，0..1
    quality     REAL,
    model_id    INTEGER REFERENCES models(id),
    embedding   BLOB,                             -- float32 LE x 512，L2 正規化；臉太小時為 NULL
    person_id   INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    cluster_id  INTEGER,                          -- 最近一次分群的原始標籤（-1 = 雜訊）
    assign_src  TEXT    CHECK (assign_src IN ('auto','manual')),
                                                  -- 'manual' = 使用者指定，重新分群不可覆蓋
    is_ignored  INTEGER NOT NULL DEFAULT 0,       -- 使用者標為「不是臉 / 不要管」
    created_at  INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX idx_faces_photo   ON faces(photo_id);
CREATE INDEX idx_faces_person  ON faces(person_id) WHERE person_id IS NOT NULL;
-- 待分群：有特徵、還沒指派
CREATE INDEX idx_faces_pending ON faces(id) WHERE person_id IS NULL AND embedding IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 匯出交易日誌（支援回復）
--   寫入協定（write-ahead）：先寫 state='pending' 並 commit，再動檔案，成功後改 'done'。
--   任何時刻當機，留下的 'pending' 列就是「可能做了一半」的操作，由 ExportManager.recover() 處理。
-- ---------------------------------------------------------------------------
CREATE TABLE export_batches (
    id               INTEGER PRIMARY KEY,
    mode             TEXT    NOT NULL CHECK (mode IN ('copy','move','hardlink','symlink')),
    dest_root        TEXT    NOT NULL,
    template         TEXT    NOT NULL,
    conflict_policy  TEXT    NOT NULL,
    state            TEXT    NOT NULL DEFAULT 'running'
                             CHECK (state IN ('running','done','cancelled','failed','rolled_back','partial_rollback')),
    total            INTEGER NOT NULL DEFAULT 0,
    started_at       INTEGER NOT NULL DEFAULT (unixepoch()),
    finished_at      INTEGER
);

CREATE TABLE export_logs (
    id              INTEGER PRIMARY KEY,
    batch_id        INTEGER NOT NULL REFERENCES export_batches(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,             -- 批次內的執行順序；回復時倒著走
    photo_id        INTEGER REFERENCES photos(id) ON DELETE SET NULL,
    action          TEXT    NOT NULL CHECK (action IN ('copy','move','hardlink','symlink')),
    src_path        TEXT    NOT NULL,
    dst_path        TEXT    NOT NULL,
    file_size       INTEGER,
    sha256          BLOB,
    state           TEXT    NOT NULL DEFAULT 'pending'
                            CHECK (state IN ('pending','done','failed','rolled_back','rollback_failed')),
    error           TEXT,
    executed_at     INTEGER,
    rolled_back_at  INTEGER
);
CREATE INDEX idx_export_logs_batch ON export_logs(batch_id, seq);
CREATE INDEX idx_export_logs_photo ON export_logs(photo_id);
CREATE INDEX idx_export_logs_open  ON export_logs(batch_id) WHERE state = 'pending';

-- ---------------------------------------------------------------------------
-- 雜項設定（UI 偏好、上次掃描的根目錄…）
-- ---------------------------------------------------------------------------
CREATE TABLE kv (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  INTEGER NOT NULL DEFAULT (unixepoch())
) WITHOUT ROWID;
