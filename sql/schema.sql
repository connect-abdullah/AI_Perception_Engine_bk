-- =============================================================================
--  AI Perception Engine – PostgreSQL Schema DDL
--  Apply to a fresh Supabase (or local Postgres 14+) database.
--
--  Execution order honours FK dependencies:
--    rooms → cameras → videos → persons → tracks
--    tracks → observations, state_intervals, events
--
--  Run once with:
--      psql "$DIRECT_URL" -f sql/schema.sql
--
--  After this, Alembic tracks further changes.
-- =============================================================================

-- gen_random_uuid() is provided by pgcrypto (enabled by default on Supabase).
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ─── rooms ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rooms (
    id    UUID        NOT NULL DEFAULT gen_random_uuid(),
    name  VARCHAR(120) NOT NULL,
    floor INTEGER,

    CONSTRAINT pk_rooms PRIMARY KEY (id)
);

-- ─── cameras ─────────────────────────────────────────────────────────────────
-- room_id is nullable: a camera can be registered before its room is known.
-- ON DELETE SET NULL preserves camera records when a room is removed.
CREATE TABLE IF NOT EXISTS cameras (
    id          UUID        NOT NULL DEFAULT gen_random_uuid(),
    camera_code VARCHAR(50) NOT NULL,
    room_id     UUID,

    CONSTRAINT pk_cameras         PRIMARY KEY (id),
    CONSTRAINT uq_cameras_code    UNIQUE      (camera_code),
    CONSTRAINT fk_cameras_room    FOREIGN KEY (room_id)
                                      REFERENCES rooms (id)
                                      ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS ix_cameras_room_id ON cameras (room_id);

-- ─── videos ──────────────────────────────────────────────────────────────────
-- ON DELETE RESTRICT: a camera cannot be deleted while video records exist.
-- source_video_id: stable string key from the perception handoff (e.g.
--   "sha256:<hex>") that matches manifest.json / observations.jsonl.
-- source: 'cv' for real computer-vision output, 'simulated' for synthetic.
CREATE TABLE IF NOT EXISTS videos (
    id               UUID         NOT NULL DEFAULT gen_random_uuid(),
    camera_id        UUID         NOT NULL,
    file_path        TEXT         NOT NULL,
    video_start_time TIMESTAMPTZ  NOT NULL,
    fps              REAL,
    width            INTEGER,
    height           INTEGER,
    duration_seconds REAL,
    status           VARCHAR(20)  NOT NULL DEFAULT 'uploaded',
    source           VARCHAR(20)  NOT NULL DEFAULT 'cv',
    source_video_id  TEXT,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT pk_videos              PRIMARY KEY (id),
    CONSTRAINT uq_videos_source_video_id UNIQUE (source_video_id),
    CONSTRAINT fk_videos_camera       FOREIGN KEY (camera_id)
                                          REFERENCES cameras (id)
                                          ON DELETE RESTRICT,

    CONSTRAINT ck_videos_status       CHECK (status IN
                                          ('uploaded','processing','completed','failed')),
    CONSTRAINT ck_videos_source       CHECK (source IN ('cv','simulated')),
    CONSTRAINT ck_videos_fps          CHECK (fps IS NULL OR fps > 0),
    CONSTRAINT ck_videos_width        CHECK (width IS NULL OR width > 0),
    CONSTRAINT ck_videos_height       CHECK (height IS NULL OR height > 0),
    CONSTRAINT ck_videos_duration     CHECK (duration_seconds IS NULL
                                            OR duration_seconds >= 0)
);

CREATE INDEX IF NOT EXISTS ix_videos_camera_id      ON videos (camera_id);
CREATE INDEX IF NOT EXISTS ix_videos_start_time     ON videos (video_start_time);
CREATE INDEX IF NOT EXISTS ix_videos_source_video_id ON videos (source_video_id);

-- ─── persons ─────────────────────────────────────────────────────────────────
-- Global anonymous identity; person_code is the human-readable key (P001 …).
CREATE TABLE IF NOT EXISTS persons (
    id          UUID         NOT NULL DEFAULT gen_random_uuid(),
    person_code VARCHAR(20)  NOT NULL,
    known_name  VARCHAR(120),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT pk_persons         PRIMARY KEY (id),
    CONSTRAINT uq_persons_code    UNIQUE      (person_code)
);

-- ─── tracks ──────────────────────────────────────────────────────────────────
-- local_track_id is unique per video (composite key).
-- person_id is nullable: unresolved ReID matches stay unlinked.
-- ON DELETE SET NULL preserves tracks when a person record is deleted.
-- identity_* fields: provisional; open strings, no CHECK on status.
CREATE TABLE IF NOT EXISTS tracks (
    id                   UUID        NOT NULL DEFAULT gen_random_uuid(),
    video_id             UUID        NOT NULL,
    local_track_id       VARCHAR(80) NOT NULL,
    person_id            UUID,
    start_time           TIMESTAMPTZ NOT NULL,
    end_time             TIMESTAMPTZ,
    identity_status      VARCHAR(30),
    identity_confidence  REAL,
    identity_meta        JSONB       NOT NULL DEFAULT '{}',

    CONSTRAINT pk_tracks              PRIMARY KEY (id),
    CONSTRAINT uq_tracks_video_local  UNIQUE      (video_id, local_track_id),
    CONSTRAINT fk_tracks_video        FOREIGN KEY (video_id)
                                          REFERENCES videos (id)
                                          ON DELETE CASCADE,
    CONSTRAINT fk_tracks_person       FOREIGN KEY (person_id)
                                          REFERENCES persons (id)
                                          ON DELETE SET NULL,
    CONSTRAINT ck_tracks_end_after_start
                                      CHECK (end_time IS NULL
                                             OR end_time >= start_time),
    CONSTRAINT ck_tracks_identity_confidence
                                      CHECK (identity_confidence IS NULL
                                             OR identity_confidence BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS ix_tracks_video_id       ON tracks (video_id);
CREATE INDEX IF NOT EXISTS ix_tracks_person_start   ON tracks (person_id, start_time);

-- ─── observations ────────────────────────────────────────────────────────────
-- High-volume, write-once rows; BIGINT IDENTITY is more compact than UUID.
-- (track_id, frame_number) is unique: one detection snapshot per frame.
-- world_x/y/z and room_id are optional spatial fields.
-- crop_path: relative path of the crop PNG saved by the perception producer.
CREATE TABLE IF NOT EXISTS observations (
    id                   BIGINT      GENERATED ALWAYS AS IDENTITY,
    track_id             UUID        NOT NULL,
    frame_number         INTEGER     NOT NULL,
    timestamp            TIMESTAMPTZ NOT NULL,
    bbox_x               INTEGER     NOT NULL,
    bbox_y               INTEGER     NOT NULL,
    bbox_width           INTEGER     NOT NULL,
    bbox_height          INTEGER     NOT NULL,
    detection_confidence REAL        NOT NULL,
    activity             VARCHAR(30),
    activity_confidence  REAL,
    world_x              REAL,
    world_y              REAL,
    world_z              REAL,
    room_id              UUID,
    crop_path            TEXT,

    CONSTRAINT pk_observations             PRIMARY KEY (id),
    CONSTRAINT uq_observations_track_frame UNIQUE (track_id, frame_number),
    CONSTRAINT fk_observations_track       FOREIGN KEY (track_id)
                                               REFERENCES tracks (id)
                                               ON DELETE CASCADE,
    CONSTRAINT fk_observations_room        FOREIGN KEY (room_id)
                                               REFERENCES rooms (id)
                                               ON DELETE SET NULL,
    CONSTRAINT ck_obs_frame_non_negative   CHECK (frame_number >= 0),
    CONSTRAINT ck_obs_bbox_width           CHECK (bbox_width  > 0),
    CONSTRAINT ck_obs_bbox_height          CHECK (bbox_height > 0),
    CONSTRAINT ck_obs_det_confidence       CHECK (detection_confidence  BETWEEN 0.0 AND 1.0),
    CONSTRAINT ck_obs_act_confidence       CHECK (activity_confidence IS NULL
                                                  OR activity_confidence BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS ix_observations_track_ts
    ON observations (track_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_observations_room_ts
    ON observations (room_id, timestamp);

-- ─── state_intervals ─────────────────────────────────────────────────────────
-- Summarises a continuous activity window for a track.
CREATE TABLE IF NOT EXISTS state_intervals (
    id         UUID        NOT NULL DEFAULT gen_random_uuid(),
    track_id   UUID        NOT NULL,
    activity   VARCHAR(30) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time   TIMESTAMPTZ,
    confidence REAL,

    CONSTRAINT pk_state_intervals   PRIMARY KEY (id),
    CONSTRAINT fk_si_track          FOREIGN KEY (track_id)
                                        REFERENCES tracks (id)
                                        ON DELETE CASCADE,
    CONSTRAINT ck_si_end_after_start CHECK (end_time IS NULL
                                            OR end_time >= start_time),
    CONSTRAINT ck_si_confidence      CHECK (confidence IS NULL
                                           OR confidence BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS ix_si_track_id    ON state_intervals (track_id);
CREATE INDEX IF NOT EXISTS ix_si_track_start ON state_intervals (track_id, start_time);

-- ─── events ──────────────────────────────────────────────────────────────────
-- Discrete security-relevant facts.
-- track_id and room_id are both nullable (see model comments).
-- metadata JSONB stores event-specific extras; never NULL.
CREATE TABLE IF NOT EXISTS events (
    id         UUID        NOT NULL DEFAULT gen_random_uuid(),
    track_id   UUID,
    event_type VARCHAR(50) NOT NULL,
    timestamp  TIMESTAMPTZ NOT NULL,
    room_id    UUID,
    confidence REAL,
    metadata   JSONB       NOT NULL DEFAULT '{}',

    CONSTRAINT pk_events           PRIMARY KEY (id),
    CONSTRAINT fk_events_track     FOREIGN KEY (track_id)
                                       REFERENCES tracks (id)
                                       ON DELETE SET NULL,
    CONSTRAINT fk_events_room      FOREIGN KEY (room_id)
                                       REFERENCES rooms (id)
                                       ON DELETE SET NULL,
    CONSTRAINT ck_events_confidence CHECK (confidence IS NULL
                                           OR confidence BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS ix_events_room_ts  ON events (room_id,    timestamp);
CREATE INDEX IF NOT EXISTS ix_events_track_id ON events (track_id);
CREATE INDEX IF NOT EXISTS ix_events_type_ts  ON events (event_type, timestamp);
