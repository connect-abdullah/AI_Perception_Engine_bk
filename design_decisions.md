# Database Schema Design Notes

## 1. `rooms`

Goal: Store room and floor information.


| Field                       | Why it is used                        |
| --------------------------- | ------------------------------------- |
| `id` (UUID, PK)             | Uniquely identifies each room.        |
| `name` (VARCHAR 120)        | Stores the room name.                 |
| `floor` (INTEGER, nullable) | Stores the floor number if available. |


## 2. `cameras`

Goal: Store camera information and its room association.


| Field                              | Why it is used                                                        |
| ---------------------------------- | --------------------------------------------------------------------- |
| `id` (UUID, PK)                    | Uniquely identifies each camera.                                      |
| `camera_code` (VARCHAR 50, unique) | Provides a unique, readable camera identifier.                        |
| `room_id` (UUID, FK, nullable)     | Connects the camera to its room. Nullable if its location is unknown. |




## 3. `videos`

Goal: Store metadata about uploaded or prerecorded videos.


| Field                               | Why it is used                                                                                        |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `id` (UUID, PK)                     | Uniquely identifies each video.                                                                       |
| `camera_id` (UUID, FK)              | Identifies which camera recorded the video.                                                           |
| `file_path` (TEXT)                  | Stores the video's file location.                                                                     |
| `video_start_time` (TIMESTAMPTZ)    | Records when the video footage begins.                                                                |
| `fps` (REAL, nullable)              | Stores frames per second.                                                                             |
| `width` (INTEGER, nullable)         | Stores the video's frame width in pixels.                                                             |
| `height` (INTEGER, nullable)        | Stores the video's frame height in pixels.                                                            |
| `duration_seconds` (REAL, nullable) | Stores the video's duration in seconds.                                                               |
| `status` (VARCHAR 20)               | Tracks processing status: `uploaded`, `processing`, `completed`, or `failed`. Defaults to `uploaded`. |
| `created_at` (TIMESTAMPTZ)          | Records when the video record was created. Defaults to the current timestamp.                         |




## 4. `persons`

Goal: Store global, anonymous person identities across multiple tracking sessions.


| Field                                | Why it is used                                   |
| ------------------------------------ | ------------------------------------------------ |
| `id` (UUID, PK)                      | Uniquely identifies each person in the database. |
| `person_code` (VARCHAR 20, unique)   | Provides a readable identifier, such as `P001`.  |
| `known_name` (VARCHAR 120, nullable) | Stores a person's name if known.                 |
| `created_at` (TIMESTAMPTZ)           | Records when the person record was created.      |


A global person can be associated with multiple local tracks when ReID matches them with sufficient confidence.

## 5. `tracks`

Goal: Store local tracking sessions produced by the tracking algorithm.


| Field                              | Why it is used                                                      |
| ---------------------------------- | ------------------------------------------------------------------- |
| `id` (UUID, PK)                    | Uniquely identifies each tracking session.                          |
| `video_id` (UUID, FK)              | Identifies the video containing the track.                          |
| `local_track_id` (VARCHAR 80)      | Stores the tracking algorithm's local ID.                           |
| `person_id` (UUID, FK, nullable)   | Connects the local track to a global person when identity is known. |
| `start_time` (TIMESTAMPTZ)         | Records when the track begins.                                      |
| `end_time` (TIMESTAMPTZ, nullable) | Records when the track ends, if known.                              |


Constraint: `(video_id, local_track_id)` must be unique because a local track ID is only unique within its video.

`person_id` is nullable so uncertain identity matches remain unresolved.

## 6. `observations`

Goal: Store individual timestamped detections of a tracked person.


| Field                                  | Why it is used                                                  |
| -------------------------------------- | --------------------------------------------------------------- |
| `id` (BIGINT, identity PK)             | Provides a unique, automatically generated observation ID.      |
| `track_id` (UUID, FK)                  | Identifies the tracking session this observation belongs to.    |
| `frame_number` (INTEGER)               | Records the video frame where the detection occurred.           |
| `timestamp` (TIMESTAMPTZ)              | Records the exact time of the observation.                      |
| `bbox_x` (INTEGER)                     | Stores the bounding box's horizontal position.                  |
| `bbox_y` (INTEGER)                     | Stores the bounding box's vertical position.                    |
| `bbox_width` (INTEGER)                 | Stores the bounding box's width in pixels.                      |
| `bbox_height` (INTEGER)                | Stores the bounding box's height in pixels.                     |
| `detection_confidence` (REAL)          | Stores the detector's confidence score between 0 and 1.         |
| `activity` (VARCHAR 30, nullable)      | Stores the detected activity at that moment, if available.      |
| `activity_confidence` (REAL, nullable) | Stores confidence in the activity classification, if available. |


Why BIGINT instead of UUID?

- Observations are expected to be the highest-volume table because a track can produce many observations.
- BIGINT uses 8 bytes, while UUID uses 16 bytes, reducing primary-key storage and index size.
- Automatically generated sequential IDs generally work well for frequent inserts.
- No other table needs to reference an individual observation, so a UUID provides little additional benefit here.

Constraint: `(track_id, frame_number)` must be unique so the same track cannot have duplicate observations for one frame.

Frame numbers must be non-negative, bounding-box dimensions positive, and confidence values between 0 and 1. Each observation represents one moment, not an entire activity period.

## 7. `state_intervals`

Goal: Store activities over continuous time ranges.


| Field                              | Why it is used                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| `id` (UUID, PK)                    | Uniquely identifies each activity interval.                                         |
| `track_id` (UUID, FK)              | Identifies the track performing the activity.                                       |
| `activity` (VARCHAR 30)            | Stores an activity such as `standing`, `sitting`, `walking`, `running`, or `lying`. |
| `start_time` (TIMESTAMPTZ)         | Records when the activity begins.                                                   |
| `end_time` (TIMESTAMPTZ, nullable) | Records when the activity ends, if known.                                           |
| `confidence` (REAL, nullable)      | Stores confidence in the activity classification.                                   |


Interval boundaries must be valid, and confidence values must be between 0 and 1.

Observations represent individual moments; state intervals summarize activities over time.

## 8. `events`

Goal: Store meaningful security-related occurrences.


| Field                           | Why it is used                                                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `id` (UUID, PK)                 | Uniquely identifies each event.                                                                                       |
| `track_id` (UUID, FK, nullable) | Connects the event to a track when known.                                                                             |
| `event_type` (VARCHAR 50)       | Identifies the event, such as `entered_building`, `exited_building`, `entered_room`, `left_room`, or `floor_changed`. |
| `timestamp` (TIMESTAMPTZ)       | Records when the event occurred.                                                                                      |
| `room_id` (UUID, FK, nullable)  | Identifies the related room, if applicable.                                                                           |
| `confidence` (REAL, nullable)   | Stores confidence in the event detection.                                                                             |
| `metadata` (JSONB)              | Stores additional event-specific details that do not need dedicated columns. Defaults to an empty JSON object.        |


Confidence values must be between 0 and 1. Use JSONB only for additional event-specific metadata.

## Relationships

- One room can have multiple cameras and events.
- One camera can have multiple videos.
- One video can have multiple tracks.
- One global person can be associated with multiple local tracks.
- One track can have multiple observations, state intervals, and events.
- One room can be associated with multiple events.



## Constraints and Indexes

- Unique camera codes and person codes.
- Unique local track IDs per video.
- Unique frame numbers per track.
- Confidence values between 0 and 1.
- Positive bounding-box dimensions and non-negative frame numbers.
- Valid video metadata and activity interval boundaries.
- Index frequently queried foreign keys and timestamp columns.
- Use composite indexes for person-history, event-history, and time-range queries where appropriate.
- Avoid unnecessary indexes and duplicated data.



## Why BigINT in Observations.id instead of UUID?

The main idea is simple: observations are just records of detections, so they don't need the same kind of ID as a person or a video.

That person can be referenced by many tracks and other records.
UUIDs are useful because they provide a unique identifier that can be generated before inserting the record into the database.

Suppose our video pipeline detects a person in three frames.
observations


| id  | track_id | frame_number | timestamp |
| --- | -------- | ------------ | --------- |
| 1   | TRACK_7  | 100          | 18:00:01  |
| 2   | TRACK_7  | 105          | 18:00:02  |
| 3   | TRACK_7  | 110          | 18:00:03  |


Each row represents one detection snapshot.
We already know which observation it is from:
track_id + frame_number

For example:
TRACK_7 + Frame 100

This combination uniquely identifies that observation, provided we enforce a unique constraint on those two fields.

So, a simple sequential integer is sufficient for the primary key.

### Why is an integer better here?



### Smaller index

An integer ID uses 8 bytes with BIGINT, whereas a UUID uses 16 bytes. With millions of observations, that difference adds up.