"""
TrackFace: Wraps tracked face data with recognition state and caching.

Maintains:
- Track ID and bbox
- Face object for re-recognition
- Cached identity (name, score)
- Refresh strategy (when to re-run recognition)
"""

import time
from typing import Tuple, Optional


class TrackFace:
    """Represents a tracked face with recognition state."""
    
    def __init__(
        self,
        track_id: int,
        bbox: Tuple,
        face_object,
        confidence: float = 1.0
    ):
        """
        Initialize TrackFace.
        
        Parameters:
        -----------
        track_id : int
            Unique ID from tracker
        bbox : Tuple
            (x1, y1, x2, y2) bounding box
        face_object : insightface.Face-like object
            Contains .embedding, .bbox, .kps for recognition/drawing
        confidence : float
            Detection confidence (0-1)
        """
        self.track_id = track_id
        self.bbox = bbox
        self.face_object = face_object
        self.detection_confidence = confidence
        
        # Recognition state
        self.identity_name = "Unknown"
        self.identity_score = 0.0
        self.last_recognition_time = 0.0
        self.recognition_count = 0  # Times this track was recognized
        
        # Refresh strategy
        self.created_time = time.time()
        self.age = 0  # Incremented by manager each frame
        self.needs_recognition = True  # Always recognize on first frame
        
        # Label stability (prevent flickering)
        self.prev_name = None
        self.name_change_count = 0
    
    def update_identity(
        self,
        name: str,
        score: float,
        timestamp: Optional[float] = None
    ):
        """
        Update cached recognition result.
        
        Parameters:
        -----------
        name : str
            Recognized person name or "Unknown"
        score : float
            Similarity/confidence score (0-1)
        timestamp : float, optional
            Recognition timestamp (default: now)
        """
        # Track name changes for flicker detection
        if name != self.identity_name:
            self.name_change_count += 1
        else:
            self.name_change_count = 0
        
        self.prev_name = self.identity_name
        self.identity_name = name
        self.identity_score = score
        self.last_recognition_time = timestamp or time.time()
        self.recognition_count += 1
    
    def set_recognition_needed(self, needed: bool = True):
        """Mark track for recognition on next cycle."""
        self.needs_recognition = needed
    
    def get_lifetime(self) -> float:
        """Get track lifetime in seconds."""
        return time.time() - self.created_time
    
    def get_time_since_recognition(self) -> float:
        """Get seconds since last recognition."""
        return time.time() - self.last_recognition_time
    
    def to_output(self) -> Tuple:
        """Return output tuple: (track_id, bbox, name, score)."""
        return (self.track_id, self.bbox, self.identity_name, self.identity_score)


class TrackFaceList:
    """Manages collection of TrackFace objects."""
    
    def __init__(self):
        self.tracks = {}  # {track_id: TrackFace}
    
    def add(self, track_face: TrackFace):
        """Add TrackFace."""
        self.tracks[track_face.track_id] = track_face
    
    def remove(self, track_id: int):
        """Remove TrackFace by ID."""
        if track_id in self.tracks:
            del self.tracks[track_id]
    
    def get(self, track_id: int) -> Optional[TrackFace]:
        """Get TrackFace by ID."""
        return self.tracks.get(track_id)
    
    def get_all(self) -> list:
        """Get all TrackFaces."""
        return list(self.tracks.values())
    
    def get_needing_recognition(self) -> list:
        """Get tracks that need recognition."""
        return [t for t in self.tracks.values() if t.needs_recognition]
    
    def increment_ages(self):
        """Increment age counter for all tracks."""
        for track in self.tracks.values():
            track.age += 1