"""
RecognitionManager: Manages recognition lifecycle for tracked faces.

Strategies:
- Recognize new tracks immediately
- Recognize unknown faces periodically
- Recognize known faces periodically for freshness
- Cache results to prevent flickering
"""

import time
from typing import List, Tuple, Dict, Optional
from .track_face import TrackFace, TrackFaceList


class RecognitionManager:
    """
    Manages face recognition for tracked faces.
    
    Features:
    - Event-driven recognition (new tracks, low confidence)
    - Periodic refresh (every N seconds per track)
    - Smart caching to prevent redundant recognition
    - Label stability to prevent flickering
    
    Parameters:
    -----------
    recognizer : object
        Has .recognize(face) → (name, score)
    new_track_recognition : bool
        Immediately recognize new tracks (default: True)
    periodic_refresh_interval : float
        Seconds between re-recognition (default: 2.0)
    unknown_threshold : float
        Score below this = "Unknown" (default: 0.5)
    flickering_threshold : int
        Name changes needed to accept new label (default: 2)
    """
    
    def __init__(
        self,
        recognizer,
        new_track_recognition: bool = True,
        periodic_refresh_interval: float = 2.0,
        unknown_threshold: float = 0.5,
        flickering_threshold: int = 2
    ):
        self.recognizer = recognizer
        self.new_track_recognition = new_track_recognition
        self.periodic_refresh_interval = periodic_refresh_interval
        self.unknown_threshold = unknown_threshold
        self.flickering_threshold = flickering_threshold
        
        self.track_faces = TrackFaceList()
        self.recognition_count = 0
        self.frame_count = 0
    
    def update(
        self,
        tracker_output: List[Tuple],
        detections_map: Dict[int, object]
    ) -> List[Tuple]:
        """
        Update recognition state for all tracks.
        
        Parameters:
        -----------
        tracker_output : List[Tuple]
            From tracker: [(track_id, bbox, detection_conf), ...]
        detections_map : Dict[int, object]
            Maps track_id → face object with .embedding
        
        Returns:
        --------
        List[Tuple]
            [(track_id, bbox, name, score), ...]
        """
        self.frame_count += 1
        
        # 1. UPDATE EXISTING TRACKS + REMOVE DEAD ONES
        current_track_ids = set(t[0] for t in tracker_output)
        dead_track_ids = set(self.track_faces.tracks.keys()) - current_track_ids
        
        for track_id in dead_track_ids:
            self.track_faces.remove(track_id)
        
        # 2. ADD NEW TRACKS
        for track_id, bbox, det_conf in tracker_output:
            if track_id not in self.track_faces.tracks:
                if track_id in detections_map:
                    face_obj = detections_map[track_id]
                    track_face = TrackFace(track_id, bbox, face_obj, det_conf)
                    self.track_faces.add(track_face)
                    
                    # Immediately recognize new tracks if enabled
                    if self.new_track_recognition:
                        track_face.set_recognition_needed(True)
            else:
                # Update bbox for existing track
                track_face = self.track_faces.get(track_id)
                track_face.bbox = bbox
        
        # 3. INCREMENT AGES
        self.track_faces.increment_ages()
        
        # 4. DETERMINE WHICH TRACKS NEED RECOGNITION
        self._compute_recognition_strategy()
        
        # 5. RUN RECOGNITION ON SELECTED TRACKS
        tracks_to_recognize = self.track_faces.get_needing_recognition()
        for track_face in tracks_to_recognize:
            self._recognize_track(track_face)
        
        # 6. RETURN RESULTS
        return [t.to_output() for t in self.track_faces.get_all()]
    
    def _compute_recognition_strategy(self):
        """Determine which tracks need re-recognition."""
        for track_face in self.track_faces.get_all():
            # Skip if already marked for recognition
            if track_face.needs_recognition:
                continue
            
            # Strategy 1: New track (age == 1)
            if track_face.age <= 1:
                track_face.set_recognition_needed(True)
                continue
            
            # Strategy 2: Unknown face → refresh frequently
            if track_face.identity_score < self.unknown_threshold:
                time_since = track_face.get_time_since_recognition()
                if time_since > self.periodic_refresh_interval * 0.5:
                    track_face.set_recognition_needed(True)
                continue
            
            # Strategy 3: Known face → refresh every N seconds
            time_since = track_face.get_time_since_recognition()
            if time_since > self.periodic_refresh_interval:
                track_face.set_recognition_needed(True)
    
    def _recognize_track(self, track_face: TrackFace):
        """Run recognition on a single track."""
        try:
            if track_face.face_object is None:
                name, score = "Unknown", 0.0
            else:
                name, score = self.recognizer.recognize(track_face.face_object)
                
                # Threshold check
                if score < self.unknown_threshold:
                    name = "Unknown"
            
            # Apply stability filter (prevent flickering)
            stabilized_name = self._apply_label_stability(track_face, name, score)
            
            # Update cache
            track_face.update_identity(stabilized_name, score)
            self.recognition_count += 1
            
        except Exception as e:
            print(f"Recognition error for track {track_face.track_id}: {e}")
            track_face.update_identity("Error", 0.0)
        
        finally:
            track_face.set_recognition_needed(False)
    
    def _apply_label_stability(
        self,
        track_face: TrackFace,
        new_name: str,
        new_score: float
    ) -> str:
        """
        Prevent label flickering with hysteresis.
        
        Only accept new label if it's been consistent for N frames.
        """
        # First recognition - accept immediately
        if track_face.recognition_count == 0:
            return new_name
        
        # Same name - reset flicker counter
        if new_name == track_face.identity_name:
            track_face.name_change_count = 0
            return new_name
        
        # Different name - count changes
        track_face.name_change_count += 1
        
        # Require N consistent changes before accepting new label
        if track_face.name_change_count >= self.flickering_threshold:
            return new_name
        
        # Keep previous label (hysteresis)
        return track_face.identity_name
    
    def get_results(self) -> List[Tuple]:
        """Get current recognition results."""
        return [t.to_output() for t in self.track_faces.get_all()]
    
    def get_stats(self) -> Dict:
        """Get recognition statistics."""
        tracks = self.track_faces.get_all()
        unknown_count = sum(1 for t in tracks if t.identity_score < self.unknown_threshold)
        
        return {
            'total_tracks': len(tracks),
            'total_recognitions': self.recognition_count,
            'known_faces': len(tracks) - unknown_count,
            'unknown_faces': unknown_count,
            'frame_count': self.frame_count,
            'avg_score': sum(t.identity_score for t in tracks) / len(tracks) if tracks else 0
        }
    
    def reset(self):
        """Reset manager state."""
        self.track_faces = TrackFaceList()
        self.recognition_count = 0
        self.frame_count = 0