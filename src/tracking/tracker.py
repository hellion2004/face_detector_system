"""
Lightweight centroid-based face tracker for real-time pipelines.
Suitable for CPU/Raspberry Pi deployment.

Usage:
    tracker = Tracker(max_age=30, distance_threshold=50)
    tracked_faces = tracker.update(detections)  # detections = [(x1,y1,x2,y2), ...]
    # Returns: [(track_id, (x1,y1,x2,y2), confidence), ...]
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple


class Track:
    """Represents a single tracked face."""
    
    def __init__(self, track_id: int, bbox: Tuple, confidence: float = 1.0):
        self.track_id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.centroid = self._compute_centroid(bbox)
        self.age = 1  # Frames since creation
        self.hits = 1  # Successful consecutive matches
        self.confidence = confidence
        self.prediction = None  # Optional: for motion prediction
    
    def _compute_centroid(self, bbox: Tuple) -> Tuple[float, float]:
        """Compute centroid from bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)
    
    def update(self, bbox: Tuple, confidence: float = 1.0):
        """Update track with new detection."""
        self.bbox = bbox
        self.centroid = self._compute_centroid(bbox)
        self.hits += 1
        self.age += 1
        self.confidence = confidence
    
    def increment_age(self):
        """Increment age when no match found."""
        self.age += 1


class Tracker:
    """
    Centroid-based multi-object tracker.
    
    Parameters:
    -----------
    max_age : int
        Maximum frames to keep inactive track before deletion
    distance_threshold : float
        Maximum centroid distance for matching (pixels)
    iou_threshold : float
        Minimum IoU for fallback matching (0-1)
    min_hits : int
        Minimum hits before track becomes "confirmed"
    """
    
    def __init__(
        self,
        max_age: int = 30,
        distance_threshold: float = 50.0,
        iou_threshold: float = 0.3,
        min_hits: int = 1
    ):
        self.max_age = max_age
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        
        self.tracks = {}  # {track_id: Track}
        self.next_id = 1
        self.frame_count = 0
    
    def update(
        self,
        detections: List[Tuple]
    ) -> List[Tuple]:
        """
        Update tracker with new detections.
        
        Parameters:
        -----------
        detections : List[Tuple]
            List of (x1, y1, x2, y2) bounding boxes or 
            List of (x1, y1, x2, y2, confidence) if confidence included
        
        Returns:
        --------
        List[Tuple]
            List of (track_id, bbox, confidence) for confirmed tracks
        """
        self.frame_count += 1
        
        # Parse detections
        bboxes = []
        confidences = []
        for det in detections:
            if len(det) == 5:  # bbox + confidence
                bbox, conf = det[:4], det[4]
            else:  # bbox only
                bbox, conf = tuple(det), 1.0
            bboxes.append(bbox)
            confidences.append(conf)
        
        # Match detections to existing tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._match_detections(
            bboxes, confidences
        )
        
        # Update matched tracks
        for track_id, det_idx in matched_tracks:
            self.tracks[track_id].update(bboxes[det_idx], confidences[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(bboxes[det_idx], confidences[det_idx])
        
        # Mark unmatched tracks as inactive
        for track_id in unmatched_tracks:
            self.tracks[track_id].increment_age()
        
        # Remove dead tracks
        self._cleanup_tracks()
        
        # Return confirmed tracks
        return self._get_output()
    
    def _match_detections(
        self,
        bboxes: List[Tuple],
        confidences: List[float]
    ) -> Tuple[List, List, List]:
        """
        Match detections to existing tracks.
        
        Returns:
        --------
        matched_tracks : List[(track_id, det_idx)]
        unmatched_detections : List[det_idx]
        unmatched_tracks : List[track_id]
        """
        if len(self.tracks) == 0:
            return [], list(range(len(bboxes))), []
        
        # Compute centroid distances
        matched = []
        unmatched_det = set(range(len(bboxes)))
        unmatched_trk = set(self.tracks.keys())
        
        track_ids = list(self.tracks.keys())
        
        # Primary: Centroid distance matching
        distances = self._compute_centroid_distances(bboxes, track_ids)
        
        for trk_idx, track_id in enumerate(track_ids):
            best_det_idx = np.argmin(distances[trk_idx])
            best_distance = distances[trk_idx][best_det_idx]
            
            if best_distance < self.distance_threshold:
                matched.append((track_id, best_det_idx))
                unmatched_det.discard(best_det_idx)
                unmatched_trk.discard(track_id)
        
        # Secondary: IoU matching for remaining tracks/detections
        remaining_trk = list(unmatched_trk)
        remaining_det = list(unmatched_det)
        
        if len(remaining_trk) > 0 and len(remaining_det) > 0:
            ious = self._compute_iou_matrix(bboxes, remaining_trk, remaining_det)
            
            for i, trk_idx in enumerate(remaining_trk):
                if len(ious[i]) > 0:
                    best_det_idx_in_remaining = np.argmax(ious[i])
                    best_iou = ious[i][best_det_idx_in_remaining]
                    
                    if best_iou > self.iou_threshold:
                        best_det_idx = remaining_det[best_det_idx_in_remaining]
                        matched.append((trk_idx, best_det_idx))
                        unmatched_det.discard(best_det_idx)
                        unmatched_trk.discard(trk_idx)
        
        return matched, list(unmatched_det), list(unmatched_trk)
    
    def _compute_centroid_distances(
        self,
        bboxes: List[Tuple],
        track_ids: List[int]
    ) -> np.ndarray:
        """Compute Euclidean distances between detection and track centroids."""
        distances = np.zeros((len(track_ids), len(bboxes)))
        
        for i, track_id in enumerate(track_ids):
            track_centroid = self.tracks[track_id].centroid
            for j, bbox in enumerate(bboxes):
                det_centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                distances[i, j] = np.linalg.norm(
                    np.array(track_centroid) - np.array(det_centroid)
                )
        
        return distances
    
    def _compute_iou_matrix(
        self,
        bboxes: List[Tuple],
        track_ids: List[int],
        det_indices: List[int]
    ) -> np.ndarray:
        """Compute IoU between tracks and detections."""
        ious = np.zeros((len(track_ids), len(det_indices)))
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id].bbox
            for j, det_idx in enumerate(det_indices):
                det_bbox = bboxes[det_idx]
                ious[i, j] = self._iou(track_bbox, det_bbox)
        
        return ious
    
    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _create_track(self, bbox: Tuple, confidence: float = 1.0):
        """Create new track."""
        self.tracks[self.next_id] = Track(self.next_id, bbox, confidence)
        self.next_id += 1
    
    def _cleanup_tracks(self):
        """Remove tracks that exceeded max_age."""
        dead_tracks = [
            tid for tid, track in self.tracks.items()
            if track.age > self.max_age
        ]
        for tid in dead_tracks:
            del self.tracks[tid]
    
    def _get_output(self) -> List[Tuple]:
        """Return confirmed tracks only."""
        output = []
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits:
                output.append((track_id, track.bbox, track.confidence))
        return output
    
    def get_all_tracks(self) -> List[Tuple]:
        """Return all tracks (including unconfirmed)."""
        return [(t.track_id, t.bbox, t.confidence) for t in self.tracks.values()]
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0