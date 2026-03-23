import numpy as np


class CentroidTracker:
    def __init__(self, max_distance=3.0):
        self.max_distance = max_distance
        self.next_track_id = 1
        self.tracks = {}

    def update(self, object_features):
        updated_tracks = {}
        assignments = []

        for feature in object_features:
            centroid = np.asarray(feature["centroid"], dtype=np.float32)
            best_track_id = None
            best_distance = None

            for track_id, track in self.tracks.items():
                distance = np.linalg.norm(track["centroid"] - centroid)
                if distance <= self.max_distance and (best_distance is None or distance < best_distance):
                    best_track_id = track_id
                    best_distance = distance

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1

            track_record = {
                "track_id": best_track_id,
                "centroid": centroid,
                "semantic_label": feature.get("semantic_label", "unknown"),
            }
            updated_tracks[best_track_id] = track_record
            enriched = dict(feature)
            enriched["track_id"] = best_track_id
            assignments.append(enriched)

        self.tracks = updated_tracks
        return assignments
