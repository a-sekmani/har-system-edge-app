"""
HAR-System: Face Identity Manager
==================================
Manages face identities and their association with track IDs
"""

import time
from typing import Dict, Optional, Any
from collections import defaultdict


class FaceIdentityManager:
    """
    Manages the mapping between track IDs and person identities
    
    This class handles:
    - Associating track IDs with person names
    - Tracking confidence levels
    - Managing identity confirmations
    - Handling identity timeouts
    """
    
    def __init__(self, min_confirmations: int = 2, identity_timeout: float = 5.0):
        """
        Initialize Face Identity Manager
        
        Args:
            min_confirmations: Minimum number of confirmations before identity is trusted
            identity_timeout: Seconds before an identity needs re-confirmation
        """
        self.min_confirmations = min_confirmations
        self.identity_timeout = identity_timeout
        
        # Track identities: {track_id: identity_info}
        self.track_identities = {}
        
        # Identity candidates (for confirmation): {track_id: [candidate1, candidate2, ...]}
        self.identity_candidates = defaultdict(list)
        
    def update_identity(self, track_id: int, name: str, confidence: float, 
                       global_id: Optional[str] = None) -> bool:
        """
        Update or set identity for a track
        
        Args:
            track_id: Track ID from pose estimation
            name: Person name (or "Unknown")
            confidence: Recognition confidence (0.0 to 1.0)
            global_id: Database global ID (optional)
        
        Returns:
            True if identity was updated/confirmed, False otherwise
        """
        current_time = time.time()
        
        # If name is Unknown, just update and return
        if name == "Unknown":
            if track_id not in self.track_identities:
                self.track_identities[track_id] = {
                    'name': 'Unknown',
                    'global_id': None,
                    'confidence': 0.0,
                    'first_identified': current_time,
                    'last_confirmed': current_time,
                    'confirmation_count': 0,
                    'is_confirmed': False
                }
            return False
        
        # Add to candidates
        candidate = {
            'name': name,
            'global_id': global_id,
            'confidence': confidence,
            'timestamp': current_time
        }
        self.identity_candidates[track_id].append(candidate)
        
        # Keep only recent candidates (last 5 seconds)
        self.identity_candidates[track_id] = [
            c for c in self.identity_candidates[track_id]
            if current_time - c['timestamp'] < 5.0
        ]
        
        # Check if we have enough confirmations
        if len(self.identity_candidates[track_id]) >= self.min_confirmations:
            # Get most common name from candidates
            name_counts = {}
            for cand in self.identity_candidates[track_id]:
                cname = cand['name']
                name_counts[cname] = name_counts.get(cname, 0) + 1
            
            # Find name with most confirmations
            best_name = max(name_counts.items(), key=lambda x: x[1])
            
            # If we have enough confirmations for this name
            if best_name[1] >= self.min_confirmations:
                # Get average confidence for this name
                name_confidences = [
                    c['confidence'] for c in self.identity_candidates[track_id]
                    if c['name'] == best_name[0]
                ]
                avg_confidence = sum(name_confidences) / len(name_confidences)
                
                # Get global_id for this name
                name_global_id = None
                for c in self.identity_candidates[track_id]:
                    if c['name'] == best_name[0] and c['global_id']:
                        name_global_id = c['global_id']
                        break
                
                # Update or create identity
                if track_id not in self.track_identities:
                    self.track_identities[track_id] = {
                        'name': best_name[0],
                        'global_id': name_global_id,
                        'confidence': avg_confidence,
                        'first_identified': current_time,
                        'last_confirmed': current_time,
                        'confirmation_count': best_name[1],
                        'is_confirmed': True
                    }
                    print(f"[IDENTITY] Track #{track_id} identified as: {best_name[0]} (confidence: {avg_confidence:.2f})")
                    return True
                else:
                    # Update existing identity
                    old_name = self.track_identities[track_id]['name']
                    self.track_identities[track_id].update({
                        'name': best_name[0],
                        'global_id': name_global_id,
                        'confidence': avg_confidence,
                        'last_confirmed': current_time,
                        'confirmation_count': self.track_identities[track_id]['confirmation_count'] + 1,
                        'is_confirmed': True
                    })
                    
                    if old_name != best_name[0]:
                        print(f"[IDENTITY] Track #{track_id} identity changed: {old_name} → {best_name[0]}")
                    
                    return True
        
        return False
    
    def get_identity(self, track_id: int) -> str:
        """
        Get the name associated with a track ID
        
        Args:
            track_id: Track ID
        
        Returns:
            Person name or "Unknown"
        """
        if track_id not in self.track_identities:
            return "Unknown"
        
        identity = self.track_identities[track_id]
        
        # Check if identity needs re-confirmation
        if identity['is_confirmed']:
            time_since_confirmation = time.time() - identity['last_confirmed']
            if time_since_confirmation > self.identity_timeout:
                # Mark as needing re-confirmation but keep the name
                identity['is_confirmed'] = False
        
        return identity['name']
    
    def get_confidence(self, track_id: int) -> float:
        """
        Get the confidence level for a track's identity
        
        Args:
            track_id: Track ID
        
        Returns:
            Confidence level (0.0 to 1.0) or 0.0 if unknown
        """
        if track_id not in self.track_identities:
            return 0.0
        
        return self.track_identities[track_id].get('confidence', 0.0)
    
    def get_identity_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get full identity information for a track
        
        Args:
            track_id: Track ID
        
        Returns:
            Dictionary with identity info or None
        """
        if track_id not in self.track_identities:
            return None
        
        return self.track_identities[track_id].copy()
    
    def is_identified(self, track_id: int) -> bool:
        """
        Check if a track has a confirmed identity
        
        Args:
            track_id: Track ID
        
        Returns:
            True if identity is confirmed and not "Unknown"
        """
        if track_id not in self.track_identities:
            return False
        
        identity = self.track_identities[track_id]
        return identity['is_confirmed'] and identity['name'] != "Unknown"
    
    def needs_recognition(self, track_id: int) -> bool:
        """
        Check if a track needs face recognition
        
        Args:
            track_id: Track ID
        
        Returns:
            True if recognition is needed
        """
        # New track - needs recognition
        if track_id not in self.track_identities:
            return True
        
        identity = self.track_identities[track_id]
        
        # Unknown identity - needs recognition
        if identity['name'] == "Unknown":
            return True
        
        # Not confirmed - needs recognition
        if not identity['is_confirmed']:
            return True
        
        # Confirmed but timeout expired - needs re-confirmation
        time_since_confirmation = time.time() - identity['last_confirmed']
        if time_since_confirmation > self.identity_timeout:
            return True
        
        return False
    
    def remove_track(self, track_id: int):
        """
        Remove a track from identity management
        
        Args:
            track_id: Track ID to remove
        """
        if track_id in self.track_identities:
            del self.track_identities[track_id]
        
        if track_id in self.identity_candidates:
            del self.identity_candidates[track_id]
    
    def get_all_identities(self) -> Dict[int, str]:
        """
        Get all track identities
        
        Returns:
            Dictionary mapping track_id to name
        """
        return {
            track_id: info['name']
            for track_id, info in self.track_identities.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about identities
        
        Returns:
            Dictionary with statistics
        """
        total_tracks = len(self.track_identities)
        identified = sum(1 for info in self.track_identities.values() 
                        if info['name'] != "Unknown" and info['is_confirmed'])
        unknown = total_tracks - identified
        
        # Get unique names
        unique_names = set()
        for info in self.track_identities.values():
            if info['name'] != "Unknown":
                unique_names.add(info['name'])
        
        return {
            'total_tracks': total_tracks,
            'identified_tracks': identified,
            'unknown_tracks': unknown,
            'unique_persons': len(unique_names),
            'person_names': sorted(list(unique_names))
        }
    
    def reset(self):
        """Reset all identities (useful for processing new video)"""
        self.track_identities = {}
        self.identity_candidates = defaultdict(list)
