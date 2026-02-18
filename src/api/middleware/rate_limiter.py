"""In-memory rate limiting middleware for ChromaGuide API.

Implements sliding window rate limiting using timestamps.
Tracks requests per client (IP address) with configurable limits.
"""

import time
from typing import Dict, List
from collections import defaultdict
from threading import Lock


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RateLimiter:
    """In-memory sliding window rate limiter.
    
    Tracks request timestamps per client and enforces rate limits.
    Thread-safe using locks.
    
    Attributes:
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
        cleanup_interval: Interval (in requests) to cleanup old entries
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        cleanup_interval: int = 100,
    ):
        """Initialize rate limiter.
        
        Args:
            max_requests: Max requests per client in time window (default: 100)
            window_seconds: Time window in seconds (default: 60)
            cleanup_interval: Cleanup old entries every N requests (default: 100)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval
        
        # Storage for request timestamps per client
        # Format: {client_id: [timestamp1, timestamp2, ...]}
        self.request_history: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = Lock()
        
        # Track requests for cleanup
        self._request_count = 0
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client is allowed.
        
        Args:
            client_id: Unique identifier for the client (e.g., IP address)
            
        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - self.window_seconds
            
            # Get timestamps within the window
            if client_id not in self.request_history:
                self.request_history[client_id] = []
            
            # Remove old timestamps outside the window
            self.request_history[client_id] = [
                ts for ts in self.request_history[client_id]
                if ts > window_start
            ]
            
            # Check if limit exceeded
            if len(self.request_history[client_id]) >= self.max_requests:
                return False
            
            # Add current request timestamp
            self.request_history[client_id].append(current_time)
            
            # Periodic cleanup of empty entries and old clients
            self._request_count += 1
            if self._request_count % self.cleanup_interval == 0:
                self._cleanup()
            
            return True
    
    def check_rate_limit(self, client_id: str) -> None:
        """Check rate limit and raise exception if exceeded.
        
        Args:
            client_id: Unique identifier for the client (e.g., IP address)
            
        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        if not self.is_allowed(client_id):
            raise RateLimitExceededError(
                f"Rate limit exceeded: {self.max_requests} requests per "
                f"{self.window_seconds} seconds"
            )
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for a client in current window.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Number of requests remaining before hitting limit
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - self.window_seconds
            
            # Count requests in current window
            if client_id not in self.request_history:
                return self.max_requests
            
            recent_requests = [
                ts for ts in self.request_history[client_id]
                if ts > window_start
            ]
            
            return max(0, self.max_requests - len(recent_requests))
    
    def get_window_reset_time(self, client_id: str) -> float:
        """Get Unix timestamp when the rate limit window resets for a client.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            Unix timestamp when window resets (or 0 if no recent requests)
        """
        with self._lock:
            if client_id not in self.request_history or not self.request_history[client_id]:
                return 0.0
            
            oldest_request = min(self.request_history[client_id])
            reset_time = oldest_request + self.window_seconds
            return reset_time
    
    def _cleanup(self) -> None:
        """Remove empty entries and clients with no recent requests.
        
        Should only be called when holding the lock.
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Remove entries that are completely stale or empty
        clients_to_remove = []
        for client_id, timestamps in self.request_history.items():
            # Remove timestamps outside window
            self.request_history[client_id] = [
                ts for ts in timestamps
                if ts > window_start
            ]
            # Mark client for removal if no recent requests
            if not self.request_history[client_id]:
                clients_to_remove.append(client_id)
        
        # Remove empty clients
        for client_id in clients_to_remove:
            del self.request_history[client_id]
    
    def reset(self, client_id: str = None) -> None:
        """Reset rate limit data.
        
        Args:
            client_id: Specific client to reset, or None to reset all
        """
        with self._lock:
            if client_id is None:
                self.request_history.clear()
                self._request_count = 0
            else:
                self.request_history.pop(client_id, None)
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics.
        
        Returns:
            Dictionary with stats including tracked clients and total requests
        """
        with self._lock:
            total_requests = sum(
                len(timestamps)
                for timestamps in self.request_history.values()
            )
            return {
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
                "tracked_clients": len(self.request_history),
                "total_tracked_requests": total_requests,
            }
