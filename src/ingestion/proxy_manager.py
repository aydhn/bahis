"""
proxy_manager.py - Simple rotating proxy manager for scraping.
Handles a list of proxies, tracks their health, and rotates them on demand.
"""
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Free proxy list (Example - unreliable but good for structure testing)
# In production, replace this with a paid service or a dynamic scraper source.
FREE_PROXY_LIST = [
    # Format: "http://ip:port" or "http://user:pass@ip:port"
    # These are just placeholders/examples. Real free proxies die fast.
    "http://20.111.54.16:80",
    "http://20.206.106.192:80",
    "http://20.210.113.32:80",
    "http://51.159.115.233:3128",
    "http://51.158.154.173:3128",
    "http://51.158.189.143:3128",
    "http://62.171.168.163:3128",
    "http://163.172.189.32:3128",
    "http://195.154.222.25:3128",
    "http://163.172.146.119:3128",
]

class ProxyManager:
    """Manages a pool of proxies with health tracking."""

    def __init__(self, proxies: list[str] = None):
        self._proxies = proxies or FREE_PROXY_LIST.copy()
        self._bad_proxies = set()
        self._current_proxy: Optional[str] = None
        self._stats = {p: {"success": 0, "fail": 0} for p in self._proxies}
        
    def get_proxy(self) -> Optional[str]:
        """Returns a healthy proxy from the pool or None if exhausted."""
        available = [p for p in self._proxies if p not in self._bad_proxies]
        
        if not available:
            # If all are bad, reset the bad list to retry (desperate mode)
            if self._bad_proxies:
                logger.warning("[ProxyManager] All proxies marked bad. Resetting pool.")
                self._bad_proxies.clear()
                available = self._proxies
            else:
                return None # No proxies configured

        # Simple random strategy for now. Round-robin is also fine.
        self._current_proxy = random.choice(available)
        return self._current_proxy

    def report_status(self, proxy: str, success: bool):
        """Reports the outcome of a request using the given proxy."""
        if proxy not in self._stats:
            self._stats[proxy] = {"success": 0, "fail": 0}
            
        if success:
            self._stats[proxy]["success"] += 1
            # If it was bad, maybe forgive it? (Not implemented for simplicity)
        else:
            self._stats[proxy]["fail"] += 1
            # Mark as bad if too many failures?
            # For now, immediate ban on 403 usually managed by caller via 'rotate'
            # But here we track generic connection errors
            if self._stats[proxy]["fail"] > 3:
                 self._bad_proxies.add(proxy)

    def mark_bad(self, proxy: str):
        """Explicitly marks a proxy as bad (e.g. got 403)."""
        if proxy:
            self._bad_proxies.add(proxy)
            logger.info(f"[ProxyManager] Prx marked BAD: {proxy}")

    @property
    def stats(self):
        return self._stats
