import threading
import time
from collections import OrderedDict
from typing import Callable, Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class _Entry(Generic[T]):
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: Optional[T] = None
        self.created = time.monotonic()


class IdempotencyCache(Generic[T]):
    """Rovnaké message_id sa spracuje len raz (v rámci jednej inštancie).

    n8n pri timeoute request zopakuje. Bez tejto ochrany by zákazník mohol dostať dve odpovede.
    Opakovaný request dostane uloženú odpoveď; ak prvý ešte beží, počká na jeho výsledok.
    Neúspešné spracovanie sa neukladá, takže retry po chybe prebehne normálne.
    V produkcii s viacerými inštanciami by stav patril do zdieľaného úložiska (napr. Firestore).
    """

    def __init__(self, ttl_seconds: float = 3600, max_items: int = 1000, wait_seconds: float = 110):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.wait_seconds = wait_seconds
        self._items: "OrderedDict[str, _Entry[T]]" = OrderedDict()
        self._lock = threading.Lock()

    def run(self, key: str, fn: Callable[[], T]) -> Tuple[T, bool]:
        """Vráti (výsledok, bol_to_duplikát)."""
        with self._lock:
            self._evict()
            entry = self._items.get(key)
            owner = entry is None
            if owner:
                entry = self._items[key] = _Entry()

        if not owner:
            entry.done.wait(self.wait_seconds)
            if entry.result is not None:
                return entry.result, True
            # Prvý pokus zlyhal (alebo ešte beží príliš dlho) – spracuj ako nový request
            with self._lock:
                if self._items.get(key) is entry:
                    del self._items[key]
            return self.run(key, fn)

        try:
            entry.result = fn()
            return entry.result, False
        except BaseException:
            with self._lock:
                if self._items.get(key) is entry:
                    del self._items[key]
            raise
        finally:
            entry.done.set()

    def _evict(self) -> None:
        now = time.monotonic()
        while self._items:
            key, entry = next(iter(self._items.items()))
            if len(self._items) > self.max_items or (entry.done.is_set() and now - entry.created > self.ttl):
                del self._items[key]
            else:
                break
