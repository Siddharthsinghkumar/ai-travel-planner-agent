import asyncio
from functools import wraps

_HIGH_IMPACT_REGISTRY = set()


def high_impact(action_type: str):
    def decorator(fn):
        _HIGH_IMPACT_REGISTRY.add(fn.__qualname__)
        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                self_or_first = args[0] if args else None
                approval_granted = getattr(self_or_first, "_approval_granted", True)
                if not approval_granted:
                    raise PermissionError(
                        f"high_impact action '{action_type}' ({fn.__qualname__}) called without approval"
                    )
                return await fn(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                self_or_first = args[0] if args else None
                approval_granted = getattr(self_or_first, "_approval_granted", True)
                if not approval_granted:
                    raise PermissionError(
                        f"high_impact action '{action_type}' ({fn.__qualname__}) called without approval"
                    )
                return fn(*args, **kwargs)
            return sync_wrapper
    return decorator


def get_registry():
    return frozenset(_HIGH_IMPACT_REGISTRY)
