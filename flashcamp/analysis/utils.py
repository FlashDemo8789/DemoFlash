def get(d: dict, name: str, default=0):
    """Safe fetch: returns default if key missing or empty."""
    val = d.get(name)
    return default if val in (None, "", []) else val 