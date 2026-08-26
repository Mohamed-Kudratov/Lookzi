"""Lookzi service: web tier, queue, workers.

Imported as a package so `python -m service.fake_worker` and
`uvicorn service.app:app` both resolve the sibling modules the same way.
"""
