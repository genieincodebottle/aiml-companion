"""Typed failures, so a caller can tell a bad image from a broken model.

Two of these are permanent and one is temporary. Retrying an unreadable image
burns the queue forever; retrying a missing checkpoint succeeds as soon as the
model server returns. A caller cannot make that decision from a bare Exception.
"""


class TriageError(Exception):
    """Base for everything this service raises."""


class UnreadableImage(TriageError):
    """The file is corrupt or not an image. Route to review, do not retry."""


class IndexStale(TriageError):
    """Index was built by a different embedding_version. Rebuild before serving."""


class ModelUnavailable(TriageError):
    """Checkpoint missing or the device is gone. Retry is worth it."""
