"""HTTP layer: upload a video, watch it process, download the clips.

The pipeline is the product; this package is the way a customer reaches it.
It deliberately owns no video logic of its own — every job calls
``pipeline.run.process`` exactly as the CLI does.
"""

__all__ = ["app", "store", "worker"]
