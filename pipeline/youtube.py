"""YouTube Data API v3: metadata and ownership for a pasted link.

What this module deliberately does not do is download. The Data API has no
endpoint that returns video bytes — it covers metadata, search, uploads and
captions, and that is the whole reason yt-dlp exists. Bytes come from
`pipeline.fetch`; this module answers *what* the video is and *whose* it is.

The ownership check is the point. Reposting other creators' videos is what
gets API access revoked, and losing API access kills the product rather than
one account. Comparing the video's channel against the authenticated user's
own channel turns that from a line in the README into something the code can
actually enforce.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

API = "https://www.googleapis.com/youtube/v3"

#: watch?v=, youtu.be/, /shorts/, /embed/, /live/, and the /v/ legacy form
_ID_IN_PATH = re.compile(r"^/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})")
_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_DURATION = re.compile(
    r"^P(?:(?P<d>\d+)D)?T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?$"
)


class YouTubeError(RuntimeError):
    """The API refused, or the link is not a YouTube video."""


@dataclass
class VideoInfo:
    id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    duration: float
    tags: list[str]
    url: str

    @property
    def watch_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"


def video_id(url: str) -> str:
    """Pull the 11-character id out of any of YouTube's link shapes."""
    url = url.strip()
    if _BARE_ID.match(url):
        return url

    parts = urllib.parse.urlsplit(url if "//" in url else f"https://{url}")
    host = parts.netloc.lower().removeprefix("www.").removeprefix("m.")

    if host in ("youtu.be", "youtube.be"):
        candidate = parts.path.lstrip("/").split("/")[0]
        if _BARE_ID.match(candidate):
            return candidate
        raise YouTubeError(f"no video id in {url!r}")

    if host not in ("youtube.com", "youtube-nocookie.com", "music.youtube.com"):
        raise YouTubeError(f"{url!r} is not a YouTube link")

    if parts.path in ("/watch", "/watch/"):
        found = urllib.parse.parse_qs(parts.query).get("v", [""])[0]
        if _BARE_ID.match(found):
            return found
        raise YouTubeError(f"no video id in {url!r}")

    match = _ID_IN_PATH.match(parts.path)
    if match:
        return match.group(1)
    raise YouTubeError(f"no video id in {url!r}")


def parse_duration(iso: str) -> float:
    """ISO-8601 duration (PT1H2M3S) -> seconds. Live streams report P0D."""
    match = _DURATION.match(iso or "")
    if not match:
        return 0.0
    d = match.groupdict(default="0")
    return (int(d["d"]) * 86400 + int(d["h"]) * 3600
            + int(d["m"]) * 60 + int(d["s"]))


def _get(path: str, params: dict, *, token: str | None = None) -> dict:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v})
    req = urllib.request.Request(f"{API}/{path}?{query}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read())["error"]["message"]
        except Exception:
            detail = exc.reason
        raise YouTubeError(f"YouTube API {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise YouTubeError(f"cannot reach the YouTube API: {exc.reason}") from None


def describe(
    url: str, *, api_key: str | None = None, token: str | None = None
) -> VideoInfo:
    """Metadata for a link. Needs an API key, or an OAuth access token."""
    api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key and not token:
        raise YouTubeError(
            "set YOUTUBE_API_KEY in .env, or authorise with OAuth"
        )

    vid = video_id(url)
    data = _get("videos", {"part": "snippet,contentDetails",
                           "id": vid, "key": api_key}, token=token)
    items = data.get("items") or []
    if not items:
        raise YouTubeError(f"video {vid} not found, or it is private")

    snippet = items[0].get("snippet", {})
    content = items[0].get("contentDetails", {})
    return VideoInfo(
        id=vid,
        title=snippet.get("title", ""),
        description=snippet.get("description", ""),
        channel_id=snippet.get("channelId", ""),
        channel_title=snippet.get("channelTitle", ""),
        duration=parse_duration(content.get("duration", "")),
        tags=snippet.get("tags", []) or [],
        url=url,
    )


def my_channel_id(token: str) -> str:
    """The authenticated user's own channel. Needs an OAuth access token."""
    data = _get("channels", {"part": "id", "mine": "true"}, token=token)
    items = data.get("items") or []
    if not items:
        raise YouTubeError("this account has no YouTube channel")
    return items[0]["id"]


def owns(info: VideoInfo, token: str) -> bool:
    """Whether the authenticated user owns the video behind the link.

    Without OAuth there is no way to answer this: an API key identifies the
    project, not a person. Callers that cannot authorise should treat
    ownership as unknown rather than assuming it.
    """
    return bool(info.channel_id) and info.channel_id == my_channel_id(token)
