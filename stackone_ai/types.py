"""Shared types, configuration and errors for the StackOne AI SDK."""

from __future__ import annotations

import functools
import re
import unicodedata
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias, TypedDict

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

JsonDict: TypeAlias = dict[str, Any]
Headers: TypeAlias = dict[str, str]

# StackOne API base URL
DEFAULT_BASE_URL: str = "https://api.stackone.com"


ToolMode = Literal["individual", "search_execute"]
"""How the MCP endpoint lists tools.

``"individual"`` (the server default) lists one tool per action — hundreds per
account. ``"search_execute"`` lists two meta tools per connector instead, a
``*_search_actions`` that ranks actions for a natural-language query and an
``*_execute_action`` that runs one by id. The catalog stays small regardless of
how many accounts are linked, which is what keeps it inside a model's context.
"""

SUBMIT_FEEDBACK_TOOL_NAME: str = "stackone_submit_feedback"
"""The one global tool the MCP endpoint serves in every mode when feedback is enabled.

It is not a connector action, so the SDK calls it over MCP ``tools/call`` — where it
was listed — rather than ``/actions/rpc``, whatever the toolset's mode.
"""

FeedbackRating = Literal["positive", "negative", "neutral"]
FeedbackSource = Literal["model", "user", "system"]
FeedbackCategory = Literal["search", "execute", "defender", "connection", "general"]


class StackOneError(Exception):
    """Base exception for StackOne errors"""

    pass


class StackOneAPIError(StackOneError):
    """Raised when the StackOne API returns an error"""

    def __init__(self, message: str, status_code: int, response_body: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ToolsetError(StackOneError):
    """Base exception for toolset errors.

    A subclass of StackOneError, so ``except StackOneError`` catches everything this
    SDK raises. The two used to be unrelated siblings, which meant the obvious
    catch-all silently missed ToolsetConfigError and ToolsetLoadError — the errors a
    user is most likely to hit on their very first call. Existing
    ``except ToolsetError`` clauses are unaffected.
    """

    pass


class ToolsetConfigError(ToolsetError):
    """Raised when there is an error in the toolset configuration"""

    pass


class ToolsetLoadError(ToolsetError):
    """Raised when there is an error loading tools"""

    pass


class ExecuteToolsConfig(TypedDict, total=False):
    """Execution configuration for the StackOneToolSet constructor.

    Controls default account scoping and timeout for tool execution.
    """

    account_ids: list[str]
    """Account IDs to scope tool discovery and execution."""

    timeout: float
    """Request timeout in seconds. Default: 60. Can also be set as a top-level
    constructor param which takes precedence."""


class ParameterLocation(str, Enum):
    """Valid locations for parameters in requests"""

    HEADER = "header"
    QUERY = "query"
    PATH = "path"
    BODY = "body"
    FILE = "file"  # For file uploads


def validate_method(v: str) -> str:
    """Validate HTTP method is uppercase and supported"""
    if not isinstance(v, str):
        raise ValueError(f"Unsupported HTTP method: {v}")
    method = v.upper()
    if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
        raise ValueError(f"Unsupported HTTP method: {method}")
    return method


def is_json_content_type(content_type: str) -> bool:
    """Whether a response body should be parsed as JSON based on its Content-Type.

    Only genuine JSON media types are parsed (``application/json`` and structured
    suffixes such as ``application/problem+json``). Anything else - including a
    missing Content-Type - is treated as opaque content (a file download), so the
    raw bytes are returned instead of being force-decoded as UTF-8/JSON. This mirrors
    how the StackOne generated SDKs default unknown bodies to ``application/octet-stream``.
    """
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


# The longest filename, in UTF-8 bytes, that common filesystems (ext4, APFS, NTFS) accept.
_MAX_FILENAME_BYTES = 255

# `/` and `\` are path separators whatever the host OS, and so is `:` — "C:evil.exe" writes
# to drive C:'s current directory on Windows, and "report.pdf:payload" to an NTFS alternate
# data stream.
_PATH_SEPARATORS = re.compile(r"[/\\:]")

# Format characters Unicode 15 added. Python 3.11 ships Unicode 14, where they are
# unassigned, so unicodedata alone would let them through.
_NEW_FORMAT_CHARS = frozenset(chr(code) for code in range(0x13439, 0x13440))

# JavaScript's whitespace, which String.prototype.trim() and `\s` use. The header grammar
# is matched the way the Node SDK matches it, so the two SDKs agree on every header; Python's
# own `\s` and str.strip() also take U+001C-U+001F and U+0085, and miss U+FEFF.
_JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_WS = f"[{_JS_WHITESPACE}]*"

# re.ASCII keeps IGNORECASE to ASCII case folding, as a JavaScript regex without the `u`
# flag does: otherwise "fılename" (dotless ı) matches `filename`.
_EXTENDED_FILENAME = re.compile(rf"(?:^|;){_WS}filename\*{_WS}={_WS}([^']*)'[^']*'([^;]+)", re.I | re.A)
_QUOTED_FILENAME = re.compile(rf'(?:^|;){_WS}filename{_WS}={_WS}"([^"]*)"', re.I | re.A)
_BARE_FILENAME = re.compile(rf"(?:^|;){_WS}filename{_WS}={_WS}([^;]+)", re.I | re.A)
_PERCENT_ESCAPES = re.compile(r"(?:%[0-9A-Fa-f]{2})+")

# The charset labels Node's TextDecoder accepts (the WHATWG Encoding Standard's, less
# `iso-8859-16`, `x-user-defined` and the `replacement` labels, which it refuses), keyed by
# the Python codec for that encoding. Any other label falls back to UTF-8, as it does there.
# Single-byte and UTF charsets decode exactly as there. The CJK ones do not quite: Node's
# ICU converters depart from the standard on malformed bytes and some extension ranges
# (EUC-KR, Big5, Shift_JIS most), which the stdlib codecs cannot reproduce.
_CHARSET_LABELS: dict[str, tuple[str, ...]] = {
    "utf-8": ("unicode-1-1-utf-8", "unicode11utf8", "unicode20utf8", "utf-8", "utf8", "x-unicode20utf8"),
    "utf-16-be": ("unicodefffe", "utf-16be"),
    "utf-16-le": ("csunicode", "iso-10646-ucs-2", "ucs-2", "unicode", "unicodefeff", "utf-16", "utf-16le"),
    "cp866": ("866", "cp866", "csibm866", "ibm866"),
    "iso8859_2": ("csisolatin2", "iso-8859-2", "iso-ir-101", "iso8859-2", "iso88592", "iso_8859-2")
    + ("iso_8859-2:1987", "l2", "latin2"),
    "iso8859_3": ("csisolatin3", "iso-8859-3", "iso-ir-109", "iso8859-3", "iso88593", "iso_8859-3")
    + ("iso_8859-3:1988", "l3", "latin3"),
    "iso8859_4": ("csisolatin4", "iso-8859-4", "iso-ir-110", "iso8859-4", "iso88594", "iso_8859-4")
    + ("iso_8859-4:1988", "l4", "latin4"),
    "iso8859_5": ("csisolatincyrillic", "cyrillic", "iso-8859-5", "iso-ir-144", "iso8859-5", "iso88595")
    + ("iso_8859-5", "iso_8859-5:1988"),
    "iso8859_6": ("arabic", "asmo-708", "csiso88596e", "csiso88596i", "csisolatinarabic", "ecma-114")
    + ("iso-8859-6", "iso-8859-6-e", "iso-8859-6-i", "iso-ir-127", "iso8859-6", "iso88596", "iso_8859-6")
    + ("iso_8859-6:1987",),
    "iso8859_7": ("csisolatingreek", "ecma-118", "elot_928", "greek", "greek8", "iso-8859-7", "iso-ir-126")
    + ("iso8859-7", "iso88597", "iso_8859-7", "iso_8859-7:1987", "sun_eu_greek"),
    # ISO-8859-8 and its logical-order twin ISO-8859-8-I decode identically.
    "iso8859_8": ("csiso88598e", "csisolatinhebrew", "hebrew", "iso-8859-8", "iso-8859-8-e", "iso-ir-138")
    + ("iso8859-8", "iso88598", "iso_8859-8", "iso_8859-8:1988", "visual", "csiso88598i", "iso-8859-8-i")
    + ("logical",),
    "iso8859_10": ("csisolatin6", "iso-8859-10", "iso-ir-157", "iso8859-10", "iso885910", "l6", "latin6"),
    "iso8859_13": ("iso-8859-13", "iso8859-13", "iso885913"),
    "iso8859_14": ("iso-8859-14", "iso8859-14", "iso885914"),
    "iso8859_15": ("csisolatin9", "iso-8859-15", "iso8859-15", "iso885915", "iso_8859-15", "l9"),
    "koi8_r": ("cskoi8r", "koi", "koi8", "koi8-r", "koi8_r"),
    "koi8_u": ("koi8-ru", "koi8-u"),
    "mac_roman": ("csmacintosh", "mac", "macintosh", "x-mac-roman"),
    "mac_cyrillic": ("x-mac-cyrillic", "x-mac-ukrainian"),
    "cp874": ("dos-874", "iso-8859-11", "iso8859-11", "iso885911", "tis-620", "windows-874"),
    "cp1250": ("cp1250", "windows-1250", "x-cp1250"),
    "cp1251": ("cp1251", "windows-1251", "x-cp1251"),
    # Latin-1 and ASCII labels mean windows-1252 here, as they do in every browser.
    "cp1252": ("ansi_x3.4-1968", "ascii", "cp1252", "cp819", "csisolatin1", "ibm819", "iso-8859-1")
    + ("iso-ir-100", "iso8859-1", "iso88591", "iso_8859-1", "iso_8859-1:1987", "l1", "latin1")
    + ("us-ascii", "windows-1252", "x-cp1252"),
    "cp1253": ("cp1253", "windows-1253", "x-cp1253"),
    "cp1254": ("cp1254", "csisolatin5", "iso-8859-9", "iso-ir-148", "iso8859-9", "iso88599", "iso_8859-9")
    + ("iso_8859-9:1989", "l5", "latin5", "windows-1254", "x-cp1254"),
    "cp1255": ("cp1255", "windows-1255", "x-cp1255"),
    "cp1256": ("cp1256", "windows-1256", "x-cp1256"),
    "cp1257": ("cp1257", "windows-1257", "x-cp1257"),
    "cp1258": ("cp1258", "windows-1258", "x-cp1258"),
    # GBK labels decode as GB18030, its superset.
    "gb18030": ("chinese", "csgb2312", "csiso58gb231280", "gb2312", "gb_2312", "gb_2312-80", "gbk")
    + ("iso-ir-58", "x-gbk", "gb18030"),
    "big5hkscs": ("big5", "big5-hkscs", "cn-big5", "csbig5", "x-x-big5"),
    "euc_jp": ("cseucpkdfmtjapanese", "euc-jp", "x-euc-jp"),
    "iso2022_jp": ("csiso2022jp", "iso-2022-jp"),
    "cp932": ("csshiftjis", "ms932", "ms_kanji", "shift-jis", "shift_jis", "sjis", "windows-31j", "x-sjis"),
    "cp949": ("cseuckr", "csksc56011987", "euc-kr", "iso-ir-149", "korean", "ks_c_5601-1987")
    + ("ks_c_5601-1989", "ksc5601", "ksc_5601", "windows-949"),
}
_CODEC_BY_LABEL = {label: codec for codec, labels in _CHARSET_LABELS.items() for label in labels}

# Bytes Node decodes that Python's codec for the same Windows code page leaves undefined.
# Every byte 0x80-0x9F a code page does not assign passes through as that C1 control, and
# these few decode as below.
_WINDOWS_CODE_PAGE_EXTRAS: dict[str, dict[int, str]] = {
    "cp874": {0xDB: "\uf8c1", 0xDC: "\uf8c2", 0xDD: "\uf8c3", 0xDE: "\uf8c4"}
    | {0xFC: "\uf8c5", 0xFD: "\uf8c6", 0xFE: "\uf8c7", 0xFF: "\uf8c8"},
    "cp1253": {0xAA: "\u00aa"},
}
_WINDOWS_CODE_PAGES = frozenset({"cp874", *(f"cp125{digit}" for digit in range(9))})


@functools.cache
def _windows_code_page_table(codec: str) -> str:
    """The 256-character decoding table Node uses for a Windows code page."""
    table = []
    for byte in range(256):
        char = bytes([byte]).decode(codec, "replace")
        if char == "\ufffd" and 0x80 <= byte <= 0x9F:
            char = chr(byte)
        table.append(_WINDOWS_CODE_PAGE_EXTRAS.get(codec, {}).get(byte, char))
    return "".join(table)


def _percent_decode(encoded: str, charset: str) -> str:
    """Decode ``%XX`` escapes as bytes in ``charset``, the way the Node SDK does.

    Each run of escapes is decoded on its own, malformed escapes stay literal, bytes that
    do not decode become U+FFFD, and an unknown label means UTF-8. A leading BOM, which
    Node drops from each run, is left in: it is a format character, removed later anyway.
    """
    # Full Unicode lowercasing, as Node's TextDecoder does, so "\u212aoi8-r" (a Kelvin
    # sign) is KOI8-R there and here.
    label = charset.strip("\t\n\f\r ").lower()
    codec = _CODEC_BY_LABEL.get(label, "utf-8")

    def decode(run: re.Match[str]) -> str:
        data = bytes.fromhex(run.group(0).replace("%", ""))
        if codec in _WINDOWS_CODE_PAGES:
            table = _windows_code_page_table(codec)
            return "".join(table[byte] for byte in data)
        return data.decode(codec, "replace")

    return _PERCENT_ESCAPES.sub(decode, encoded)


def _as_js_string(value: str) -> str:
    """Join each surrogate pair into the one character a JavaScript string would hold.

    A JavaScript string is UTF-16, so the two halves of a pair are one character wherever
    they meet, including once whatever separated them is removed. Lone surrogates are kept.
    """
    return value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def _utf8_size(char: str) -> int:
    # A lone surrogate counts as the three bytes of the U+FFFD that TextEncoder writes.
    return len(char.encode("utf-8", "surrogatepass"))


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Truncate to at most ``max_bytes`` UTF-8 bytes without splitting a character."""
    size = 0
    for index, char in enumerate(value):
        size += _utf8_size(char)
        if size > max_bytes:
            return value[:index]
    return value


def _safe_basename(name: str | None) -> str | None:
    """Reduce a server-supplied filename to a bare, writable basename.

    The value comes from a remote ``Content-Disposition``, which in practice is chosen
    by whoever uploaded the file to the connected provider — so it is attacker-controlled.
    Returned unsanitised it is an arbitrary-file-write primitive for any caller that does
    the obvious thing and passes it to ``open()``: ``../../.ssh/authorized_keys`` and
    ``/etc/cron.d/x`` both round-trip. The RFC 5987 branch percent-decodes, so a filter
    applied before this point would be bypassed anyway; sanitise last, here, once.

    The result is the Node SDK's ``safeBasename()`` for every input: the last segment after
    any ``/``, ``\\`` or ``:``, without control or format characters, capped at 255 UTF-8
    bytes keeping the extension where one fits, or ``None`` if nothing usable is left.
    """
    if name is None:
        return None
    base = _PATH_SEPARATORS.split(_as_js_string(name))[-1]
    # Strip control characters (log and header injection) and Unicode format characters
    # — U+202E renders "\u202egnp.exe" as "…exe.png", the classic extension spoof.
    base = "".join(
        char
        for char in base
        if unicodedata.category(char) not in ("Cc", "Cf") and char not in _NEW_FORMAT_CHARS
    )
    base = _as_js_string(base).strip(_JS_WHITESPACE)
    if base in ("", ".", ".."):
        return None
    if sum(_utf8_size(char) for char in base) <= _MAX_FILENAME_BYTES:
        return base
    dot = base.rfind(".")
    suffix = base[dot:] if dot > 0 else ""
    suffix_bytes = sum(_utf8_size(char) for char in suffix)
    # An extension that cannot fit alongside at least one character of stem is not worth
    # keeping: truncate the whole name instead of emitting something still over the limit.
    if not suffix or suffix_bytes >= _MAX_FILENAME_BYTES:
        return _truncate_utf8(base, _MAX_FILENAME_BYTES)
    return _truncate_utf8(base[:dot], _MAX_FILENAME_BYTES - suffix_bytes) + suffix


def filename_from_content_disposition(value: str | None) -> str | None:
    """Extract the filename from a Content-Disposition header value, if present.

    Handles both the plain ``filename="example.pdf"`` form and the RFC 5987 extended
    ``filename*=UTF-8''example%20file.pdf`` form (which takes precedence when present).
    The extended form is percent-decoded using its declared charset (RFC 5987 permits
    both ``UTF-8`` and ``ISO-8859-1``); an unknown or empty charset falls back to UTF-8.
    Parameters are matched at a parameter boundary, so ``notfilename=`` is ignored.
    """
    if not value:
        return None
    extended = _EXTENDED_FILENAME.search(value)
    if extended:
        charset = extended.group(1).strip(_JS_WHITESPACE) or "utf-8"
        encoded = extended.group(2).strip(_JS_WHITESPACE).strip('"')
        return _safe_basename(_percent_decode(encoded, charset))
    quoted = _QUOTED_FILENAME.search(value)
    if quoted:
        return _safe_basename(quoted.group(1))
    bare = _BARE_FILENAME.search(value)
    if bare:
        return _safe_basename(bare.group(1).strip('"'))
    return None


class ExecuteConfig(BaseModel):
    """Configuration for executing a tool against an API endpoint"""

    headers: Headers = Field(default_factory=dict, description="HTTP headers to include in the request")
    method: Annotated[str, BeforeValidator(validate_method)] = Field(description="HTTP method to use")
    url: str = Field(description="API endpoint URL")
    name: str = Field(description="Tool name")
    body_type: str | None = Field(default=None, description="Content type for request body")
    parameter_locations: dict[str, ParameterLocation] = Field(
        default_factory=dict, description="Maps parameter names to their location in the request"
    )
    timeout: float = Field(default=60.0, description="Request timeout in seconds")


class ToolParameters(BaseModel):
    """Schema definition for tool parameters.

    ``properties`` is a faithful mirror of the ``inputSchema`` the MCP server served,
    with the SDK's internal ``nullable`` marker added per property. Every other root
    keyword the server sent, ``required`` included, is kept verbatim as an extra field.
    Consumers that need the raw served schema (for example the ADK plugin) read this
    directly, so nothing here may be invented or dropped.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="JSON Schema type")
    properties: JsonDict = Field(description="JSON Schema properties")
