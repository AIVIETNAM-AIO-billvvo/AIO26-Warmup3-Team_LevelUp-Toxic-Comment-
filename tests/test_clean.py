"""Unit tests for src/clean.py — focus on string-level functions."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.clean import (
    cap_repeats,
    clean_text_universal,
    collapse_ws,
    decode_html,
    expand_contractions,
    lowercase,
    remove_emails,
    remove_ips,
    remove_paths,
    remove_punct,
    remove_urls,
    strip_control_ws,
    strip_wiki_markup,
)


def test_decode_html_entities():
    assert decode_html("Tom &amp; Jerry") == "Tom & Jerry"
    assert decode_html("&quot;hello&quot;") == '"hello"'
    assert decode_html("plain text") == "plain text"


def test_strip_control_ws():
    assert strip_control_ws("line1\nline2") == "line1 line2"
    assert strip_control_ws("a\tb\rc") == "a b c"
    assert strip_control_ws("a\n\n\nb") == "a b"


def test_remove_urls():
    out = remove_urls("see http://example.com here")
    assert "http" not in out and "example.com" not in out
    assert "https" not in remove_urls("https://foo.bar/baz?x=1 ok")
    assert "www" not in remove_urls("visit www.site.org now")


def test_remove_paths_unix():
    assert "/usr/bin/python" not in remove_paths("ran /usr/bin/python script")


def test_remove_paths_windows():
    assert "C:\\Users" not in remove_paths(r"opened C:\Users\bob\file.txt today")


def test_remove_emails():
    assert "@" not in remove_emails("contact me at foo.bar@example.com please")


def test_remove_ipv4():
    assert "192.168" not in remove_ips("from 192.168.1.1 came spam")


def test_remove_ipv6():
    assert "fe80" not in remove_ips("addr fe80::1ff:fe23:4567:890a here")
    assert "2001" not in remove_ips("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 here")


def test_strip_wiki_link_and_template():
    assert "[[" not in strip_wiki_markup("see [[Article]] for more")
    assert "{{" not in strip_wiki_markup("uses {{cite web}} template")


def test_strip_wiki_header_keeps_text():
    out = strip_wiki_markup("== Section Title ==")
    assert "Section Title" in out
    assert "==" not in out


def test_strip_wiki_bold_italic_keeps_text():
    out = strip_wiki_markup("''italic'' and '''bold'''")
    assert "italic" in out and "bold" in out


def test_cap_repeats():
    assert cap_repeats("loooool") == "lool"
    assert cap_repeats("!!!!!") == "!!"
    assert cap_repeats("ok") == "ok"


def test_collapse_ws():
    assert collapse_ws("  a   b   c  ") == "a b c"
    assert collapse_ws("\t\thello\n") == "hello"


def test_lowercase():
    assert lowercase("Hello World") == "hello world"


def test_expand_contractions():
    out = expand_contractions("don't")
    assert out.lower() in ("do not", "do not.")


def test_remove_punct():
    assert remove_punct("hello, world!") == "hello  world "


def test_clean_text_universal_pipeline():
    raw = (
        "Check &amp; visit http://foo.com\n"
        "or contact me@x.com from 1.2.3.4!!!!!\n"
        "[[Wiki]] {{template}}"
    )
    cleaned = clean_text_universal(raw)
    assert "&amp;" not in cleaned
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "1.2.3.4" not in cleaned
    assert "[[" not in cleaned
    assert "{{" not in cleaned
    assert "!!!!!" not in cleaned
    assert "  " not in cleaned
    assert "\n" not in cleaned


def test_clean_text_universal_handles_non_string():
    assert clean_text_universal(None) == ""
    assert clean_text_universal(123) == ""


def test_clean_text_universal_empty():
    assert clean_text_universal("") == ""
    assert clean_text_universal("   \n\t  ") == ""


@pytest.mark.parametrize(
    "raw",
    [
        "FUCK FUCK FUCK FUCK",
        "  trailing spaces  ",
        "MIXED Case Text",
        "no junk here",
    ],
)
def test_clean_text_universal_no_crash(raw):
    out = clean_text_universal(raw)
    assert isinstance(out, str)
