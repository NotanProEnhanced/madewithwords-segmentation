"""Every first-party brand's preview carries its own wordmark, never typortrait.com.

Faith in Words was missing from the map on 2026-09-20 and its previews read
"typortrait.com". The brand list here mirrors the skins main.py recognises.
"""
from app.pipeline.watermark import _brand_mark


def test_first_party_brands_have_their_own_mark():
    assert _brand_mark("typortrait") == "typortrait.com"
    assert _brand_mark("lovedinwords") == "LovedInWords.com"
    assert _brand_mark("faithinwords") == "FaithInWords.com"
    assert _brand_mark("pawsinwords") == "PawsInWords.com"
    assert _brand_mark("keepsake") == "LovedInWords.com"


def test_partner_and_case_insensitive():
    assert _brand_mark("everloved") == "Ever Loved"
    assert _brand_mark(" FaithInWords ") == "FaithInWords.com"
    assert _brand_mark("") == "typortrait.com"
    assert _brand_mark("unknown-skin") == "typortrait.com"
