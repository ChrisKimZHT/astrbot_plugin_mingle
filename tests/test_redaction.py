import pytest
from mingle_tests_plugin.src.redaction import redact_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("电话13800138000，邮箱a.b+test@example.com。", "电话[手机号]，邮箱[邮箱]。"),
        ("+86 13800138000", "[手机号]"),
        ("身份证11010519491231002X", "身份证[身份证号]"),
        ("110105194912310021", "110105194912310021"),
        ("订单9138001380001和版本1.2.3", "订单9138001380001和版本1.2.3"),
        ("Authorization: Bearer abc.def-123", "Authorization: Bearer [令牌]"),
        ('{"password": "hello world"}', '{"password": [凭据]}'),
        ("我的密码：abc123，继续", "我的密码：[凭据]，继续"),
        ("api_key=abc123 token='two words'", "api_key=[凭据] token=[凭据]"),
        ("sk-" + "a" * 32, "[API密钥]"),
        ("ghp_" + "a" * 36, "[API密钥]"),
        (
            "before\n-----BEGIN RSA PRIVATE KEY-----\nsecret\n"
            "-----END RSA PRIVATE KEY-----\nafter",
            "before\n[私钥]\nafter",
        ),
        ("讨论 password 的用法，今天见。", "讨论 password 的用法，今天见。"),
    ],
)
def test_redaction_rules_and_unrelated_text(text, expected):
    assert redact_text(text) == expected
    assert redact_text(expected) == expected
