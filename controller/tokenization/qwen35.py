from __future__ import annotations

from transformers import AutoTokenizer


def _normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content
    pieces = []
    for item in content:
        text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
        if text:
            pieces.append(text)
    return "".join(pieces)


class Qwen35TokenizerAdapter:
    def __init__(self, tokenizer_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    def format_messages(self, messages, enable_thinking: bool) -> str:
        normalized = [{"role": m.role, "content": _normalize_message_content(m.content)} for m in messages]
        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def encode_prompt(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
