"""Generated bounded synthetic tokenizer."""

VOCABULARY_SIZE = 48


def encode(text):
    if not text:
        raise ValueError("text cannot be empty")
    return [(ord(character) % (VOCABULARY_SIZE - 1)) + 1 for character in text]


def decode_token(token_id):
    if not 0 <= token_id < VOCABULARY_SIZE:
        raise ValueError("token outside vocabulary")
    return chr(33 + token_id % 90)
