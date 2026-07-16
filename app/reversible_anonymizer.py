# reversible_anonymizer.py
from cryptography.fernet import Fernet
import uuid
import json


class ReversibleAnonymizer:
    """
    NOTE: prototype-grade -- single in-memory Fernet key shared across all
    sessions, and the token map lives only in process memory. Production
    would need per-session/per-user keys and persistent storage (SQLite/Redis).
    """

    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.token_map = {}   # session_id -> {placeholder: real_value}

    def anonymize(self, session_id, text, entities):
        sorted_ents = sorted(entities, key=lambda x: x["start"], reverse=True)
        if session_id not in self.token_map:
            self.token_map[session_id] = {}

        for ent in sorted_ents:
            placeholder = f"[{ent['entity_type']}_{uuid.uuid4().hex[:6]}]"
            self.token_map[session_id][placeholder] = ent["text"]
            text = text[:ent["start"]] + placeholder + text[ent["end"]:]

        encrypted = self.cipher.encrypt(json.dumps(self.token_map[session_id]).encode())
        return text, encrypted

    def deanonymize(self, session_id, text, encrypted_map):
        decrypted = json.loads(self.cipher.decrypt(encrypted_map).decode())
        for placeholder, real_value in decrypted.items():
            text = text.replace(placeholder, real_value)
        return text
