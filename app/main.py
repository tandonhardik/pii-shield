# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

from trained_models import classify_sensitivity
from proxy_hydrator import semantic_hydrate
from privacy_accountant import SessionPrivacyAccountant
from reversible_anonymizer import ReversibleAnonymizer
from pii_detector import detect_pii_ensemble

app = FastAPI(title="PII Shield", description="Context-aware PII detection & anonymization middleware")
accountant = SessionPrivacyAccountant()
reversible_anonymizer = ReversibleAnonymizer()

# Entity types where a false-PUBLIC verdict from the sensitivity classifier
# would be a catastrophic leak. The classifier is the primary contextual
# signal; this is a narrow deterministic safety net on top of it, not a
# replacement for it (see README for the design rationale).
CATASTROPHIC_OVERRIDE_TYPES = {"US_SSN", "SSN", "CREDIT_CARD"}


class PromptRequest(BaseModel):
    session_id: str
    prompt: str
    use_reversible: bool = False
    policy: str = "auto"   # "auto", "public", "mask"


async def mock_llm(prompt: str) -> str:
    """Replace with a real Anthropic/OpenAI client call in production."""
    await asyncio.sleep(0.1)
    return f"Mock LLM processed: {prompt}"


def _public_bypass_allowed(prompt: str) -> bool:
    entities = detect_pii_ensemble(prompt)
    catastrophic = [e for e in entities if e["entity_type"] in CATASTROPHIC_OVERRIDE_TYPES]
    return len(catastrophic) == 0


@app.post("/process")
async def process_prompt(req: PromptRequest):
    effective_policy = req.policy

    if req.policy == "auto":
        intent = classify_sensitivity(req.prompt)
        if intent == "PUBLIC" and _public_bypass_allowed(req.prompt):
            llm_resp = await mock_llm(req.prompt)
            return {
                "sanitized_prompt": req.prompt,
                "llm_response": llm_resp,
                "risk": 0.0,
                "reversible": False,
                "policy_used": "public_bypass",
            }
        else:
            effective_policy = "mask"

    elif req.policy == "public":
        if _public_bypass_allowed(req.prompt):
            llm_resp = await mock_llm(req.prompt)
            return {
                "sanitized_prompt": req.prompt,
                "llm_response": llm_resp,
                "risk": 0.0,
                "reversible": False,
                "policy_used": "public_bypass",
            }
        else:
            effective_policy = "mask"

    pii_entities = detect_pii_ensemble(req.prompt)

    risk = accountant.update_and_check(req.session_id, pii_entities)
    if risk > 0.85:
        print(f"WARNING: Re-identification risk for session {req.session_id} is {risk:.2%}")

    if req.use_reversible:
        sanitized, encrypted_map = reversible_anonymizer.anonymize(
            req.session_id, req.prompt, pii_entities
        )
    else:
        sanitized = semantic_hydrate(req.prompt, pii_entities)
        encrypted_map = None

    llm_resp = await mock_llm(sanitized)

    final_response = llm_resp
    if req.use_reversible and encrypted_map:
        final_response = reversible_anonymizer.deanonymize(
            req.session_id, llm_resp, encrypted_map
        )

    return {
        "sanitized_prompt": sanitized,
        "llm_response": final_response,
        "risk": risk,
        "reversible": req.use_reversible,
        "policy_used": effective_policy,
        "entities_detected": len(pii_entities),
    }


@app.get("/session/{session_id}/risk")
def get_session_risk(session_id: str):
    return {"session_id": session_id, "risk": accountant.get_risk(session_id)}


@app.get("/health")
def health():
    return {"status": "ok"}
