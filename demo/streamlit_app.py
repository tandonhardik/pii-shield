import streamlit as st
import requests

st.set_page_config(page_title="PII Shield", page_icon="🛡️", layout="wide")
st.title("🛡️ PII Shield")
st.caption("Context-aware PII detection and semantic anonymization for LLM prompts")

API_URL = st.text_input("API URL", "http://localhost:8000")
session_id = st.session_state.setdefault("session_id", "streamlit-demo")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    prompt = st.text_area(
        "Enter a prompt (try adding a name, SSN, medical condition...)",
        height=150,
        placeholder="e.g. My SSN is 123-45-6789 and I live in Chicago",
    )
    use_reversible = st.checkbox("Reversible mode (recover exact original after LLM response)")
    submit = st.button("Analyze & Send", type="primary")

with col2:
    st.subheader("Result")
    if submit and prompt:
        try:
            resp = requests.post(
                f"{API_URL}/process",
                json={
                    "session_id": session_id,
                    "prompt": prompt,
                    "use_reversible": use_reversible,
                    "policy": "auto",
                },
                timeout=15,
            ).json()

            policy = resp.get("policy_used", "unknown")
            badge_color = {"public_bypass": "green", "mask": "orange"}.get(policy, "gray")
            st.markdown(f"**Policy applied:** :{badge_color}[{policy}]")

            st.markdown("**Sanitized prompt sent to LLM:**")
            st.code(resp["sanitized_prompt"])

            st.markdown("**LLM response:**")
            st.info(resp["llm_response"])

            risk = resp.get("risk", 0.0)
            st.markdown("**Cumulative session re-identification risk:**")
            st.progress(min(risk, 1.0))
            st.caption(f"{risk:.1%}")

            if risk > 0.85:
                st.error("⚠️ High re-identification risk for this session")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.divider()
st.caption(f"Session ID: `{session_id}` — risk accumulates across turns in this session. Refresh to reset.")

with st.expander("Try the multi-turn risk demo"):
    st.markdown(
        "Send these one at a time (same session) and watch the risk bar climb:\n\n"
        "1. `I am 45 years old`\n"
        "2. `I live in zip code 60614`\n"
        "3. `I have diabetes`"
    )
