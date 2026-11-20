import streamlit as st
from feedback_summarizer.inference import load_model, summarize

st.set_page_config(page_title="Student Feedback Summarizer", page_icon="📝")
st.title("Student Feedback Summarizer")
st.write("Turn student comments into concise, constructive curriculum-improvement insights.")

@st.cache_resource
def get_model():
    return load_model()

try:
    tokenizer, model, device = get_model()
    st.caption(f"Merged LoRA model loaded on {device.upper()}")
    feedback = st.text_area("Paste student feedback", height=220,
        placeholder="Students found recursion difficult and requested more worked examples...")
    if st.button("Generate summary", type="primary"):
        if not feedback.strip():
            st.warning("Please enter some student feedback.")
        else:
            with st.spinner("Summarizing feedback..."):
                result = summarize(feedback, tokenizer, model, device)
            st.subheader("Instructor summary")
            st.write(result)
            st.caption("Review the generated summary before making curriculum decisions.")
except FileNotFoundError as e:
    st.warning(str(e))
    st.info("Train the LoRA adapter and merge it before launching the app.")
