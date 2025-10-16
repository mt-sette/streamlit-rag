from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import streamlit as st

prompt_options = {
    "basic_context": "You are a Chatbot conversing with a human about consciousness. Base all your answers on the provided context.",
    "Helpful": "Be constructive, provide simple explanations and analogies, and answer clearly and succinctly.",
    "Unhelpful": "Be sarcastic, cryptic, and deliberately unhelpful while referencing the provided context in a convoluted way.",
}

if "mode" not in st.session_state:
    st.session_state.mode = "Helpful"


@st.cache_resource
def init_llm(temp=0.01):
    return Groq(
        model="llama-3.3-70b-versatile",
        max_new_tokens=768,
        temperature=temp,
        top_p=0.95,
        repetition_penalty=1.03,
        api_key=st.secrets["GROQ_API_KEY"],
    )


@st.cache_resource
def init_retriever(num_chunks=2):
    embeddings = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=".cache/embedding_model",
    )
    storage_context = StorageContext.from_defaults(persist_dir="./vector_index")
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
    return vector_index.as_retriever(similarity_top_k=num_chunks)


@st.cache_resource
def init_memory():
    return ChatMemoryBuffer.from_defaults()


@st.cache_resource
def init_bot(mode="Helpful", temp=0.01, num_chunks=2):
    llm = init_llm(temp)
    retriever = init_retriever(num_chunks)
    memory = init_memory()
    # Always include basic_context plus the selected mode (if not basic)
    keys = ["basic_context"] + ([mode] if mode != "basic_context" else [])
    prefix_messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt_options[k]) for k in keys
    ]
    return ContextChatEngine(
        llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages
    )


st.title("Car Mechanic Assistant")

temp = st.slider("Adjust the bot's creativity level", 0.0, 2.0, 0.01)

mode = st.selectbox(
    "Choose an attitude for the bot:", ["Helpful", "Unhelpful"], index=0
)
st.session_state.mode = mode

st.sidebar.subheader("Context size")
chunk_levels = {"Small": 2, "Medium": 3, "Big": 4}
selected = st.sidebar.radio(
    "How many chunks should be retrieved?", list(chunk_levels.keys()), index=1
)
num_chunks = chunk_levels[selected]
st.sidebar.caption(f"Retrieving {num_chunks} chunk(s) per query")

rag_bot = init_bot(mode=st.session_state.mode, temp=temp, num_chunks=num_chunks)

# New chat button to reset conversation
if st.button("New chat"):
    rag_bot.reset()
    st.rerun()

for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)

if prompt := st.chat_input("Enter your message"):
    st.chat_message("human").markdown(prompt)
    with st.spinner("Computing..."):
        answer = rag_bot.chat(prompt)
        response = answer.response
        with st.chat_message("assistant"):
            st.markdown(response)
