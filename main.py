import os
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import streamlit as st

# llm
model = "llama-3.3-70b-versatile"

llm = Groq(
    model=model,
    token=st.secrets["GROQ_API_KEY"],  # when you're running local
)


# embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "./embedding_model/"

embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=embeddings_folder,
)


# loading data from source
persist_dir = "./vector_index"

if os.path.exists(persist_dir):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
else:
    documents = SimpleDirectoryReader("./data", required_exts=[".pdf"]).load_data(
        show_progress=True
    )
    text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)
    vector_index = VectorStoreIndex.from_documents(
        documents, transformations=[text_splitter], embed_model=embeddings
    )
    vector_index.storage_context.persist(persist_dir=persist_dir)

# retriever
retriever = vector_index.as_retriever(similarity_top_k=2)

# prompt
prefix_messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="You are a nice chatbot having a conversation with a human.",
    ),
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="Answer the question based only on the following context and previous conversation.",
    ),
    ChatMessage(
        role=MessageRole.SYSTEM, content="Keep your answers short and succinct."
    ),
]

# memory
memory = ChatMemoryBuffer.from_defaults()


# bot with memory
@st.cache_resource
def init_bot():
    return ContextChatEngine(
        llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages
    )


rag_bot = init_bot()

##### streamlit #####

st.title("Chatier & chatier: conversations in Wonderland")


# Display chat messages from history on app rerun
for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)

# React to user input
if prompt := st.chat_input("Curious minds wanted!"):
    # Display user message in chat message container
    st.chat_message("human").markdown(prompt)

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Going down the rabbithole for answers..."):
        # send question to chain to get answer
        answer = rag_bot.chat(prompt)

        # extract answer from dictionary returned by chain
        response = answer.response

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
