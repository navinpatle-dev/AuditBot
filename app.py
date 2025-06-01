import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
import pdfplumber
import os
import json
from dotenv import load_dotenv

# --- Configuration and Setup ---
try:
    GOOGLE_API_KEY = st.secrets["API_KEY"] # Or st.secrets["GOOGLE_API_KEY"] if you name it that way in secrets
except KeyError:
    st.error("API_KEY not found in Streamlit secrets. Please add it to your secrets.toml or deployment configuration.")
    st.stop()
except Exception as e: # Catch other potential errors like st.secrets not being available in some contexts
    st.error(f"Could not load API_KEY from Streamlit secrets: {e}")
    st.stop()
DOCS_DIRECTORY = "C:\\Users\\Computer HuB\\Desktop\\Projects\\Audit RAG\\docs"  # Directory containing your documents
FAISS_INDEX_DIR = "faiss_index_streamlit_rag_multi" # Directory to store FAISS index and manifest
MANIFEST_FILE = os.path.join(FAISS_INDEX_DIR, "index_manifest.json")
SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf") # Add more like .pdf, .docx if you install parsers

st.set_page_config(layout="wide", page_title="� RAG Tool (Multi-Doc)")
st.title("🧠 RAG Tool with Multiple Document Context")

# --- Helper Functions & Caching ---

# Custom Stream Handler to update Streamlit UI
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource
def get_embeddings_model(api_key):
    """Loads the embedding model, cached for efficiency."""
    if not api_key:
        st.error("Google API Key not found. Please set it in your .env file (API_KEY=...).")
        st.stop()
    try:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()

@st.cache_data # Using cache_data as this returns data, not a resource like a model
def get_document_file_states(docs_dir_path):
    """
    Gets a sorted tuple of (filepath, modification_time) for supported files
    in the directory. This is hashable and can be used to detect changes.
    """
    file_states = []
    if not os.path.isdir(docs_dir_path):
        st.sidebar.warning(f"Documents directory not found: {docs_dir_path}")
        return tuple() # Return empty tuple if dir doesn't exist

    for filename in sorted(os.listdir(docs_dir_path)): # Sort for consistent order
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            filepath = os.path.join(docs_dir_path, filename)
            try:
                mod_time = os.path.getmtime(filepath)
                file_states.append((filepath, mod_time))
            except OSError:
                # Handle cases where file might be deleted between listdir and getmtime
                st.sidebar.warning(f"Could not access {filepath}, skipping.")
                continue
    return tuple(file_states)

def load_manifest():
    """Loads the manifest file if it exists."""
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.sidebar.warning("Manifest file is corrupted. Index will be rebuilt.")
            return None
    return None

def save_manifest(file_states_tuple):
    """Saves the current file states to the manifest."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump({"file_states": list(file_states_tuple)}, f) # Convert tuple to list for JSON

def table_to_markdown(table_data, headers=None):
    """Converts a list of lists (table data) to a Markdown formatted string."""
    if not table_data:
        return ""
    
    markdown = ""
    
    # If headers are provided separately or are the first row of table_data
    if headers:
        actual_headers = [str(h) if h is not None else "" for h in headers]
        data_rows = table_data
    elif table_data:
        actual_headers = [str(cell) if cell is not None else "" for cell in table_data[0]]
        data_rows = table_data[1:]
    else: # Should not happen if table_data is not empty, but as a fallback
        return ""

    markdown += "| " + " | ".join(actual_headers) + " |\n"
    markdown += "| " + " | ".join("---" for _ in actual_headers) + " |\n"
    
    for row in data_rows:
        markdown += "| " + " | ".join(str(cell) if cell is not None else "" for cell in row) + " |\n"
    return markdown

@st.cache_resource(show_spinner="Loading or Creating Vector Database...")
def load_or_create_vector_db(_embedding_model, current_file_states_tuple):
    if not current_file_states_tuple:
        st.sidebar.error(f"No supported documents found in '{DOCS_DIRECTORY}'. Please add some.")
        if not os.path.exists(FAISS_INDEX_DIR):
             os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        dummy_doc = [Document(page_content="No documents loaded.")]
        empty_db = FAISS.from_documents(dummy_doc, _embedding_model)
        empty_db.save_local(FAISS_INDEX_DIR)
        save_manifest(tuple())
        st.sidebar.warning("Created an empty vector database as no documents were found.")
        return empty_db

    manifest_data = load_manifest()
    faiss_files_exist = os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")) and \
                        os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.pkl"))

    rebuild_needed = True
    if faiss_files_exist and manifest_data:
        saved_file_states = tuple(map(tuple, manifest_data.get("file_states", [])))
        if saved_file_states == current_file_states_tuple:
            rebuild_needed = False
        else:
            st.sidebar.info("Document changes detected. Rebuilding FAISS index.")
    elif not faiss_files_exist:
        st.sidebar.info("FAISS index not found. Building new index.")
    else:
        st.sidebar.info("FAISS index or manifest incomplete. Rebuilding.")

    if not rebuild_needed:
        try:
            vectordb = FAISS.load_local(FAISS_INDEX_DIR, _embedding_model, allow_dangerous_deserialization=True)
            st.sidebar.success(f"Loaded FAISS index from {FAISS_INDEX_DIR}")
            return vectordb
        except Exception as e:
            st.sidebar.warning(f"Failed to load FAISS index: {e}. Rebuilding...")

    st.sidebar.info(f"Creating new FAISS index from documents in '{DOCS_DIRECTORY}'. This may take a moment...")
    all_docs_for_langchain = [] # Renamed from all_docs_content for clarity
    loaded_file_count = 0

    for filepath, _ in current_file_states_tuple:
        base_filename = os.path.basename(filepath)
        try:
            if filepath.lower().endswith((".txt", ".md")):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    doc = Document(page_content=content, metadata={"source": base_filename, "content_type": "text"})
                    all_docs_for_langchain.append(doc)
                    loaded_file_count +=1 # Count only if content is added
                else:
                    st.sidebar.warning(f"Skipping empty file: {base_filename}")
            
            elif filepath.lower().endswith(".pdf"):
                file_processed_successfully = False
                with pdfplumber.open(filepath) as pdf:
                    st.sidebar.write(f"Processing PDF: {base_filename} ({len(pdf.pages)} pages)")
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            all_docs_for_langchain.append(Document(
                                page_content=page_text,
                                metadata={"source": base_filename, "page_number": i + 1, "content_type": "pdf_page_text"}
                            ))
                            file_processed_successfully = True
                        
                        # Extract tables from the page
                        tables = page.extract_tables() # Returns a list of tables
                        if tables:
                            for j, table_data in enumerate(tables):
                                if table_data: # Ensure table_data is not empty
                                    # Assuming the first row of table_data might be headers if not explicitly provided
                                    markdown_table = table_to_markdown(table_data)
                                    if markdown_table.strip():
                                        all_docs_for_langchain.append(Document(
                                            page_content=markdown_table,
                                            metadata={"source": base_filename, "page_number": i + 1, "table_number": j + 1, "content_type": "table"}
                                        ))
                                        file_processed_successfully = True
                if file_processed_successfully:
                    loaded_file_count +=1
                else:
                    st.sidebar.warning(f"No text or tables extracted from PDF: {base_filename}")


        except Exception as e:
            st.sidebar.error(f"Error reading or processing {base_filename}: {e}")
            continue
    
    if not all_docs_for_langchain:
        st.error("No content could be loaded from any documents. Cannot build index.")
        if not os.path.exists(FAISS_INDEX_DIR):
             os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        dummy_doc = [Document(page_content="No valid document content loaded.")]
        empty_db = FAISS.from_documents(dummy_doc, _embedding_model)
        empty_db.save_local(FAISS_INDEX_DIR)
        save_manifest(current_file_states_tuple)
        st.sidebar.error("Created a dummy vector database as no valid content was loaded.")
        return empty_db

    st.sidebar.info(f"Processed {loaded_file_count} file(s), generating {len(all_docs_for_langchain)} document segments for embedding.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_docs = text_splitter.split_documents(all_docs_for_langchain)

    if not split_docs:
        st.error("Failed to split documents into chunks. Ensure documents have sufficient content.")
        if not os.path.exists(FAISS_INDEX_DIR):
             os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        dummy_doc = [Document(page_content="Failed to split documents.")]
        empty_db = FAISS.from_documents(dummy_doc, _embedding_model)
        empty_db.save_local(FAISS_INDEX_DIR)
        save_manifest(current_file_states_tuple)
        st.sidebar.error("Created a dummy vector database due to splitting failure.")
        return empty_db

    try:
        vectordb = FAISS.from_documents(split_docs, _embedding_model)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectordb.save_local(FAISS_INDEX_DIR)
        save_manifest(current_file_states_tuple)
        st.sidebar.success(f"FAISS index built from {len(all_docs_for_langchain)} document segments and saved to {FAISS_INDEX_DIR}")
        return vectordb
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        st.stop()
    st.sidebar.info(f"DEBUG: Total Langchain documents collected (all_docs_for_langchain): {len(all_docs_for_langchain)}")
    if all_docs_for_langchain:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### DEBUG: Sample Loaded Document Segments (First 3):")
        for i, doc_debug in enumerate(all_docs_for_langchain[:3]): # Display first 3 loaded segments
            st.sidebar.markdown(f"**Segment {i+1} Metadata:** `{doc_debug.metadata}`")
            st.sidebar.markdown(f"**Segment {i+1} Content Snippet (first 200 chars):**")
            st.sidebar.code(f"{doc_debug.page_content[:200]}...") # Show a snippet of the content
        st.sidebar.markdown("---")
    else:
        st.sidebar.warning("DEBUG: `all_docs_for_langchain` list is EMPTY after processing all files!")


@st.cache_resource
def get_llm(api_key):
    if not api_key:
        st.error("Google API Key not found for LLM.")
        st.stop()
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        st.stop()

def get_memory():
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )
    return st.session_state.conversation_memory

# --- Initialize components ---
if not os.path.exists(DOCS_DIRECTORY):
    os.makedirs(DOCS_DIRECTORY)
    st.sidebar.info(f"Created '{DOCS_DIRECTORY}' directory. Please add your supported files there.")

embedding_model = get_embeddings_model(GOOGLE_API_KEY)
current_doc_states = get_document_file_states(DOCS_DIRECTORY)

if current_doc_states:
    st.sidebar.markdown("### Monitored Documents:")
    for f_path, _ in current_doc_states:
        st.sidebar.markdown(f"- `{os.path.basename(f_path)}`")
else:
    st.sidebar.info(f"No documents found in '{DOCS_DIRECTORY}'. Add some {', '.join(SUPPORTED_EXTENSIONS)} files.")

vectordb = load_or_create_vector_db(embedding_model, current_doc_states)

if not vectordb:
    st.error("Vector DB could not be initialized. Please check errors above and ensure documents are present.")
    st.stop()

retriever = vectordb.as_retriever(search_kwargs={"k": 3}) # You might want to increase k if you have many small doc segments
llm = get_llm(GOOGLE_API_KEY)
memory = get_memory()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=False
)

# --- Chat UI ---
if "display_chat_history" not in st.session_state:
    st.session_state.display_chat_history = []

# Displaying historical messages
for msg_entry in st.session_state.display_chat_history:
    role_avatar = "🧑" if msg_entry["role"] == "user" else "🤖"
    with st.chat_message(name=msg_entry["role"], avatar=role_avatar):
        st.markdown(msg_entry["content"])
        if msg_entry["role"] == "ai" and "sources" in msg_entry and msg_entry["sources"]:
            with st.expander("View Sources", expanded=False):
                for i, source_doc in enumerate(msg_entry["sources"]):
                    source_name = source_doc.metadata.get('source', 'Unknown Source')
                    page_num_info = f" (Page {source_doc.metadata.get('page_number')})" if 'page_number' in source_doc.metadata else ""
                    content_type_info = f" [{source_doc.metadata.get('content_type', 'text')}]"
                    table_num_info = f" Table {source_doc.metadata.get('table_number')}" if 'table_number' in source_doc.metadata else ""
                    
                    score_info = "" # Placeholder for score if available from retriever
                    # If your retriever directly adds a score to metadata or as an attribute:
                    if 'score' in source_doc.metadata:
                         score_info = f"(Score: {source_doc.metadata['score']:.2f})"
                    elif hasattr(source_doc, 'score'): # FAISS retriever might add it as an attribute to DocumentWithScore
                         score_info = f"(Score: {source_doc.score:.2f})"

                    st.info(f"**Source {i+1}: {source_name}{page_num_info}{table_num_info}{content_type_info} {score_info}**\n\n```\n{source_doc.page_content}\n```")


# Handling new user input and AI response
user_input = st.chat_input(placeholder="Ask me anything about the loaded documents...")

if user_input:
    st.session_state.display_chat_history.append({"role": "user", "content": user_input})
    with st.chat_message(name="user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message(name="ai", avatar="🤖"):
        response_container = st.empty()
        stream_handler = StreamHandler(response_container)
        
        ai_response_content = "" 
        source_documents = []    

        input_data = {"question": user_input}
        try:
            result = qa_chain.invoke(input_data, callbacks=[stream_handler])
            
            ai_response_content = result.get("answer")
            if not ai_response_content and hasattr(stream_handler, 'text'):
                ai_response_content = stream_handler.text
            
            source_documents = result.get("source_documents", [])

            if ai_response_content:
                response_container.markdown(ai_response_content)
            else:
                default_message = "🤖 I don't have a specific answer for that based on the documents."
                response_container.markdown(default_message)
                ai_response_content = default_message

            if source_documents:
                with st.expander("View Sources", expanded=True): # Keep expanded for current message for immediate visibility
                    for i, source_doc in enumerate(source_documents):
                        source_name = source_doc.metadata.get('source', 'Unknown Source')
                        page_num_info = f" (Page {source_doc.metadata.get('page_number')})" if 'page_number' in source_doc.metadata else ""
                        content_type_info = f" [{source_doc.metadata.get('content_type', 'text')}]"
                        table_num_info = f" Table {source_doc.metadata.get('table_number')}" if 'table_number' in source_doc.metadata else ""
                        
                        score_info = ""
                        if 'score' in source_doc.metadata:
                             score_info = f"(Score: {source_doc.metadata['score']:.2f})"
                        elif hasattr(source_doc, 'score'):
                             score_info = f"(Score: {source_doc.score:.2f})"
                        st.info(f"**Source {i+1}: {source_name}{page_num_info}{table_num_info}{content_type_info} {score_info}**\n\n```\n{source_doc.page_content}\n```")


        except Exception as e:
            st.error(f"An error occurred: {e}") 
            error_message_content = "Sorry, I encountered an error while processing your request."
            response_container.markdown(error_message_content) 
            ai_response_content = error_message_content
            source_documents = [] # Ensure sources are empty on error

    st.session_state.display_chat_history.append({
        "role": "ai",
        "content": ai_response_content, 
        "sources": source_documents
    })
