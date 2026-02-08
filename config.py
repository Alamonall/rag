from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR=PROJECT_ROOT / 'knowledge_base_new'
OUTPUT_DIR="./rag_data"

RAG_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
RAG_MODEL="./mistral-7b-instruct-v0.2.Q4_K_M.gguf"
PROMT_TYPE="standart" 

EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" # better accuracy 
    # Alternative English-optimized models:
    #sentence-transformers/all-MiniLM-L6-v2"  # Good for English
    # "sentence-transformers/all-mpnet-base-v2" - better accuracy
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" - multilingual
#"TinyLlama/TinyLlama-1.1B-Chat-v1.0"

CHUNKS_SIZE=500
CHUNK_OVERLAP=50

METADATA_FILE= "./rag_data/metadata.json"
FAISS_INDEX= "./rag_data/faiss.index"
EMBEDDING_FILE= "./rag_data/embeddings.npy"
