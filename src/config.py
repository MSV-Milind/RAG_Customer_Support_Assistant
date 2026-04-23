import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  

# Project Paths 
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOGS_DIR   = BASE_DIR / "logs"

# Groq LLM 
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.3-70b-versatile"   
GROQ_TEMP      = 0.0                
GROQ_MAX_TOKENS = 512

# Embedding Model 
EMBED_MODEL = "all-MiniLM-L6-v2"   

# Chunking 
CHUNK_SIZE    = 700   
CHUNK_OVERLAP = 75    

# Retrieval 
TOP_K               = 4     
SIMILARITY_THRESHOLD = 0.30  
COLLECTION_NAME     = "rag_support_kb"

# LangGraph State Keys 
STATE_QUERY    = "query"
STATE_CONTEXT  = "context"
STATE_RESPONSE = "response"
STATE_ESCALATE = "escalate"

# Routing Keywords 
ESCALATE_PREFIX = "ESCALATE"
FINAL_PREFIX    = "FINAL ANSWER:"
