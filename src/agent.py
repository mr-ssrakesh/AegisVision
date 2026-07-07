import os
import sys
import pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Reconfigure stdout for Windows console emoji support
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Load environment variables
load_dotenv()

# ==================== RAG RETRIEVAL ====================

def get_rag_context(defect_class: str) -> str:
    """Retrieve safety guidelines and manuals from the local vector database."""
    db_path = "models/vector_store.pkl"
    query = f"Safety inspection guidelines and repair protocols for {defect_class}"
    
    if os.path.exists(db_path):
        try:
            with open(db_path, "rb") as f:
                vector_store = pickle.load(f)
                
            # Perform similarity search (retrieve top 2 documents)
            docs = vector_store.similarity_search(query, k=2)
            context = "\n\n".join([doc.page_content for doc in docs])
            print(f"✅ Successfully retrieved RAG context for '{defect_class}' from local vector store.")
            return context
        except Exception as e:
            print(f"⚠️ Error retrieving from vector store: {e}. Falling back to default context.")
    else:
        print("⚠️ Vector store database not found. Falling back to default context.")
        
    # Hardcoded default guidelines in case vector db is not built yet
    fallbacks = {
        "crack": "[Structural Crack Fallback Guide] Inspect concrete cracks for depth. Epoxy injection is required for cracks exceeding 2mm. Wear safety boots, nitrile gloves, and face shield. Maintain LOTO protocols.",
        "rust": "[Surface Rust Fallback Guide] Sandblast corroded steel surfaces to white metal (SSPC-SP10). Apply zinc-rich epoxy primer (3.0 mils DFT) within 4 hours. Operators must wear a supplied-air blasting helmet.",
        "wear": "[Mechanical Wear Fallback Guide] Measure wear tolerances using digital micrometers. Compare to OEM limits (+/- 0.05mm). Replace component if limits are exceeded. Wear heavy-duty canvas safety gloves.",
        "normal": "[Normal State Fallback Guide] Standard Visual Check. Clear surface dust and apply anti-corrosion spray. Schedule next weekly inspection."
    }
    
    return fallbacks.get(defect_class, "Perform general safety checks and document current state.")

# ==================== MAINTENANCE AGENT CHAIN ====================

agent_prompt = PromptTemplate.from_template(
    "You are an expert Industrial Safety & Maintenance Engineer Agent.\n"
    "Your task is to draft a comprehensive, structured Maintenance & Repair Plan based on the visual defect analysis and safety manuals.\n\n"
    "================== INPUT METADATA ==================\n"
    "Classified Defect: {defect_class}\n"
    "Classification Confidence: {confidence:.2%}\n"
    "Assessed Severity: {severity}\n\n"
    "================== RETRIEVED MANUALS & DIRECTIVES ==================\n"
    "{rag_context}\n\n"
    "================== REPORT REQUIREMENTS ==================\n"
    "Draft a professional maintenance ticket. Include:\n"
    "1. **Job Ticket Details**: Defect category, severity flag, and CNN model accuracy score.\n"
    "2. **Safety Guidelines (OSHA/LOTO)**: List all required safety steps, lockouts, and PPE (Face shield, gloves, respirators) based on the manuals.\n"
    "3. **Step-by-Step Remediation Plan**: A bulleted procedure explaining how to repair/treat the defect based on the retrieved manuals.\n"
    "4. **Verification & Audit**: Instructions on how to check/measure the repair once completed.\n\n"
    "Write an exhaustive, high-quality plan in clean Markdown format. Do NOT include intro or outro chatter. Output ONLY the raw Markdown report."
)

def generate_maintenance_plan(defect_class: str, confidence: float, severity: str) -> str:
    """Execute RAG pipeline and compile the maintenance markdown plan."""
    context = get_rag_context(defect_class)
    
    # Check Groq API Key dynamically on invocation
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file or enter it in the sidebar.")
        
    # Initialize the LLM dynamically
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.2)
    except Exception:
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.2)
        
    # Build the dynamic agent chain
    agent_chain = agent_prompt | llm | StrOutputParser()
    
    report = agent_chain.invoke({
        "defect_class": defect_class,
        "confidence": confidence,
        "severity": severity,
        "rag_context": context
    })
    
    return report
