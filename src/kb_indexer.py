import os
import sys
import pickle
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Reconfigure stdout for Windows console emoji support
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 1. Define detailed industrial guidelines and manuals
MANUALS = [
    # --- CRACK MANUALS ---
    Document(
        page_content="[Structural Crack - Inspection Standard] Directives under OSHA 1926.502: Any crack exceeding 2.0mm width in load-bearing concrete structural elements (beams, columns, slabs) requires immediate load suspension and barricading. Visually inspect for depth, spalling, and moisture ingress.",
        metadata={"category": "crack", "type": "inspection"}
    ),
    Document(
        page_content="[Structural Crack - Repair Protocol] Epoxy Resin Injection: Clean the crack area using high-pressure air. Install injection ports along the crack line at 6-inch intervals. Seal the surface with an epoxy gel paste (wait 24 hours to cure). Inject low-viscosity epoxy resin starting from the lowest port until it flows from the next highest port. Seal ports after completion.",
        metadata={"category": "crack", "type": "repair"}
    ),
    Document(
        page_content="[Structural Crack - Safety Requirements] Safety Gear & Hazard mitigation: Personnel performing epoxy injections must wear steel-toed safety boots, chemical-resistant nitrile gloves, and a full-face impact shield. Ensure adequate ventilation in confined spaces to avoid chemical vapor inhalation.",
        metadata={"category": "crack", "type": "safety"}
    ),
    
    # --- RUST MANUALS ---
    Document(
        page_content="[Surface Rust - Inspection Standard] Corrosion Rating: Assess rust coverage on steel components. Level A (light rust scale), Level B (rust pitting/wall loss). Measure remaining metal thickness using an ultrasonic thickness gauge to ensure it is above structural tolerance limits.",
        metadata={"category": "rust", "type": "inspection"}
    ),
    Document(
        page_content="[Surface Rust - Treatment Protocol] Sandblasting & Coating: Isolate the component. Perform abrasive sandblasting to Near-White Metal Blast Cleaning standard (SSPC-SP10 / NACE No. 2). Remove all rust, mill scale, and paint. Within 4 hours of blasting, apply a zinc-rich epoxy primer coating to a minimum dry film thickness (DFT) of 3.0 mils. Apply top coat once primer cures.",
        metadata={"category": "rust", "type": "repair"}
    ),
    Document(
        page_content="[Surface Rust - Safety Requirements] Lockout-Tagout (LOTO) and Air filtration: Verify LOTO is active on all adjacent moving machinery. Sandblasting operators must wear a supplied-air respirator hood (Type CE), safety boots, and heavy leather blasting gloves. Collect abrasive run-off.",
        metadata={"category": "rust", "type": "safety"}
    ),
    
    # --- WEAR MANUALS ---
    Document(
        page_content="[Mechanical Wear - Inspection Standard] Tolerance Audit: Inspect gears, shafts, and bearing surfaces for scoring and abrasion. Measure wear tolerance using digital outside micrometers. Compare measurements against original equipment manufacturer (OEM) design blueprints (tolerance limit: +/- 0.05mm).",
        metadata={"category": "wear", "type": "inspection"}
    ),
    Document(
        page_content="[Mechanical Wear - Refurbishment Protocol] Component replacement: If the wear exceeds OEM tolerance (+/- 0.05mm), schedule immediate component replacement. If within tolerance, polish the surface using high-grit emery cloth, flush bearing housings with clean lubricating oil to remove metal shavings, and refill with high-viscosity synthetic gear lubricant (ISO VG 220).",
        metadata={"category": "wear", "type": "repair"}
    ),
    Document(
        page_content="[Mechanical Wear - Safety Requirements] Lockout-Tagout (LOTO) & Hand protection: Complete LOTO procedure on the main electrical panel before removing protective casings. Wear heavy-duty canvas safety gloves to protect hands from sharp metal edges and hot gear components.",
        metadata={"category": "wear", "type": "safety"}
    ),
    
    # --- NORMAL MANUALS ---
    Document(
        page_content="[Normal State - Preventive Maintenance] Standard Inspection: Surface shows no significant crack, corrosion, or mechanical wear. Continue regular weekly visual inspections. Clean the component surface using clean lint-free cloths and apply standard anti-dust protective spray.",
        metadata={"category": "normal", "type": "preventive"}
    )
]

def index_knowledge_base():
    """Build and save the local vector store."""
    print("📥 Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("🗄️ Initializing In-Memory Vector Store and indexing documents...")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(MANUALS)
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save the database locally using pickle
    save_path = "models/vector_store.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(vector_store, f)
        
    print(f"💾 Local RAG Vector database successfully saved to {save_path}!")

if __name__ == "__main__":
    index_knowledge_base()
