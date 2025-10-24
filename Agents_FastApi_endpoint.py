from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import sys
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Add Services to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Services'))

from Services.InventoryAgent import (
    DatabaseManager as InventoryDatabaseManager,
    AutoFileProcessor,
    InventoryChatbot,
    Config as InventoryConfig
)
from Services.MarketingAgent import (
    AdvancedDBManager as MarketingDatabaseManager,
    StateOfTheArtMarketingAI as MarketingChatbot,
    Config as MarketingConfig
)
from Services.FinanceAgent import (
    FinanceDatabaseManager,
    ContentExtractionAgent,
    RAGFinanceAgent
)

# Global variables for agents and initialization state
inventory_db = None
inventory_processor = None
inventory_chatbot = None
marketing_db = None
marketing_chatbot = None
finance_db = None
finance_extractor = None
finance_agent = None
_agents_initialized = False
_initialization_error = None
_initialization_lock = threading.Lock()

# Function to initialize agents in the background
def initialize_agents():
    global inventory_db, inventory_processor, inventory_chatbot
    global marketing_db, marketing_chatbot
    global finance_db, finance_extractor, finance_agent
    global _agents_initialized, _initialization_error
    
    logger.info("Starting agent initialization in background...")
    try:
        # Initialize Inventory Agent
        inventory_db = InventoryDatabaseManager(
            uri=InventoryConfig.MONGODB_URI,
            db_name=InventoryConfig.DB_NAME
        )
        inventory_processor = AutoFileProcessor(inventory_db)
        inventory_chatbot = InventoryChatbot(inventory_db)
        
        # Initialize Marketing Agent
        marketing_db = MarketingDatabaseManager()
        marketing_chatbot = MarketingChatbot(marketing_db)
        
        # Initialize Finance Agent
        finance_db = FinanceDatabaseManager(MarketingConfig.MONGODB_URI)
        finance_extractor = ContentExtractionAgent(MarketingConfig.OPENAI_API_KEY)
        finance_agent = RAGFinanceAgent(MarketingConfig.OPENAI_API_KEY, finance_db)
        
        with _initialization_lock:
            _agents_initialized = True
            _initialization_error = None
        logger.info("✅ All agents initialized successfully")
    except Exception as e:
        with _initialization_lock:
            _agents_initialized = False
            _initialization_error = str(e)
        logger.error(f"❌ Error initializing agents: {str(e)}")
        raise

# FastAPI lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start agent initialization in a background task
    background_tasks = BackgroundTasks()
    background_tasks.add_task(initialize_agents)
    
    # Run background tasks manually to ensure they start
    import asyncio
    asyncio.create_task(background_tasks())
    
    logger.info("Application startup complete, agent initialization running in background")
    yield
    # Cleanup on shutdown
    logger.info("Shutting down...")
    with _initialization_lock:
        global _agents_initialized
        _agents_initialized = False

app = FastAPI(
    title="Multi-Agent System API",
    version="1.0.0",
    timeout=300,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job tracking system
job_status = {}
job_lock = threading.Lock()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    agent: str

class ChatResponse(BaseModel):
    response: str
    agent: str

class StatsResponse(BaseModel):
    inventory: Dict[str, Any]
    marketing: Dict[str, Any]
    finance: Dict[str, Any]

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    progress: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Helper functions for job tracking
def create_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())
    with job_lock:
        job_status[job_id] = {
            "status": "processing",
            "progress": "Starting...",
            "type": job_type,
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }
    return job_id

def update_job_progress(job_id: str, progress: str):
    with job_lock:
        if job_id in job_status:
            job_status[job_id]["progress"] = progress

def complete_job(job_id: str, result: Dict[str, Any]):
    with job_lock:
        if job_id in job_status:
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["progress"] = "Completed"
            job_status[job_id]["result"] = result

def fail_job(job_id: str, error: str):
    with job_lock:
        if job_id in job_status:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["progress"] = "Failed"
            job_status[job_id]["error"] = error

# Background task for inventory processing
def process_inventory_background(job_id: str, file_bytes: bytes, filename: str):
    try:
        update_job_progress(job_id, "📄 Validating file...")
        
        update_job_progress(job_id, "🔤 Extracting text from PDF...")
        
        update_job_progress(job_id, "🤖 Analyzing with AI (this may take 30-60 seconds)...")
        result = inventory_processor.process_single_file(file_bytes, filename)
        
        update_job_progress(job_id, "💾 Storing results...")
        
        if result.get('errors'):
            error_info = result['errors'][0]
            fail_job(job_id, error_info.get('error', 'Unknown error'))
        elif result.get('skipped'):
            complete_job(job_id, {
                "success": False,
                "message": "File already processed (duplicate detected)",
                "result": result
            })
        else:
            complete_job(job_id, {
                "success": True,
                "message": "File processed successfully",
                "result": result
            })
    except Exception as e:
        fail_job(job_id, str(e))

# Background task for finance processing
def process_finance_background(job_id: str, file_bytes: bytes, filename: str):
    try:
        update_job_progress(job_id, "📸 Analyzing image...")
        
        update_job_progress(job_id, "🤖 Extracting data with AI (this may take 30-60 seconds)...")
        extracted_data = finance_extractor.extract_content(file_bytes, filename)
        
        update_job_progress(job_id, "💾 Storing in database...")
        success, message = finance_db.store_document(extracted_data)
        
        if success:
            complete_job(job_id, {
                "success": True,
                "message": message,
                "document_type": extracted_data.get('document_type', 'unknown'),
                "metadata": extracted_data.get('_metadata', {})
            })
        else:
            fail_job(job_id, message)
    except Exception as e:
        fail_job(job_id, str(e))

# Readiness endpoint
@app.get("/ready")
async def readiness_check():
    """Check if all agents are initialized"""
    with _initialization_lock:
        if _agents_initialized:
            return {"status": "ready", "message": "All agents initialized"}
        elif _initialization_error:
            raise HTTPException(status_code=503, detail=f"Agent initialization failed: {_initialization_error}")
        else:
            raise HTTPException(status_code=503, detail="Agents are still initializing")

# Health check
@app.get("/")
async def root():
    return {
        "status": "running",
        "agents": ["inventory", "marketing", "finance"],
        "version": "1.0.0"
    }

# ============================================================================
# JOB STATUS ENDPOINT
# ============================================================================

@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check the status of a processing job"""
    with job_lock:
        if job_id not in job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = job_status[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            result=job.get("result"),
            error=job.get("error")
        )

# ============================================================================
# INVENTORY ENDPOINTS
# ============================================================================

@app.post("/inventory/upload", response_model=JobResponse)
async def upload_inventory_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload PDF file for inventory processing - Returns job_id for tracking"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized yet")
        if inventory_processor is None:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Create job
        job_id = create_job("inventory_upload")
        
        # Start background processing
        background_tasks.add_task(process_inventory_background, job_id, file_bytes, file.filename)
        
        return JobResponse(
            job_id=job_id,
            status="processing",
            message=f"File upload started. Use /job/{job_id} to check progress"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inventory/chat", response_model=ChatResponse)
async def inventory_chat(request: ChatRequest):
    """Chat with inventory agent"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized yet")
        if inventory_chatbot is None:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized")
    try:
        response = inventory_chatbot.chat(request.message)
        return ChatResponse(response=response, agent="inventory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inventory/stats")
async def get_inventory_stats():
    """Get inventory statistics"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Inventory agent not initialized yet")
        if inventory_db is None:
            raise HTTPException(status_code=503, detail="Inventory database not initialized")
    try:
        summary = inventory_db.get_inventory_summary(days=90)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MARKETING ENDPOINTS
# ============================================================================

@app.post("/marketing/upload")
async def upload_marketing_file(file: UploadFile = File(...)):
    """Upload Facebook Ads file (CSV/Excel) - Fast processing"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Marketing agent not initialized yet")
        if marketing_db is None:
            raise HTTPException(status_code=503, detail="Marketing database not initialized")
    try:
        if not (file.filename.endswith('.csv') or 
                file.filename.endswith('.xlsx') or 
                file.filename.endswith('.xls') or
                file.filename.endswith('.tsv')):
            raise HTTPException(
                status_code=400, 
                detail="Only CSV, TSV and Excel files are supported"
            )
        
        file_bytes = await file.read()
        success, message = marketing_db.save_facebook_ads(file_bytes, file.filename)
        
        if success:
            return {"success": True, "message": message}
        else:
            raise HTTPException(status_code=400, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/marketing/chat", response_model=ChatResponse)
async def marketing_chat(request: ChatRequest):
    """Chat with marketing agent"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Marketing agent not initialized yet")
        if marketing_chatbot is None:
            raise HTTPException(status_code=503, detail="Marketing agent not initialized")
    try:
        response, chart_data = marketing_chatbot.chat(request.message)
        return ChatResponse(response=response, agent="marketing")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/marketing/stats")
async def get_marketing_stats():
    """Get marketing statistics"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Marketing agent not initialized yet")
        if marketing_db is None:
            raise HTTPException(status_code=503, detail="Marketing database not initialized")
    try:
        fb_ads = marketing_db.get_facebook_ads()
        inventory = marketing_db.get_inventory_data(days=90)
        finance = marketing_db.get_finance_data()
        
        stats = {
            "facebook_ads": {},
            "campaigns_count": 0,
            "inventory_value": inventory.get('total_value', 0),
            "bank_balance": finance.get('bank_balance', 0),
            "cash_flow": finance.get('cash_flow', 0)
        }
        
        if fb_ads:
            latest_ad = fb_ads[0]
            analysis = latest_ad.get('analysis', {})
            stats["facebook_ads"] = analysis.get('totals', {})
            stats["campaigns_count"] = len(analysis.get('campaigns', []))
            stats["best_campaign"] = analysis.get('best_campaign', {})
            stats["worst_campaign"] = analysis.get('worst_campaign', {})
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FINANCE ENDPOINTS
# ============================================================================

@app.post("/finance/upload", response_model=JobResponse)
async def upload_finance_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload financial document image - Returns job_id immediately"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Finance agent not initialized yet")
        if finance_extractor is None or finance_db is None:
            raise HTTPException(status_code=503, detail="Finance agent not initialized")
    try:
        if not (file.filename.lower().endswith('.jpg') or 
                file.filename.lower().endswith('.jpeg') or 
                file.filename.lower().endswith('.png')):
            raise HTTPException(
                status_code=400, 
                detail="Only JPG and PNG image files are supported"
            )
        
        # Read file content
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Create job FIRST
        job_id = create_job("finance_upload")
        
        # Add background task AFTER creating job
        background_tasks.add_task(process_finance_background, job_id, file_bytes, file.filename)
        
        # Return IMMEDIATELY
        return JobResponse(
            job_id=job_id,
            status="processing",
            message=f"Upload successful! Processing started. Check progress at: GET /job/{job_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/finance/chat", response_model=ChatResponse)
async def finance_chat(request: ChatRequest):
    """Chat with finance agent"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Finance agent not initialized yet")
        if finance_agent is None:
            raise HTTPException(status_code=503, detail="Finance agent not initialized")
    try:
        response = finance_agent.query(request.message)
        return ChatResponse(response=response, agent="finance")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/finance/stats")
async def get_finance_stats():
    """Get finance statistics"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Finance agent not initialized yet")
        if finance_db is None:
            raise HTTPException(status_code=503, detail="Finance database not initialized")
    try:
        all_docs = finance_db.get_all_documents()
        
        total_balance = 0
        total_invoices = 0
        
        for statement in all_docs.get('bank_statements', []):
            balances = statement.get('balances', {})
            total_balance += balances.get('current_balance', 0)
        
        for invoice in all_docs.get('invoices', []):
            totals = invoice.get('totals', {})
            total_invoices += totals.get('total', 0)
        
        return {
            "bank_statements_count": len(all_docs.get('bank_statements', [])),
            "invoices_count": len(all_docs.get('invoices', [])),
            "price_lists_count": len(all_docs.get('price_lists', [])),
            "other_documents_count": len(all_docs.get('documents', [])),
            "total_balance": total_balance,
            "total_invoice_value": total_invoices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/finance/documents")
async def get_finance_documents():
    """Get all finance documents"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Finance agent not initialized yet")
        if finance_db is None:
            raise HTTPException(status_code=503, detail="Finance database not initialized")
    try:
        all_docs = finance_db.get_all_documents()
        return all_docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UNIFIED ENDPOINTS
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def unified_chat(request: ChatRequest):
    """Unified chat endpoint that routes to the correct agent"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail=f"{request.agent} agent not initialized yet")
    try:
        if request.agent == "inventory":
            if inventory_chatbot is None:
                raise HTTPException(status_code=503, detail="Inventory agent not initialized")
            response = inventory_chatbot.chat(request.message)
        elif request.agent == "marketing":
            if marketing_chatbot is None:
                raise HTTPException(status_code=503, detail="Marketing agent not initialized")
            response, _ = marketing_chatbot.chat(request.message)
        elif request.agent == "finance":
            if finance_agent is None:
                raise HTTPException(status_code=503, detail="Finance agent not initialized")
            response = finance_agent.query(request.message)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid agent. Choose: inventory, marketing, or finance"
            )
        
        return ChatResponse(response=response, agent=request.agent)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_all_stats():
    """Get statistics from all agents"""
    with _initialization_lock:
        if not _agents_initialized:
            raise HTTPException(status_code=503, detail="Agents not initialized yet")
        if any(x is None for x in [inventory_db, marketing_db, finance_db]):
            raise HTTPException(status_code=503, detail="One or more agents not initialized")
    try:
        inventory_stats = inventory_db.get_inventory_summary(days=90)
        
        fb_ads = marketing_db.get_facebook_ads()
        marketing_stats = {
            "total_campaigns": 0,
            "total_spend": 0,
            "total_impressions": 0,
            "total_clicks": 0
        }
        
        if fb_ads:
            latest = fb_ads[0]
            analysis = latest.get('analysis', {})
            totals = analysis.get('totals', {})
            marketing_stats = {
                "total_campaigns": len(analysis.get('campaigns', [])),
                "total_spend": totals.get('total_spend', 0),
                "total_impressions": totals.get('total_impressions', 0),
                "total_clicks": totals.get('total_clicks', 0),
                "avg_ctr": totals.get('avg_ctr', 0),
                "avg_cpc": totals.get('avg_cpc', 0)
            }
        
        all_finance_docs = finance_db.get_all_documents()
        finance_stats = {
            "bank_statements": len(all_finance_docs.get('bank_statements', [])),
            "invoices": len(all_finance_docs.get('invoices', [])),
            "price_lists": len(all_finance_docs.get('price_lists', [])),
            "total_documents": (
                len(all_finance_docs.get('bank_statements', [])) +
                len(all_finance_docs.get('invoices', [])) +
                len(all_finance_docs.get('price_lists', [])) +
                len(all_finance_docs.get('documents', []))
            )
        }
        
        return StatsResponse(
            inventory=inventory_stats,
            marketing=marketing_stats,
            finance=finance_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Detailed health check for all agents"""
    health = {
        "status": "healthy",
        "agents": {}
    }
    
    with _initialization_lock:
        if not _agents_initialized:
            health["status"] = "degraded"
            health["agents"]["inventory"] = "unhealthy: not initialized yet"
            health["agents"]["marketing"] = "unhealthy: not initialized yet"
            health["agents"]["finance"] = "unhealthy: not initialized yet"
            if _initialization_error:
                health["error"] = f"Initialization failed: {_initialization_error}"
            return health
    
    try:
        if inventory_db is None:
            raise Exception("Inventory database not initialized")
        inventory_db.get_inventory_summary(days=1)
        health["agents"]["inventory"] = "healthy"
    except Exception as e:
        health["agents"]["inventory"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"
    
    try:
        if marketing_db is None:
            raise Exception("Marketing database not initialized")
        marketing_db.get_facebook_ads()
        health["agents"]["marketing"] = "healthy"
    except Exception as e:
        health["agents"]["marketing"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"
    
    try:
        if finance_db is None:
            raise Exception("Finance database not initialized")
        finance_db.get_all_documents()
        health["agents"]["finance"] = "healthy"
    except Exception as e:
        health["agents"]["finance"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"
    
    return health

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    import time

    load_dotenv()

    port = int(os.getenv("PORT", "8000"))  # Default to 8000 for Render
    logger.info(f"Starting Uvicorn on 0.0.0.0:{port}")
    start_time = time.time()
    try:
        uvicorn.run("Agents_FastApi_endpoint:app", host="0.0.0.0", port=port, log_level="info", workers=1)
    except Exception as e:
        logger.error(f"Failed to start Uvicorn: {str(e)}")
        raise
    finally:
        logger.info(f"Uvicorn startup took {time.time() - start_time:.2f} seconds")
