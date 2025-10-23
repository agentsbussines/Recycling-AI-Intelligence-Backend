import os
import json
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pymongo
from pymongo import MongoClient
import hashlib
import base64
from pathlib import Path
import threading

# PDF Processing
try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("Please install: pip install PyPDF2 pdfplumber")

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install: pip install sentence-transformers")

# LangChain & OpenAI
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    from langchain.chains import LLMChain
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field, validator
    import openai
except ImportError:
    print("Please install: pip install langchain langchain-openai pydantic openai")

# Language detection
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("Please install: pip install langdetect")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    MONGODB_URI = "mongodb+srv://agentsbussines:Pakistan21@cluster0.ddj675b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DB_NAME = "InventoryAgentDB"
    COLLECTION_EMBEDDINGS = "embeddings"
    COLLECTION_INVENTORY = "inventory_records"
    COLLECTION_METADATA = "file_metadata"
    COLLECTION_CONVERSATIONS = "chat_history"
    COLLECTION_LEARNED_PATTERNS = "learned_patterns"
    DATA_FOLDER = "./Data"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
    LLM_MODEL = "gpt-4o-mini"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150

# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class InventoryItem(BaseModel):
    metal_name: str = Field(description="Name of the metal or material")
    unit: str = Field(description="Unit of measurement (e.g., Kilogramo, Gramo, Pieza)")
    quantity: float = Field(description="Quantity amount")
    unit_price: float = Field(description="Price per unit")
    total_cost: float = Field(description="Total cost (quantity * unit_price)")

class DocumentMetadata(BaseModel):
    date: str = Field(description="Document date in YYYY-MM-DD format")
    concept: str = Field(description="Document concept (e.g., Compra, Venta, Inventario)")
    owner: Optional[str] = Field(description="Owner or business name if mentioned")
    total_items: int = Field(description="Total number of items in document")
    total_amount: float = Field(description="Total amount/value from document")
    currency: str = Field(default="MXN", description="Currency code")

class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    items: List[InventoryItem]
    raw_text_summary: str = Field(description="Brief summary of document content")

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self._create_indexes()
    
    def _create_indexes(self):
        self.db[Config.COLLECTION_EMBEDDINGS].create_index([("file_hash", 1)])
        self.db[Config.COLLECTION_METADATA].create_index([("file_path", 1)])
        self.db[Config.COLLECTION_INVENTORY].create_index([("metadata.date", -1)])
        self.db[Config.COLLECTION_LEARNED_PATTERNS].create_index([("pattern_type", 1)])
    
    def file_already_processed(self, file_path: str, file_hash: str) -> bool:
        return self.db[Config.COLLECTION_METADATA].find_one({
            "file_path": file_path,
            "file_hash": file_hash
        }) is not None
    
    def store_inventory(self, data: Dict):
        self.db[Config.COLLECTION_INVENTORY].insert_one(data)
    
    def store_embeddings(self, embeddings: List[Dict]):
        if embeddings:
            self.db[Config.COLLECTION_EMBEDDINGS].insert_many(embeddings)
    
    def store_metadata(self, metadata: Dict):
        self.db[Config.COLLECTION_METADATA].insert_one(metadata)
    
    def search_embeddings(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        all_docs = list(self.db[Config.COLLECTION_EMBEDDINGS].find())
        
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0
        
        for doc in all_docs:
            doc['similarity'] = cosine_sim(query_embedding, doc['embedding'])
        
        all_docs.sort(key=lambda x: x['similarity'], reverse=True)
        return all_docs[:top_k]
    
    def query_inventory(self, filters: Dict = None, sort_by: str = "metadata.date", limit: int = 100) -> List[Dict]:
        query = filters or {}
        return list(self.db[Config.COLLECTION_INVENTORY]
                   .find(query)
                   .sort(sort_by, -1)
                   .limit(limit))
    
    def get_inventory_summary(self, days: int = None, date_range: Tuple = None) -> Dict:
        query = {}
        
        if days:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            query["metadata.date"] = {"$gte": start_date}
        elif date_range:
            query["metadata.date"] = {"$gte": date_range[0], "$lte": date_range[1]}
        
        inventories = list(self.db[Config.COLLECTION_INVENTORY].find(query))
        
        total_items = sum(len(inv.get('items', [])) for inv in inventories)
        total_value = sum(inv.get('metadata', {}).get('total_amount', 0) for inv in inventories)
        
        metal_stats = {}
        for inv in inventories:
            for item in inv.get('items', []):
                metal = item.get('metal_name', 'Unknown')
                qty = item.get('quantity', 0)
                cost = item.get('total_cost', 0)
                
                if metal not in metal_stats:
                    metal_stats[metal] = {"quantity": 0, "value": 0, "count": 0}
                
                metal_stats[metal]["quantity"] += qty
                metal_stats[metal]["value"] += cost
                metal_stats[metal]["count"] += 1
        
        return {
            "total_records": len(inventories),
            "total_items": total_items,
            "total_value": total_value,
            "metal_breakdown": metal_stats,
            "date_range": query.get("metadata.date", "all")
        }
    
    def save_conversation(self, user_id: str, message: str, response: str, language: str):
        self.db[Config.COLLECTION_CONVERSATIONS].insert_one({
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "ai_response": response,
            "language": language
        })
    
    def store_learned_pattern(self, pattern_type: str, pattern_data: Dict):
        self.db[Config.COLLECTION_LEARNED_PATTERNS].update_one(
            {"pattern_type": pattern_type},
            {"$set": {
                "pattern_data": pattern_data,
                "last_updated": datetime.now().isoformat()
            }},
            upsert=True
        )
    
    def get_learned_patterns(self, pattern_type: str = None) -> List[Dict]:
        query = {"pattern_type": pattern_type} if pattern_type else {}
        return list(self.db[Config.COLLECTION_LEARNED_PATTERNS].find(query))

# =============================================================================
# LLM-DRIVEN PDF PARSER
# =============================================================================

class LLMDocumentParser:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=ParsedDocument)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
        except Exception as e:
            print(f"pdfplumber failed: {e}, trying PyPDF2...")
        
        if not text.strip():
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Failed to extract text: {e}")
        
        return text
    
    def parse_document_with_llm(self, text: str, filename: str) -> ParsedDocument:
        try:
            format_instructions = self.parser.get_format_instructions()
        except Exception as e:
            format_instructions = """
Return a JSON object with this structure:
{
  "metadata": {
    "date": "YYYY-MM-DD",
    "concept": "Compra/Venta/etc",
    "owner": "Name or null",
    "total_items": number,
    "total_amount": number,
    "currency": "MXN"
  },
  "items": [
    {
      "metal_name": "string",
      "unit": "string", 
      "quantity": number,
      "unit_price": number,
      "total_cost": number
    }
  ],
  "raw_text_summary": "brief summary"
}
"""
        
        parsing_prompt = ChatPromptTemplate.from_template("""
You are an expert at understanding inventory and purchase documents in ANY format and language.

Your task: Analyze the document text below and extract ALL relevant information.

IMPORTANT INSTRUCTIONS:
1. **Be Dynamic**: Don't assume any specific format. Understand the document naturally.
2. **Extract All Metals/Materials**: ANY item mentioned (metals, batteries, scraps, materials) - extract them all
3. **Smart Date Detection**: Find dates in ANY format (Spanish, English, any format)
4. **Understand Context**: Identify if it's a purchase (Compra), sale (Venta), or inventory report
5. **Calculate Totals**: Sum up all items to get total amount
6. **Handle Missing Data**: If unit price or total isn't explicitly stated, try to calculate or estimate
7. **Be Flexible with Units**: Understand Kilogramo, Gramo, Pieza, Tonelada, Libra, etc.

Document Filename: {filename}

Document Text:
{document_text}

{format_instructions}

Extract ALL information accurately. Be thorough and intelligent.
Return ONLY valid JSON, no extra text.
""")
        
        chain = parsing_prompt | self.llm
        
        try:
            response = chain.invoke({
                "document_text": text[:4000],
                "filename": filename,
                "format_instructions": format_instructions
            })
            
            try:
                parsed_doc = self.parser.parse(response.content)
            except Exception as parse_error:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group())
                    parsed_doc = ParsedDocument(**json_data)
                else:
                    raise parse_error
            
            return parsed_doc
            
        except Exception as e:
            print(f"LLM parsing failed for {filename}: {str(e)[:100]}... Using fallback.")
            return self._create_fallback_structure(text, filename)
    
    def _create_fallback_structure(self, text: str, filename: str) -> ParsedDocument:
        return ParsedDocument(
            metadata=DocumentMetadata(
                date=datetime.now().strftime("%Y-%m-%d"),
                concept="Unknown",
                owner=None,
                total_items=0,
                total_amount=0.0,
                currency="MXN"
            ),
            items=[],
            raw_text_summary=text[:500]
        )
    
    def validate_and_enhance(self, parsed_doc: ParsedDocument, db_manager: DatabaseManager) -> ParsedDocument:
        learned_patterns = db_manager.get_learned_patterns()
        
        validation_prompt = f"""
Review this parsed inventory document and validate/enhance it:

Parsed Data:
{parsed_doc.model_dump_json(indent=2)}

Historical Patterns (for reference):
{json.dumps([p.get('pattern_data', {}) for p in learned_patterns[:5]], indent=2)}

Tasks:
1. Check if dates make sense (not in future, reasonable format)
2. Verify calculations (quantity * unit_price = total_cost)
3. Standardize metal names (e.g., "ALUMINIO DELGADO" and "Aluminio Delgado" should be same)
4. Flag any suspicious data
5. Enhance with any missing information you can infer

Return the validated/enhanced data in the same JSON structure.
If everything looks good, return it as-is.
"""
        
        return parsed_doc

# =============================================================================
# INTELLIGENT CHATBOT AGENT
# =============================================================================

class InventoryChatbot:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        if not Config.OPENAI_API_KEY:
            print("⚠️ Please set OPENAI_API_KEY environment variable")
            raise SystemExit
        
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.3,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return 'en'
    
    def query_inventory_with_llm(self, user_question: str, user_language: str) -> str:
        understanding_prompt = f"""
You are an AI that understands inventory questions.

User asked (in {user_language}): "{user_question}"

Analyze this question and determine:
1. What type of query is this? (date-specific, metal-specific, summary, comparison, etc.)
2. What parameters are needed? (dates, metal names, time periods, etc.)
3. What data should be fetched from database?

Return a JSON with:
{{
    "query_type": "date_specific|metal_search|summary|top_items|comparison|general",
    "parameters": {{
        "date": "YYYY-MM-DD or null",
        "date_range": ["start", "end"] or null,
        "metal_name": "name or null",
        "days": number or null,
        "sort_by": "quantity|value|date",
        "limit": number
    }},
    "intent": "brief description of what user wants"
}}
"""
        
        response = self.llm.invoke(understanding_prompt)
        
        try:
            query_plan = json.loads(response.content)
        except:
            query_plan = {
                "query_type": "general",
                "parameters": {},
                "intent": user_question
            }
        
        relevant_data = self._fetch_data_based_on_plan(query_plan)
        
        response_prompt = f"""
You are a helpful inventory assistant. Respond in {user_language} language.

User Question: {user_question}

Relevant Data from Database:
{json.dumps(relevant_data, indent=2, default=str)}

Instructions:
1. Answer the user's question accurately using the data provided
2. Be conversational and friendly
3. Format numbers clearly (use commas, specify units)
4. If data is missing, politely explain what's not available
5. Respond in {user_language} language
6. Don't mention technical terms like "database" or "query"
7. Organize information in a clear, easy-to-read format

Generate a helpful response:
"""
        
        final_response = self.llm.invoke(response_prompt)
        
        return final_response.content
    
    def _fetch_data_based_on_plan(self, query_plan: Dict) -> Dict:
        query_type = query_plan.get("query_type", "general")
        params = query_plan.get("parameters", {})
        
        result = {"query_type": query_type, "data": []}
        
        try:
            if query_type == "date_specific":
                date = params.get("date")
                if date:
                    inventories = self.db.query_inventory({"metadata.date": date})
                    result["data"] = [self._format_inventory_record(inv) for inv in inventories]
            
            elif query_type == "metal_search":
                metal_name = params.get("metal_name", "")
                all_inv = self.db.query_inventory()
                
                matching_items = []
                for inv in all_inv:
                    for item in inv.get('items', []):
                        if metal_name.lower() in item.get('metal_name', '').lower():
                            matching_items.append({
                                "date": inv.get('metadata', {}).get('date'),
                                "metal": item.get('metal_name'),
                                "quantity": item.get('quantity'),
                                "unit": item.get('unit'),
                                "total_cost": item.get('total_cost')
                            })
                
                result["data"] = matching_items
            
            elif query_type == "summary":
                days = params.get("days")
                summary = self.db.get_inventory_summary(days=days)
                result["data"] = summary
            
            elif query_type == "top_items":
                summary = self.db.get_inventory_summary()
                metal_breakdown = summary.get('metal_breakdown', {})
                
                sort_by = params.get("sort_by", "quantity")
                limit = params.get("limit", 5)
                
                if sort_by == "value":
                    sorted_metals = sorted(metal_breakdown.items(), 
                                         key=lambda x: x[1]['value'], 
                                         reverse=True)
                else:
                    sorted_metals = sorted(metal_breakdown.items(), 
                                         key=lambda x: x[1]['quantity'], 
                                         reverse=True)
                
                result["data"] = [
                    {
                        "metal": metal,
                        "quantity": stats['quantity'],
                        "value": stats['value'],
                        "count": stats['count']
                    }
                    for metal, stats in sorted_metals[:limit]
                ]
            
            else:
                recent = self.db.query_inventory(limit=10)
                result["data"] = [self._format_inventory_record(inv) for inv in recent]
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _format_inventory_record(self, inv: Dict) -> Dict:
        return {
            "date": inv.get('metadata', {}).get('date'),
            "concept": inv.get('metadata', {}).get('concept'),
            "total_amount": inv.get('metadata', {}).get('total_amount'),
            "items_count": len(inv.get('items', [])),
            "items": [
                {
                    "metal": item.get('metal_name'),
                    "quantity": item.get('quantity'),
                    "unit": item.get('unit'),
                    "unit_price": item.get('unit_price'),
                    "total": item.get('total_cost')
                }
                for item in inv.get('items', [])[:10]
            ]
        }
    
    def chat(self, user_message: str) -> str:
        try:
            user_lang = self.detect_language(user_message)
            response = self.query_inventory_with_llm(user_message, user_lang)
            
            self.db.save_conversation(
                user_id="default",
                message=user_message,
                response=response,
                language=user_lang
            )
            
            return response
        
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}\n\nPlease try rephrasing your question."

# =============================================================================
# AUTO FILE PROCESSOR
# =============================================================================

class AutoFileProcessor:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.llm_parser = LLMDocumentParser(Config.OPENAI_API_KEY)
        self.processing = False
    
    def process_single_file(self, file_bytes: bytes, filename: str) -> Dict:
        """Process a single uploaded PDF file from bytes - OPTIMIZED VERSION"""
        if self.processing:
            return {"status": "already_processing", "errors": [], "processed": [], "skipped": []}
        
        self.processing = True
        results = {"processed": [], "skipped": [], "errors": []}
        
        try:
            print(f"📄 Starting processing: {filename}")
            
            # Calculate file hash for duplicate detection
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            print(f"🔍 File hash: {file_hash[:16]}...")
            
            # Check if already processed
            if self.db.file_already_processed(filename, file_hash):
                print(f"⏭️  File already processed: {filename}")
                results["skipped"].append(filename)
                return results
            
            try:
                # Create a temporary file to work with PDF libraries
                print("📝 Creating temporary file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(file_bytes)
                    temp_file_path = temp_file.name
                
                try:
                    # Extract text from PDF
                    print("🔤 Extracting text from PDF...")
                    raw_text = self.llm_parser.extract_text_from_pdf(temp_file_path)
                    
                    if not raw_text.strip():
                        print("❌ No text extracted from PDF")
                        results["errors"].append({
                            "file": filename, 
                            "error": "No text extracted from PDF"
                        })
                        return results
                    
                    print(f"✅ Extracted {len(raw_text)} characters")
                    
                    # Parse with LLM (This is the slow part)
                    print("🤖 Parsing document with AI...")
                    parsed_doc = self.llm_parser.parse_document_with_llm(raw_text, filename)
                    
                    print("✅ Document parsed successfully")
                    print("🔍 Validating and enhancing...")
                    validated_doc = self.llm_parser.validate_and_enhance(parsed_doc, self.db)
                    
                    # Create inventory record
                    print("💾 Creating inventory record...")
                    inventory_record = {
                        "metadata": validated_doc.metadata.model_dump(),
                        "items": [item.model_dump() for item in validated_doc.items],
                        "raw_text_summary": validated_doc.raw_text_summary,
                        "source_file": filename,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    # Store in database
                    print("💾 Storing in database...")
                    self.db.store_inventory(inventory_record)
                    
                    # Create embeddings (can be slow for large docs)
                    print("🧠 Creating embeddings...")
                    chunks = self._create_searchable_chunks(raw_text, validated_doc)
                    
                    # Limit chunks to speed up processing
                    if len(chunks) > 20:
                        chunks = chunks[:20]  # Take only first 20 chunks
                    
                    embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
                    
                    embedding_docs = [
                        {
                            "file_path": filename,
                            "file_name": filename,
                            "file_hash": file_hash,
                            "chunk_index": i,
                            "text": chunk,
                            "embedding": emb.tolist(),
                            "created_at": datetime.now().isoformat()
                        }
                        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
                    ]
                    
                    self.db.store_embeddings(embedding_docs)
                    
                    # Store metadata
                    print("📋 Storing metadata...")
                    self.db.store_metadata({
                        "file_path": filename,
                        "file_name": filename,
                        "file_hash": file_hash,
                        "processed_date": datetime.now().isoformat(),
                        "items_extracted": len(validated_doc.items),
                        "llm_parsed": True
                    })
                    
                    # Learn patterns
                    print("🧠 Learning patterns...")
                    self._learn_from_document(validated_doc)
                    
                    print(f"✅ Successfully processed: {filename}")
                    results["processed"].append({
                        "filename": filename,
                        "items_count": len(validated_doc.items),
                        "document_date": validated_doc.metadata.date,
                        "total_amount": validated_doc.metadata.total_amount
                    })
                
                finally:
                    # Clean up temporary file
                    print("🧹 Cleaning up temporary file...")
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                results["errors"].append({
                    "file": filename, 
                    "error": str(e)
                })
            
            return results
        
        finally:
            self.processing = False
            print("✅ Processing complete")
    
    def process_file(self) -> Dict:
        """Process all PDF files in the Data folder (existing method - keep as is)"""
        if self.processing:
            return {"status": "already_processing"}
        
        self.processing = True
        results = {"processed": [], "skipped": [], "errors": []}
        
        try:
            if not os.path.exists(Config.DATA_FOLDER):
                os.makedirs(Config.DATA_FOLDER)
                return {"status": "folder_created"}
            
            pdf_files = list(Path(Config.DATA_FOLDER).glob("*.pdf"))
            
            for file_path in pdf_files:
                file_path_str = str(file_path)
                file_name = file_path.name
                
                file_hash = hashlib.sha256(open(file_path_str, 'rb').read()).hexdigest()
                
                if self.db.file_already_processed(file_path_str, file_hash):
                    results["skipped"].append(file_name)
                    continue
                
                try:
                    raw_text = self.llm_parser.extract_text_from_pdf(file_path_str)
                    
                    if not raw_text.strip():
                        results["errors"].append({"file": file_name, "error": "No text extracted"})
                        continue
                    
                    parsed_doc = self.llm_parser.parse_document_with_llm(raw_text, file_name)
                    validated_doc = self.llm_parser.validate_and_enhance(parsed_doc, self.db)
                    
                    inventory_record = {
                        "metadata": validated_doc.metadata.model_dump(),
                        "items": [item.model_dump() for item in validated_doc.items],
                        "raw_text_summary": validated_doc.raw_text_summary,
                        "source_file": file_name,
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    self.db.store_inventory(inventory_record)
                    
                    chunks = self._create_searchable_chunks(raw_text, validated_doc)
                    embeddings = self.embedding_model.encode(chunks)
                    
                    embedding_docs = [
                        {
                            "file_path": file_path_str,
                            "file_name": file_name,
                            "file_hash": file_hash,
                            "chunk_index": i,
                            "text": chunk,
                            "embedding": emb.tolist(),
                            "created_at": datetime.now().isoformat()
                        }
                        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
                    ]
                    
                    self.db.store_embeddings(embedding_docs)
                    
                    self.db.store_metadata({
                        "file_path": file_path_str,
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "processed_date": datetime.now().isoformat(),
                        "items_extracted": len(validated_doc.items),
                        "llm_parsed": True
                    })
                    
                    self._learn_from_document(validated_doc)
                    
                    results["processed"].append(file_name)
                
                except Exception as e:
                    results["errors"].append({"file": file_name, "error": str(e)})
            
            return results
        
        finally:
            self.processing = False
    
    def _create_searchable_chunks(self, raw_text: str, parsed_doc: ParsedDocument) -> List[str]:
        """Create searchable chunks from document"""
        chunks = []
        
        metadata_chunk = f"Date: {parsed_doc.metadata.date}, Concept: {parsed_doc.metadata.concept}, Total: {parsed_doc.metadata.total_amount}"
        chunks.append(metadata_chunk)
        
        for item in parsed_doc.items:
            item_chunk = f"Metal: {item.metal_name}, Quantity: {item.quantity} {item.unit}, Price: {item.unit_price}, Total: {item.total_cost}"
            chunks.append(item_chunk)
        
        chunks.append(parsed_doc.raw_text_summary)
        
        return chunks
    
    def _learn_from_document(self, parsed_doc: ParsedDocument):
        """Learn patterns from processed document"""
        metal_names = list(set(item.metal_name for item in parsed_doc.items))
        
        self.db.store_learned_pattern(
            "metal_names",
            {
                "metals": metal_names,
                "document_date": parsed_doc.metadata.date
            }
        )