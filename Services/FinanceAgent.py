import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import base64
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import hashlib

# Load environment variables
load_dotenv()

# ============================================================================
# DYNAMIC SCHEMAS FOR DIFFERENT DOCUMENT TYPES
# ============================================================================

class BankStatementSchema(BaseModel):
    """Schema for bank statements"""
    document_type: str = "bank_statement"
    account_info: Dict[str, Any] = Field(description="Account details")
    period: Dict[str, str] = Field(description="Statement period")
    balances: Dict[str, float] = Field(description="All balance information")
    transactions_summary: Dict[str, float] = Field(description="Summary of transactions")
    interest_info: Dict[str, Any] = Field(description="Interest rates and earnings")
    card_usage: Dict[str, float] = Field(description="Card usage details")
    additional_data: Dict[str, Any] = Field(default_factory=dict)

class InvoiceSchema(BaseModel):
    """Schema for invoices/purchase orders"""
    document_type: str = "invoice"
    invoice_number: Optional[str] = None
    vendor_info: Dict[str, Any] = Field(description="Vendor/supplier information")
    dates: Dict[str, str] = Field(description="Relevant dates")
    line_items: List[Dict[str, Any]] = Field(description="Products/services")
    totals: Dict[str, float] = Field(description="Total amounts")
    payment_info: Dict[str, Any] = Field(default_factory=dict)
    additional_data: Dict[str, Any] = Field(default_factory=dict)

class PriceListSchema(BaseModel):
    """Schema for price lists"""
    document_type: str = "price_list"
    contact_info: Dict[str, Any] = Field(description="Contact information")
    date: Optional[str] = None
    items: List[Dict[str, Any]] = Field(description="Items with prices")
    additional_data: Dict[str, Any] = Field(default_factory=dict)

# ============================================================================
# CONTENT EXTRACTION AGENT
# ============================================================================

class ContentExtractionAgent:
    """Agent responsible for extracting content from images using GPT-4o-mini Vision"""
    
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
    
    def encode_image_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def detect_document_type(self, image_data: bytes) -> str:
        """First pass: detect document type"""
        base64_image = self.encode_image_to_base64(image_data)
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Analyze this image and determine the document type. 
Return ONLY one of these exact types:
- bank_statement
- invoice
- purchase_order
- price_list
- receipt

Return just the type, nothing else."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        )
        
        response = self.llm.invoke([message])
        doc_type = response.content.strip().lower()
        return doc_type
    
    def extract_content(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Extract content based on document type with robust JSON parsing"""
        # Detect document type
        doc_type = self.detect_document_type(image_data)
        
        # Get appropriate extraction prompt
        extraction_prompt = self._get_extraction_prompt(doc_type)
        
        # Extract with vision
        base64_image = self.encode_image_to_base64(image_data)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": extraction_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        )
        
        # Retry logic for robust extraction
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke([message])
                
                # Parse JSON response with multiple strategies
                data = self._parse_json_response(response.content)
                
                # Add metadata
                data['_metadata'] = {
                    'source_file': filename,
                    'extraction_date': datetime.now().isoformat(),
                    'file_hash': self._get_file_hash(image_data)
                }
                
                return data
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    # Ask LLM to fix the JSON
                    fix_prompt = f"""The following JSON has an error: {str(e)}

Please fix this JSON and return ONLY valid JSON, nothing else:

{response.content}

Ensure:
1. All keys are in double quotes
2. No trailing commas
3. Proper escaping of special characters
4. Valid JSON structure"""
                    
                    message = HumanMessage(content=fix_prompt)
                    continue
                else:
                    # Last attempt failed, return with error but don't crash
                    return {
                        'document_type': doc_type,
                        'extraction_error': str(e),
                        'raw_content': response.content[:500],
                        '_metadata': {
                            'source_file': filename,
                            'extraction_date': datetime.now().isoformat(),
                            'file_hash': self._get_file_hash(image_data),
                            'status': 'partial_extraction'
                        }
                    }
            except Exception as e:
                return {
                    'document_type': doc_type,
                    'error': str(e),
                    '_metadata': {
                        'source_file': filename,
                        'extraction_date': datetime.now().isoformat(),
                        'file_hash': self._get_file_hash(image_data),
                        'status': 'failed'
                    }
                }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON with multiple strategies"""
        # Strategy 1: Direct parsing
        try:
            return json.loads(content)
        except:
            pass
        
        # Strategy 2: Extract from markdown code blocks
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content
        
        # Strategy 3: Clean and parse
        json_str = json_str.strip()
        
        # Remove potential issues
        json_str = json_str.replace('\n', ' ')
        json_str = json_str.replace('\r', ' ')
        
        # Try parsing
        return json.loads(json_str)
    
    def _get_extraction_prompt(self, doc_type: str) -> str:
        """Get extraction prompt based on document type"""
        
        base_instructions = """
CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON, no markdown, no explanations
2. All keys must be in double quotes
3. All string values must be in double quotes
4. Numbers should NOT be in quotes
5. No trailing commas
6. Use null for missing values, not empty strings
7. Ensure proper JSON structure

"""
        
        if doc_type == "bank_statement":
            return base_instructions + """Extract all information from this bank statement in the following JSON format:
{
  "document_type": "bank_statement",
  "account_info": {
    "account_type": "type of account",
    "account_number": null
  },
  "period": {
    "start_date": "start date",
    "end_date": "end date",
    "days": 0
  },
  "balances": {
    "initial_balance": 0.0,
    "current_balance": 0.0,
    "available_balance": 0.0,
    "average_balance": 0.0,
    "minimum_balance": 0.0
  },
  "transactions_summary": {
    "total_deposits": 0.0,
    "total_withdrawals": 0.0,
    "net_change": 0.0
  },
  "interest_info": {
    "interest_rate": 0.0,
    "interest_earned": 0.0,
    "isr_retention": 0.0
  },
  "card_usage": {
    "atm_withdrawals": 0.0,
    "commercial_purchases": 0.0,
    "total_card_usage": 0.0
  },
  "fees": {
    "total_fees": 0.0,
    "fee_details": []
  },
  "additional_data": {}
}"""

        elif doc_type in ["invoice", "purchase_order"]:
            return base_instructions + """Extract all information from this invoice/purchase order in the following JSON format:
{
  "document_type": "invoice",
  "invoice_number": null,
  "vendor_info": {
    "name": "vendor name",
    "city": null,
    "contact": null
  },
  "dates": {
    "issue_date": null,
    "delivery_date": null,
    "due_date": null
  },
  "line_items": [
    {
      "material": "item name",
      "quantity": 0.0,
      "unit_price": 0.0,
      "amount": 0.0,
      "details": null
    }
  ],
  "totals": {
    "subtotal": 0.0,
    "tax": 0.0,
    "total": 0.0,
    "paid": 0.0,
    "balance": 0.0
  },
  "payment_info": {},
  "additional_data": {}
}"""

        elif doc_type == "price_list":
            return base_instructions + """Extract all information from this price list in the following JSON format:
{
  "document_type": "price_list",
  "contact_info": {
    "whatsapp": null,
    "phone": null,
    "email": null
  },
  "date": null,
  "items": [
    {
      "material": "material name",
      "price": 0.0,
      "unit": null,
      "notes": null
    }
  ],
  "additional_data": {}
}"""
        
        else:
            return base_instructions + """Extract all information from this document in a structured JSON format."""
    
    def _get_file_hash(self, image_data: bytes) -> str:
        """Generate hash of image data for duplicate detection"""
        return hashlib.md5(image_data).hexdigest()

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class FinanceDatabaseManager:
    """Manages MongoDB database operations"""
    
    def __init__(self, mongodb_url: str):
        self.client = MongoClient(mongodb_url)
        self.db = self.client['FinanceAgentDB']
        self._setup_database()
    
    def _setup_database(self):
        """Setup database collections and indexes"""
        collections = {
            'bank_statements': ['_metadata.file_hash'],
            'invoices': ['_metadata.file_hash'],
            'price_lists': ['_metadata.file_hash'],
            'documents': ['_metadata.file_hash'],
            'chat_history': ['session_id', 'timestamp']
        }
        
        for collection_name, index_fields in collections.items():
            if collection_name not in self.db.list_collection_names():
                self.db.create_collection(collection_name)
            
            for field in index_fields:
                try:
                    self.db[collection_name].create_index(
                        field, 
                        unique=(field == '_metadata.file_hash')
                    )
                except:
                    pass  # Index might already exist
    
    def store_document(self, data: Dict[str, Any]) -> tuple[bool, str]:
        """Store extracted document in appropriate collection"""
        doc_type = data.get('document_type', 'unknown')
        file_hash = data.get('_metadata', {}).get('file_hash')
        
        if not file_hash:
            return False, "No file hash found"
        
        collection_map = {
            'bank_statement': 'bank_statements',
            'invoice': 'invoices',
            'purchase_order': 'invoices',
            'price_list': 'price_lists'
        }
        
        collection_name = collection_map.get(doc_type, 'documents')
        collection = self.db[collection_name]
        
        # Check for duplicates
        if collection.find_one({'_metadata.file_hash': file_hash}):
            return False, "Duplicate detected"
        
        # Insert document
        try:
            collection.insert_one(data)
            return True, f"Stored in {collection_name}"
        except DuplicateKeyError:
            return False, "Duplicate detected"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_all_documents(self) -> Dict[str, List[Dict]]:
        """Get all documents grouped by type"""
        return {
            'bank_statements': list(self.db['bank_statements'].find({}, {'_id': 0})),
            'invoices': list(self.db['invoices'].find({}, {'_id': 0})),
            'price_lists': list(self.db['price_lists'].find({}, {'_id': 0})),
            'documents': list(self.db['documents'].find({}, {'_id': 0}))
        }
    
    def save_chat_message(self, session_id: str, role: str, content: str):
        """Save chat message to history"""
        self.db['chat_history'].insert_one({
            'session_id': session_id,
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve chat history"""
        return list(self.db['chat_history']
                   .find({'session_id': session_id}, {'_id': 0})
                   .sort('timestamp', -1)
                   .limit(limit))

# ============================================================================
# RAG FINANCE AGENT
# ============================================================================

class RAGFinanceAgent:
    """RAG-based agent for answering user queries"""
    
    def __init__(self, openai_api_key: str, db_manager: FinanceDatabaseManager):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_api_key
        )
        self.db = db_manager
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def query(self, user_message: str) -> str:
        """Process user query with RAG - Completely Dynamic with Multi-language Support"""
        # Save user message
        self.db.save_chat_message(self.session_id, 'user', user_message)
        
        # Detect language of user query
        detected_language = self._detect_language(user_message)
        
        # Get chat history
        history = self.db.get_chat_history(self.session_id, limit=10)
        history.reverse()
        
        # Analyze query and retrieve data
        data_needed = self._analyze_query_intent(user_message)
        relevant_data = self._smart_retrieve(user_message, data_needed)
        
        # Build conversation
        conversation_history = []
        for msg in history[:-1]:
            conversation_history.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Create messages with language instruction
        messages = [
            {
                "role": "system",
                "content": f"""You are an intelligent multilingual financial assistant with access to the user's complete financial database.

CRITICAL INSTRUCTION: Respond in the SAME LANGUAGE as the user's query. The user is currently speaking in {detected_language}.

Your capabilities:
- Analyze bank statements, invoices, purchase orders, and price lists
- Perform calculations and provide insights
- Compare data across time periods
- Identify trends and patterns
- Answer specific queries about transactions, balances, vendors, materials, prices
- Provide summaries and detailed breakdowns as requested
- Handle complex multi-document queries
- Communicate in multiple languages (English, Spanish, Urdu, Arabic, French, German, etc.)

Guidelines:
1. ALWAYS respond in {detected_language} - match the user's language exactly
2. Always base answers on the actual data provided
3. Be precise with numbers and dates
4. When calculating, show your reasoning
5. If data is unavailable, clearly state what's missing
6. Adapt your response style to match the query
7. Use tables or formatting when presenting multiple items
8. Be conversational and helpful
9. If user switches languages, switch with them immediately

Language Detection: User is currently using {detected_language}"""
            }
        ]
        
        for msg in conversation_history:
            messages.append({
                "role": "user" if msg['role'] == 'user' else "assistant",
                "content": msg['content']
            })
        
        current_message = f"""USER QUERY (in {detected_language}): {user_message}

AVAILABLE FINANCIAL DATA:
{json.dumps(relevant_data, indent=2, ensure_ascii=False)}

IMPORTANT: Respond in {detected_language}. Analyze the data and provide a comprehensive, accurate response in the SAME language as the user's query."""
        
        messages.append({"role": "user", "content": current_message})
        
        # Get response
        response = self.llm.invoke([
            HumanMessage(content=msg["content"]) if msg["role"] == "user" 
            else AIMessage(content=msg["content"])
            for msg in messages
        ])
        
        ai_message = response.content
        self.db.save_chat_message(self.session_id, 'assistant', ai_message)
        
        return ai_message
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of user input"""
        detection_prompt = f"""Detect the language of this text and return ONLY the language name in English.

Text: "{text}"

Return ONLY one of these language names:
- English
- Spanish
- Urdu
- Arabic
- French
- German
- Chinese
- Japanese
- Portuguese
- Russian
- Hindi
- Italian

Return just the language name, nothing else."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=detection_prompt)])
            language = response.content.strip()
            return language if language else "English"
        except:
            return "English"
    
    def _analyze_query_intent(self, query: str) -> Dict[str, bool]:
        """Use LLM to understand what data the query needs"""
        analysis_prompt = f"""Analyze this query and determine what data is needed.

Query: "{query}"

Return JSON:
{{
    "needs_bank_statements": true/false,
    "needs_invoices": true/false,
    "needs_price_lists": true/false,
    "needs_all": true/false
}}

Return ONLY valid JSON."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            content = response.content
            
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
            
            return json.loads(json_str)
        except:
            return {"needs_all": True}
    
    def _smart_retrieve(self, query: str, intent: Dict[str, bool]) -> Dict[str, Any]:
        """Intelligently retrieve only relevant data"""
        all_docs = self.db.get_all_documents()
        
        result = {"bank_statements": [], "invoices": [], "price_lists": [], "summary": {}}
        
        if intent.get("needs_all", False):
            result["bank_statements"] = all_docs.get("bank_statements", [])
            result["invoices"] = all_docs.get("invoices", [])
            result["price_lists"] = all_docs.get("price_lists", [])
        else:
            if intent.get("needs_bank_statements", False):
                result["bank_statements"] = all_docs.get("bank_statements", [])
            if intent.get("needs_invoices", False):
                result["invoices"] = all_docs.get("invoices", [])
            if intent.get("needs_price_lists", False):
                result["price_lists"] = all_docs.get("price_lists", [])
        
        result["summary"] = {
            "total_bank_statements": len(all_docs.get("bank_statements", [])),
            "total_invoices": len(all_docs.get("invoices", [])),
            "total_price_lists": len(all_docs.get("price_lists", []))
        }
        
        return result