import uvicorn
import base64
import hashlib
import json
import os
import time
import random
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor
import threading
import google.generativeai as genai
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
from io import BytesIO
import asyncpg
import asyncio
import logging
import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Policy Reviewer API",
    description="API for analyzing policy documents using Gemini AI with chat history support and RAG capabilities",
    version="2.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read API key from environment variables (for Vercel deployment)
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Pinecone configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "policy-documents")

# Database connection pool
db_pool = None

# Document storage using an enhanced cache
class DocumentCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest accessed item
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = value
        self.access_times[key] = time.time()

    def clear_old_entries(self, max_age_hours=24):
        """Clear entries older than max_age_hours"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        old_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > max_age_seconds
        ]

        for key in old_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]

enhanced_document_cache = DocumentCache(max_size=1000)

# Chat history storage (in production, this should be in database)
chat_history_cache = {}

# Track analysis progress
analysis_progress = {}

# Global thread pool for reuse
thread_pool = ThreadPoolExecutor(max_workers=8)

# Create a session with connection pooling
session = requests.Session()
retry_strategy = Retry(
    total=2,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Initialize embedding model (lazy loaded)
embedding_model = None
pinecone_index = None

def get_embedding_model():
    """Lazy load embedding model to save memory"""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
    return embedding_model

def init_pinecone():
    """Initialize Pinecone connection"""
    global pinecone_index
    if PINECONE_API_KEY and not pinecone_index:
        try:
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone initialized successfully")
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {str(e)}")

def get_api_key(api_key: Optional[str] = None):
    """Get API key from environment variable or request"""
    return api_key or DEFAULT_API_KEY

class Location(BaseModel):
    continent: str
    countries: List[str]

class UseCase(BaseModel):
    category: str
    subCategories: List[str]

class PolicyAnalysisRequest(BaseModel):
    text: Optional[str] = None
    sector: str = "Technology"
    sub_sector: Optional[str] = None
    ai_usage: str = "Moderate"
    countries: List[str] = ["USA"]
    use_cases: Optional[str] = None
    api_key: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis: str
    request_id: str
    document_id: str
    raw_text: Optional[str] = None

class ApiStatusResponse(BaseModel):
    status: str
    message: str

class AnalysisStatusResponse(BaseModel):
    status: str
    request_id: str
    completion_percentage: Optional[int] = None

class ChatRequest(BaseModel):
    message: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    sector: Optional[str] = "Technology"
    subsector: Optional[str] = None
    jurisdictions: List[str] = []
    roles: List[str] = ["consumer"]
    use_cases: List[str] = []
    locations: List[Location] = []
    useCases: List[UseCase] = []
    api_key: Optional[str] = None
    conversation_context: Optional[List[Dict[str, Any]]] = None
    system_role: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    document_id: Optional[str] = None
    conversation_context: List[Dict[str, Any]]
    assistant_response: List[Dict[str, Any]]

class PolicyAnalysisFileRequest(BaseModel):
    files: List[UploadFile] = File([])
    storj_urls: Optional[List[str]] = Form(None)
    entityType: List[str] = []
    sector: str = "Technology"
    subSector: Optional[str] = None
    locations: List[Location] = []
    useCases: List[UseCase] = []
    user_id_uuid: Optional[str] = None
    context_id_uuid: Optional[str] = None
    org_id_uuid: Optional[str] = "default_context"
    api_key: Optional[str] = None

def setup_gemini(api_key):
    """Set up the Gemini API client with higher token limits"""
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel('gemini-2.0-flash', generation_config={
        "temperature": 0.4,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192
    })
    return model

def direct_api_call_optimized(prompt, api_key, max_retries=2):
    """Optimized API call with session reuse"""
    if not api_key:
        raise ValueError("API key is required")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = session.post(url, json=payload, headers=headers, timeout=45)
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
        return f"API error: {response.status_code}"
    except Exception as e:
        return f"API request failed: {str(e)}"

def download_from_storj_url(url: str) -> bytes:
    """Download file content from URL with better error handling"""
    try:
        logger.info(f"Attempting to download from URL: {url[:100]}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, timeout=60, headers=headers, stream=True)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        logger.info(f"Content-Length: {response.headers.get('content-length', 'unknown')}")

        if response.status_code == 403:
            raise HTTPException(status_code=400, detail="Access denied. The presigned URL may have expired or is invalid.")
        elif response.status_code == 404:
            raise HTTPException(status_code=400, detail="File not found at the provided URL.")
        elif response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download file. HTTP {response.status_code}: {response.reason}")

        response.raise_for_status()
        content = response.content

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Downloaded file is empty")

        logger.info(f"Successfully downloaded {len(content)} bytes")

        if content[:4] == b'%PDF':
            logger.info("Downloaded content appears to be a valid PDF")
        else:
            logger.info(f"Downloaded content starts with: {content[:20]}")

        return content

    except requests.exceptions.Timeout:
        logger.error(f"Timeout while downloading from URL: {url}")
        raise HTTPException(status_code=400, detail="Request timeout while downloading file from URL")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error while downloading from URL: {url}")
        raise HTTPException(status_code=400, detail="Connection error while downloading file from URL")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while downloading from URL {url}: {str(e)}")
        if e.response.status_code == 403:
            raise HTTPException(status_code=400, detail="Access denied. The presigned URL may have expired.")
        raise HTTPException(status_code=400, detail=f"HTTP error while downloading file: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception while downloading from URL {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while downloading from URL {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error while downloading file: {str(e)}")

def is_storj_url(url: str) -> bool:
    """Check if the URL is a Storj share URL"""
    parsed_url = urlparse(url)
    storj_domains = [
        'storjshare.io',
        'gateway.storjshare.io',
        'regulations-frameworks.gateway.storjshare.io'
    ]
    return any(parsed_url.netloc.endswith(domain) or parsed_url.netloc == domain for domain in storj_domains)

def normalize_storj_url(url: str) -> str:
    """Normalize Storj URL for consistent storage while preserving essential parameters"""
    parsed_url = urlparse(url)
    if parsed_url.netloc.endswith('storjshare.io') or parsed_url.netloc.endswith('gateway.storjshare.io'):
        return url
    return url

def get_storj_base_path(url: str) -> str:
    """Get the base path of a Storj URL for identification purposes"""
    parsed_url = urlparse(url)
    if parsed_url.netloc.endswith('storjshare.io') or parsed_url.netloc.endswith('gateway.storjshare.io'):
        return urlunparse(parsed_url._replace(query='', fragment=''))
    return url

def extract_text_from_pdf_fast(pdf_bytes):
    """Faster PDF extraction with better error handling"""
    try:
        if not pdf_bytes or len(pdf_bytes) < 4:
            return "Invalid PDF file"

        text_parts = []
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        logger.info(f"Processing PDF with {total_pages} pages")

        # Process pages in batches for better memory management
        batch_size = 10
        for start_page in range(0, total_pages, batch_size):
            end_page = min(start_page + batch_size, total_pages)
            for page_num in range(start_page, end_page):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        cleaned_text = ' '.join(page_text.split())
                        text_parts.append(cleaned_text)
                except Exception as e:
                    logger.warning(f"Error on page {page_num + 1}: {str(e)}")
                    continue

        if not text_parts:
            return "No text extracted from PDF"

        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from {total_pages} pages")
        return full_text

    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return f"PDF extraction failed: {str(e)}"

def generate_unique_id():
    """Generate a unique request ID"""
    return f"req_{int(time.time())}_{hash(str(time.time()))}"[-12:]

def generate_session_id():
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def create_system_role(sector, subsector, jurisdictions, roles, use_cases):
    """Create system role configuration"""
    return {
        "sector": sector.lower().replace(" ", "-") if sector else "technology",
        "sectorReadable": sector or "Technology",
        "subsector": subsector.lower().replace(" ", "-") if subsector else "",
        "subsectorReadable": subsector or "",
        "jurisdictions": [j.lower() for j in jurisdictions] if jurisdictions else [],
        "jurisdictionReadable": ", ".join(jurisdictions) if jurisdictions else "",
        "roles": [r.lower() for r in roles] if roles else ["consumer"],
        "rolesReadable": ", ".join(roles) if roles else "Consumer",
        "useCases": use_cases or [],
        "useCasesReadable": ", ".join(use_cases) if use_cases else ""
    }

def format_locations(locations: List[Location]) -> str:
    formatted = []
    for loc in locations:
        countries = ", ".join(loc.countries)
        formatted.append(f"{loc.continent}: {countries}")
    return "\n    ".join(formatted)

def format_use_cases(use_cases: List[UseCase]) -> str:
    formatted = []
    for uc in use_cases:
        subcats = ", ".join(uc.subCategories)
        formatted.append(f"{uc.category}:\n      - {subcats}")
    return "\n    ".join(formatted)

def smart_chunk_text(text, max_chunk_size=150000):
    """Smarter chunking that preserves context"""
    if len(text) <= max_chunk_size:
        return [text]

    # Split on double newlines first (paragraphs/sections)
    sections = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        section_size = len(section) + 2
        if current_size + section_size > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks

def smart_chunk_for_rag(text, chunk_size=1000, overlap=100):
    """Create overlapping chunks optimized for RAG"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                'text': chunk_text,
                'length': len(chunk_text),
                'sentences': len(current_chunk)
            })

            # Keep overlap sentences
            overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        chunks.append({
            'text': chunk_text,
            'length': len(chunk_text),
            'sentences': len(current_chunk)
        })

    return chunks

def build_concise_prompt(request):
    """Build shorter, more focused prompt"""
    locations_summary = f"{len(request.locations)} regions"
    use_cases_summary = f"{len(request.useCases)} categories"
    return f"""Analyze this policy document for:
Entity Types: {', '.join(request.entityType)}
Sector: {request.sector} - {request.subSector or 'General'}
Locations: {locations_summary}
Use Cases: {use_cases_summary}
Provide concise analysis covering:
1. Key policy points
2. Impact on specified entities/sectors
3. Compliance requirements
4. Recommendations
Keep response focused and actionable."""

async def init_db():
    """Initialize database connection pool"""
    global db_pool
    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'application_name': 'policy_reviewer_api',
                    'search_path': 'public',
                }
            )
            logger.info("Database connection pool created successfully")

            async with db_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        session_id UUID UNIQUE NOT NULL,
                        conversation_context JSONB NOT NULL DEFAULT '[]',
                        assistant_response JSONB NOT NULL DEFAULT '[]',
                        system_role JSONB NOT NULL DEFAULT '{}',
                        document_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS policies (
                        id SERIAL PRIMARY KEY,
                        storj_url TEXT UNIQUE,
                        document_id VARCHAR(255) NOT NULL,
                        analysis TEXT NOT NULL,
                        raw_text TEXT,
                        entityType JSONB,
                        sector VARCHAR(255),
                        subSector VARCHAR(255),
                        locations JSONB,
                        useCases JSONB,
                        created_at_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_id_uuid UUID,
                        context_id_uuid UUID,
                        org_id_uuid UUID,
                        metadata JSONB
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        document_id VARCHAR(255) UNIQUE NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

        except Exception as e:
            logger.error(f"Error creating database connection pool: {str(e)}")
            raise

async def save_chat_history_to_db(session_id: str, chat_data: Dict[str, Any]) -> bool:
    """Save chat history to database"""
    if not db_pool:
        logger.error("Database connection pool is not initialized")
        return False

    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                query = """
                INSERT INTO chat_history (
                    session_id, conversation_context, assistant_response,
                    system_role, document_id, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (session_id) DO UPDATE SET
                    conversation_context = $2,
                    assistant_response = $3,
                    system_role = $4,
                    document_id = $5,
                    updated_at = $6
                """

                await conn.execute(
                    query,
                    session_id,
                    json.dumps(chat_data.get("conversation_context", [])),
                    json.dumps(chat_data.get("assistant_response", [])),
                    json.dumps(chat_data.get("system_role", {})),
                    chat_data.get("document_id"),
                    datetime.utcnow()
                )
                return True

    except Exception as e:
        logger.error(f"Error saving chat history to database: {str(e)}")
        return False

async def get_chat_history_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """Get chat history from database"""
    if not db_pool:
        logger.error("Database connection pool is not initialized")
        return None

    try:
        async with db_pool.acquire() as conn:
            query = """
            SELECT conversation_context, assistant_response, system_role,
                   document_id, updated_at
            FROM chat_history
            WHERE session_id = $1
            """
            row = await conn.fetchrow(query, session_id)
            if row:
                return {
                    "conversation_context": json.loads(row["conversation_context"]),
                    "assistant_response": json.loads(row["assistant_response"]),
                    "system_role": json.loads(row["system_role"]),
                    "document_id": row["document_id"],
                    "updated_at": row["updated_at"].timestamp()
                }
    except Exception as e:
        logger.error(f"Error retrieving chat history from database: {str(e)}")
        return None

    return None

async def check_url_exists_in_db(storj_url: str) -> Optional[Dict]:
    """Check if Storj URL already exists in policies table"""
    if not db_pool:
        logger.error("Database connection pool is not initialized")
        return None

    try:
        if is_storj_url(storj_url):
            base_path = get_storj_base_path(storj_url)
            logger.info(f"Checking if base path exists in database: {base_path}")
            async with db_pool.acquire() as conn:
                pattern = base_path + "%"
                query = """
                SELECT document_id, analysis, created_at_timestamp, metadata, storj_url
                FROM policies
                WHERE storj_url LIKE $1
                ORDER BY created_at_timestamp DESC
                LIMIT 1
                """
                row = await conn.fetchrow(query, pattern)
        else:
            async with db_pool.acquire() as conn:
                query = """
                SELECT document_id, analysis, created_at_timestamp, metadata, storj_url
                FROM policies
                WHERE storj_url = $1
                ORDER BY created_at_timestamp DESC
                LIMIT 1
                """
                row = await conn.fetchrow(query, storj_url)

        if row:
            logger.info(f"Found existing URL in database")
            return {
                "document_id": row["document_id"],
                "analysis": row["analysis"],
                "created_at_timestamp": row["created_at_timestamp"],
                "metadata": row["metadata"],
                "from_cache": True
            }
    except Exception as e:
        logger.error(f"Error checking URL in database: {str(e)}")
        return None

    return None

async def save_analysis_to_db(
    storj_url: str,
    document_id: str,
    analysis: str,
    raw_text: str,
    metadata: Dict[str, Any]
) -> bool:
    """Save analysis results to database"""
    if not db_pool:
        logger.error("Database connection pool is not initialized")
        return False

    try:
        logger.info(f"Saving analysis to database for URL: {storj_url}")
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                query = """
                INSERT INTO policies (
                    storj_url, document_id, analysis, raw_text,
                    entityType, sector, subSector,
                    locations, useCases,
                    created_at_timestamp, updated_at_timestamp,
                    user_id_uuid, context_id_uuid, org_id_uuid, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (storj_url) DO UPDATE SET
                    updated_at_timestamp = $11,
                    analysis = $3,
                    raw_text = $4,
                    metadata = $15
                """

                locations_json = [loc.dict() for loc in metadata.get("locations", [])]
                use_cases_json = [uc.dict() for uc in metadata.get("useCases", [])]

                await conn.execute(
                    query,
                    storj_url, document_id, analysis, raw_text,
                    json.dumps(metadata.get("entityType", [])),
                    metadata.get("sector", ""),
                    metadata.get("subSector"),
                    json.dumps(locations_json),
                    json.dumps(use_cases_json),
                    metadata.get("created_at_timestamp"),
                    metadata.get("updated_at_timestamp"),
                    metadata.get("user_id_uuid"),
                    metadata.get("context_id_uuid"),
                    metadata.get("org_id_uuid"),
                    json.dumps(metadata)
                )
                return True

    except Exception as e:
        logger.error(f"Error saving analysis to database: {str(e)}")
        return False

async def save_document_to_db(document_id: str, content: str, metadata: Dict[str, Any]) -> bool:
    """Save document content to database for lifetime access"""
    if not db_pool:
        logger.error("Database connection pool is not initialized")
        return False

    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                query = """
                INSERT INTO documents (
                    document_id, content, metadata, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (document_id) DO UPDATE SET
                    content = $2,
                    metadata = $3,
                    updated_at = $5
                """

                await conn.execute(
                    query,
                    document_id,
                    content,
                    json.dumps(metadata),
                    datetime.utcnow(),
                    datetime.utcnow()
                )
                return True

    except Exception as e:
        logger.error(f"Error saving document to database: {str(e)}")
        return False

async def get_document_from_db(document_id: str) -> Optional[str]:
    """Get document content from database"""
    if not db_pool:
        logger.error("Database connection pool is not initialized")
        return None

    try:
        async with db_pool.acquire() as conn:
            query = "SELECT content FROM documents WHERE document_id = $1"
            row = await conn.fetchrow(query, document_id)
            if row:
                return row["content"]
    except Exception as e:
        logger.error(f"Error retrieving document from database: {str(e)}")
        return None

    return None

async def save_to_db_async(url, doc_id, analysis, text, request):
    """Non-blocking database save"""
    try:
        current_timestamp = datetime.utcnow().isoformat()
        metadata = {
            "entityType": request.entityType,
            "sector": request.sector,
            "subSector": request.subSector,
            "locations": [loc.dict() for loc in request.locations],
            "useCases": [uc.dict() for uc in request.useCases],
            "created_at_timestamp": current_timestamp,
            "updated_at_timestamp": current_timestamp,
            "user_id_uuid": request.user_id_uuid or str(uuid.uuid4()),
            "context_id_uuid": request.context_id_uuid or str(uuid.uuid4()),
            "org_id_uuid": request.org_id_uuid
        }
        await save_analysis_to_db(url, doc_id, analysis, text, metadata)
    except Exception as e:
        logger.warning(f"Background save failed: {str(e)}")

async def process_document_with_embeddings(document_text, document_id, metadata):
    """Process document and create embeddings for RAG"""
    try:
        # Create chunks
        chunks = smart_chunk_for_rag(document_text)

        # Generate embeddings for chunks
        model = get_embedding_model()
        embeddings_data = []

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk['text'])

            chunk_id = f"{document_id}_chunk_{i}"
            chunk_data = {
                'id': chunk_id,
                'values': embedding.tolist(),
                'metadata': {
                    'document_id': document_id,
                    'chunk_index': i,
                    'text': chunk['text'],
                    'length': chunk['length'],
                    **metadata
                }
            }
            embeddings_data.append(chunk_data)

        # Store in Pinecone
        if pinecone_index:
            pinecone_index.upsert(vectors=embeddings_data, namespace="documents")

        # Store summary embedding for quick document retrieval
        summary = document_text[:2000]  # First 2000 chars as summary
        summary_embedding = model.encode(summary)

        summary_data = {
            'id': f"{document_id}_summary",
            'values': summary_embedding.tolist(),
            'metadata': {
                'document_id': document_id,
                'type': 'summary',
                'text': summary,
                **metadata
            }
        }

        if pinecone_index:
            pinecone_index.upsert(vectors=[summary_data], namespace="summaries")

        return {
            'chunks_created': len(chunks),
            'embeddings_stored': len(embeddings_data),
            'document_id': document_id
        }

    except Exception as e:
        logger.error(f"Error processing document embeddings: {str(e)}")
        return None

async def retrieve_relevant_chunks(query, top_k=5, namespace="documents"):
    """Retrieve relevant chunks using RAG"""
    try:
        if not pinecone_index:
            return []

        # Generate query embedding
        model = get_embedding_model()
        query_embedding = model.encode(query)

        # Search in Pinecone
        results = pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )

        relevant_chunks = []
        for match in results['matches']:
            relevant_chunks.append({
                'text': match['metadata']['text'],
                'score': match['score'],
                'document_id': match['metadata']['document_id'],
                'chunk_index': match['metadata'].get('chunk_index', 0)
            })

        return relevant_chunks

    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {str(e)}")
        return []

async def analyze_chunks_parallel(chunks, prompt_context, api_key):
    """Analyze chunks in parallel"""
    loop = asyncio.get_event_loop()
    # Create tasks for parallel execution
    tasks = []
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"""
        {prompt_context}
        Part {i+1} of {len(chunks)}. Analyze this section:
        {chunk}
        """
        task = loop.run_in_executor(thread_pool, make_gemini_api_call, chunk_prompt, api_key)
        tasks.append(task)

    # Execute with controlled concurrency (batches of 3)
    batch_size = 3
    all_results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        all_results.extend(batch_results)
        # Small delay between batches to avoid rate limiting
        if i + batch_size < len(tasks):
            await asyncio.sleep(0.5)

    return [r for r in all_results if not isinstance(r, Exception)]

async def analyze_large_document_fast(text, prompt_context, api_key, request_id):
    """Fast document analysis with parallel processing"""
    chunks = smart_chunk_text(text, max_chunk_size=150000)
    if len(chunks) == 1:
        # Single chunk - process directly
        full_prompt = f"{prompt_context}\n\nDocument: {text}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(thread_pool, make_gemini_api_call, full_prompt, api_key)

    # Multiple chunks - process in parallel
    chunk_analyses = await analyze_chunks_parallel(chunks, prompt_context, api_key)

    # Quick combination of results
    combined_analysis = "\n\n".join([f"## Section {i+1}\n{analysis}"
                                   for i, analysis in enumerate(chunk_analyses)])

    # Shorter synthesis prompt for speed
    synthesis_prompt = f"""Based on these section analyses, provide a brief synthesis:
    {combined_analysis[:6000]}...
    Provide: 1) Key summary 2) Main findings 3) Critical recommendations"""

    loop = asyncio.get_event_loop()
    final_synthesis = await loop.run_in_executor(thread_pool, make_gemini_api_call, synthesis_prompt, api_key)

    return f"{combined_analysis}\n\n## Executive Summary\n{final_synthesis}"

def make_gemini_api_call(prompt, api_key):
    """Thread-safe Gemini API call"""
    return direct_api_call_optimized(prompt, api_key, max_retries=2)

async def download_and_process_single_url(url, api_key):
    """Process single URL in thread"""
    try:
        # Check cache first
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cached_result = loop.run_until_complete(check_url_exists_in_db(url))
        if cached_result:
            return {"cached": True, "result": cached_result}

        # Download content
        content = download_from_storj_url(url)

        # Extract text
        if content[:4] == b'%PDF':
            text = extract_text_from_pdf_fast(content)
        else:
            text = content.decode('utf-8', errors='replace')

        return {"cached": False, "url": url, "text": text, "content": content}
    except Exception as e:
        return {"error": str(e), "url": url}

async def process_urls_parallel(storj_urls, api_key):
    """Process URLs in parallel using thread pool"""
    loop = asyncio.get_event_loop()
    # Submit all tasks to thread pool
    tasks = []
    for url in storj_urls:
        task = loop.run_in_executor(thread_pool, download_and_process_single_url, url, api_key)
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    successful_results = []
    for result in results:
        if isinstance(result, dict) and "error" not in result:
            successful_results.append(result)

    return successful_results

async def batch_process_documents(urls_batch, api_key, batch_size=10):
    """Process documents in batches to avoid memory issues"""
    results = []

    for i in range(0, len(urls_batch), batch_size):
        batch = urls_batch[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")

        # Process batch
        batch_results = await process_urls_parallel(batch, api_key)

        # Create embeddings for successful results
        for result in batch_results:
            if 'text' in result:
                metadata = {
                    'url': result['url'],
                    'processed_at': datetime.utcnow().isoformat()
                }

                doc_id = hashlib.md5(result['url'].encode()).hexdigest()
                embedding_result = await process_document_with_embeddings(
                    result['text'], doc_id, metadata
                )

                if embedding_result:
                    result['embedding_info'] = embedding_result

        results.extend(batch_results)
        await asyncio.sleep(0.1)

    return results

async def enhanced_rag_chat(query, document_ids=None, session_id=None, api_key=None):
    """Enhanced chat using RAG for better context retrieval"""
    try:
        # Retrieve relevant chunks
        relevant_chunks = await retrieve_relevant_chunks(query, top_k=8)

        # Filter by document_ids if specified
        if document_ids:
            relevant_chunks = [
                chunk for chunk in relevant_chunks
                if chunk['document_id'] in document_ids
            ]

        # Build context from relevant chunks
        context_parts = []
        for chunk in relevant_chunks[:5]:  # Use top 5 chunks
            context_parts.append(f"[Score: {chunk['score']:.3f}] {chunk['text'][:500]}...")

        context = "\n\n".join(context_parts)

        # Build enhanced prompt
        prompt = f"""Based on the following relevant document excerpts, answer the user's question:
RELEVANT CONTEXT:
{context}
USER QUESTION: {query}
Provide a comprehensive answer based on the context provided. If the context doesn't contain enough information, mention what additional information might be needed."""

        # Generate response
        response = direct_api_call_optimized(prompt, api_key)

        return {
            'response': response,
            'sources_used': len(relevant_chunks),
            'relevant_chunks': [
                {
                    'document_id': chunk['document_id'],
                    'score': chunk['score'],
                    'preview': chunk['text'][:100] + '...'
                } for chunk in relevant_chunks[:3]
            ]
        }

    except Exception as e:
        logger.error(f"RAG chat error: {str(e)}")
        return {'response': 'Error processing request', 'sources_used': 0}

async def stream_analysis_results(request_id, chunks, api_key):
    """Stream analysis results for better UX"""
    try:
        analysis_progress[request_id] = {
            'status': 'processing',
            'completed_chunks': 0,
            'total_chunks': len(chunks),
            'results': []
        }

        for i, chunk in enumerate(chunks):
            # Analyze chunk
            prompt = f"Analyze this policy section: {chunk[:2000]}..."
            result = direct_api_call_optimized(prompt, api_key)

            # Update progress
            analysis_progress[request_id]['completed_chunks'] = i + 1
            analysis_progress[request_id]['results'].append({
                'chunk_index': i,
                'analysis': result,
                'timestamp': datetime.utcnow().isoformat()
            })

            # Small delay to prevent rate limiting
            await asyncio.sleep(0.1)

        analysis_progress[request_id]['status'] = 'completed'
        return analysis_progress[request_id]

    except Exception as e:
        analysis_progress[request_id]['status'] = 'error'
        analysis_progress[request_id]['error'] = str(e)
        logger.error(f"Streaming analysis error: {str(e)}")

async def bulk_save_analyses(analyses_data):
    """Bulk save multiple analyses to database"""
    if not db_pool or not analyses_data:
        return False

    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                query = """
                INSERT INTO policies (
                    storj_url, document_id, analysis, raw_text,
                    entityType, sector, subSector, locations, useCases,
                    created_at_timestamp, updated_at_timestamp,
                    user_id_uuid, context_id_uuid, org_id_uuid, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (storj_url) DO NOTHING
                """

                # Prepare all data for bulk insert
                bulk_data = []
                for data in analyses_data:
                    bulk_data.append((
                        data['url'], data['document_id'], data['analysis'], data['text'],
                        json.dumps(data['metadata'].get('entityType', [])),
                        data['metadata'].get('sector', ''),
                        data['metadata'].get('subSector'),
                        json.dumps(data['metadata'].get('locations', [])),
                        json.dumps(data['metadata'].get('useCases', [])),
                        data['metadata'].get('created_at_timestamp'),
                        data['metadata'].get('updated_at_timestamp'),
                        data['metadata'].get('user_id_uuid'),
                        data['metadata'].get('context_id_uuid'),
                        data['metadata'].get('org_id_uuid'),
                        json.dumps(data['metadata'])
                    ))

                # Execute bulk insert
                await conn.executemany(query, bulk_data)
                return True

    except Exception as e:
        logger.error(f"Bulk save error: {str(e)}")
        return False

async def process_bulk_documents_async(urls, request, api_key, request_id):
    """Background task for bulk document processing"""
    try:
        analysis_progress[request_id]['status'] = 'processing'

        # Process in batches
        batch_size = 20
        all_results = []
        failed_count = 0

        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]

            # Process batch
            batch_results = await batch_process_documents(batch_urls, api_key, batch_size=10)

            # Count successful and failed
            for result in batch_results:
                if 'error' in result:
                    failed_count += 1
                else:
                    all_results.append(result)

            # Update progress
            analysis_progress[request_id]['processed_documents'] = min(i + batch_size, len(urls))
            analysis_progress[request_id]['failed_documents'] = failed_count

            # Small delay between batches
            await asyncio.sleep(0.5)

        # Bulk save successful results
        if all_results:
            analyses_data = []
            for result in all_results:
                if 'text' in result:
                    doc_id = hashlib.md5(result['url'].encode()).hexdigest()

                    # Quick analysis for bulk processing
                    analysis = f"Document processed: {len(result['text'])} characters"
                    if result.get('embedding_info'):
                        analysis += f", {result['embedding_info']['chunks_created']} chunks created"

                    analyses_data.append({
                        'url': result['url'],
                        'document_id': doc_id,
                        'analysis': analysis,
                        'text': result['text'],
                        'metadata': {
                            'entityType': request.entityType,
                            'sector': request.sector,
                            'subSector': request.subSector,
                            'created_at_timestamp': datetime.utcnow().isoformat(),
                            'updated_at_timestamp': datetime.utcnow().isoformat(),
                            'user_id_uuid': request.user_id_uuid or str(uuid.uuid4()),
                            'context_id_uuid': request.context_id_uuid or str(uuid.uuid4()),
                            'org_id_uuid': request.org_id_uuid
                        }
                    })

            await bulk_save_analyses(analyses_data)

        # Mark as completed
        analysis_progress[request_id]['status'] = 'completed'
        analysis_progress[request_id]['end_time'] = time.time()
        analysis_progress[request_id]['total_processed'] = len(all_results)

    except Exception as e:
        analysis_progress[request_id]['status'] = 'error'
        analysis_progress[request_id]['error'] = str(e)
        logger.error(f"Bulk processing error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize database connection and Pinecone on startup"""
    await init_db()
    init_pinecone()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup database connections on shutdown"""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")

@app.get("/")
async def root():
    return {"message": "Welcome to the Policy Reviewer API with Chat History and RAG. Visit /docs for API documentation."}

@app.get("/api/status", response_model=ApiStatusResponse)
async def check_api_status():
    return {
        "status": "operational",
        "message": "API is running and operational"
    }

@app.post("/api/analyze/text")
async def analyze_policy_text(request: PolicyAnalysisRequest):
    """Analyze policy text"""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required")

    api_key = get_api_key(request.api_key)
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required either in request or as environment variable")

    request_id = generate_unique_id()
    document_id = hashlib.md5(request.text.encode()).hexdigest()
    enhanced_document_cache.set(document_id, request.text)

    prompt = f"""Analyze the following policy document and provide a detailed review:
    Policy Document: {request.text}...
    Sector: {request.sector}
    Sub-sector: {request.sub_sector or "Not specified"}
    AI Usage Level: {request.ai_usage}
    Countries: {', '.join(request.countries)}
    Use Cases: {request.use_cases or "Not specified"}
    Please provide:
    1. A summary of the key points of the policy
    2. Analysis of the potential impacts
    3. Identification of any ambiguities or areas needing clarification
    4. Recommendations for improvement
    5. Compliance considerations
    Format your response in markdown with clear sections.
    """

    analysis_result = direct_api_call_optimized(prompt, api_key)

    return JSONResponse(content={
        "analysis": analysis_result,
        "request_id": request_id,
        "document_id": document_id,
        "raw_text": request.text[:1000] + "..." if len(request.text) > 1000 else request.text
    })

@app.post("/api/analyze/file")
async def analyze_policy_file_optimized(request: PolicyAnalysisFileRequest):
    """Optimized file analysis endpoint"""
    try:
        # Quick validation
        api_key = get_api_key(request.api_key)
        if not api_key:
            raise HTTPException(status_code=400, detail="API key required")
        if not request.storj_urls:
            raise HTTPException(status_code=400, detail="No URLs provided")

        # Process URLs in parallel
        processed_files = await process_urls_parallel(request.storj_urls, api_key)

        # Handle cached results
        for file_result in processed_files:
            if file_result.get("cached"):
                cached_result = file_result["result"]
                return JSONResponse(content={
                    "analysis": cached_result["analysis"],
                    "request_id": generate_unique_id(),
                    "document_id": cached_result["document_id"],
                    "from_cache": True,
                    "cached_at": str(cached_result["created_at_timestamp"])
                })

        if not processed_files:
            raise HTTPException(status_code=400, detail="No files processed successfully")

        # Combine text efficiently
        all_text = []
        for file_result in processed_files:
            if "text" in file_result:
                all_text.append(f"=== {file_result['url']} ===\n{file_result['text']}")

        combined_text = "\n\n===DOCUMENT SEPARATOR===\n\n".join(all_text)

        # Generate document ID
        document_id = hashlib.md5(combined_text.encode()).hexdigest()

        # Store in cache
        enhanced_document_cache.set(document_id, combined_text)

        # Build concise prompt
        prompt_context = build_concise_prompt(request)

        # Run fast analysis
        request_id = generate_unique_id()
        analysis_result = await analyze_large_document_fast(
            combined_text, prompt_context, api_key, request_id
        )

        # Save to database without blocking (fire and forget)
        asyncio.create_task(save_to_db_async(request.storj_urls[0], document_id,
                                           analysis_result, combined_text, request))

        return JSONResponse(content={
            "analysis": analysis_result,
            "request_id": request_id,
            "document_id": document_id,
            "raw_text": combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text,
            "from_cache": False,
            "processing_optimized": True
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def enhanced_chat_with_document(request: ChatRequest):
    """Enhanced chat with document analysis and chat history support"""
    api_key = get_api_key(request.api_key)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key is required either in request or as environment variable"
        )

    session_id = request.session_id or generate_session_id()

    if request.system_role:
        system_role = request.system_role
    else:
        system_role = create_system_role(
            request.sector,
            request.subsector,
            request.jurisdictions,
            request.roles,
            request.use_cases
        )

    if request.conversation_context is not None:
        chat_history = {
            "conversation_context": request.conversation_context,
            "assistant_response": [],
            "system_role": system_role,
            "document_id": request.document_id,
            "updated_at": time.time()
        }
    else:
        chat_history = await get_chat_history_from_db(session_id)
        if not chat_history:
            chat_history = {
                "conversation_context": [],
                "assistant_response": [],
                "system_role": system_role,
                "document_id": request.document_id,
                "updated_at": time.time()
            }

    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": request.message}]
    }
    chat_history["conversation_context"].append(user_message)

    document_context = ""
    if request.document_id:
        cached_doc = enhanced_document_cache.get(request.document_id)
        if cached_doc:
            document_context = f"Document Context:\n{cached_doc[:2000]}...\n\n"
        else:
            db_document = await get_document_from_db(request.document_id)
            if db_document:
                document_context = f"Document Context:\n{db_document[:2000]}...\n\n"
                enhanced_document_cache.set(request.document_id, db_document)

    conversation_history = ""
    for msg in chat_history["conversation_context"][-10:]:
        role = msg["role"]
        content = msg["content"][0]["text"] if msg["content"] else ""
        conversation_history += f"{role.capitalize()}: {content}\n"

    locations_context = format_locations(request.locations)
    use_cases_context = format_use_cases(request.useCases)

    system_context = f"""
    Context:
    - Sector: {system_role.get('sectorReadable', 'Technology')}
    - Subsector: {system_role.get('subsectorReadable', 'Not specified')}
    - Jurisdictions: {system_role.get('jurisdictionReadable', '')}
    - Roles: {system_role.get('rolesReadable', 'Consumer')}
    - Use Cases: {system_role.get('useCasesReadable', 'Not specified')}

    Locations:
    {locations_context}

    Use Cases:
    {use_cases_context}
    """

    prompt = f"""{system_context}
{document_context}
Previous Conversation:
{conversation_history}
Current User Message: {request.message}
Please provide a helpful response based on the context, document (if available), and conversation history.
Format your response appropriately and maintain consistency with the established context."""

    ai_response = direct_api_call_optimized(prompt, api_key)

    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": ai_response}]
    }

    chat_history["conversation_context"].append(assistant_message)
    chat_history["assistant_response"] = [{"type": "text", "text": ai_response}]
    chat_history["updated_at"] = time.time()
    chat_history["system_role"] = system_role

    if request.document_id:
        chat_history["document_id"] = request.document_id

    await save_chat_history_to_db(session_id, chat_history)
    chat_history_cache[session_id] = chat_history

    return JSONResponse(content={
        "response": ai_response,
        "session_id": session_id,
        "document_id": request.document_id,
        "conversation_context": chat_history["conversation_context"],
        "assistant_response": chat_history["assistant_response"]
    })

@app.get("/api/documents")
async def list_documents():
    """List all cached documents"""
    documents = []
    for doc_id, content in enhanced_document_cache.cache.items():
        documents.append({
            "document_id": doc_id,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "content_length": len(content)
        })

    return JSONResponse(content={
        "documents": documents,
        "total_count": len(documents)
    })

@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get a specific document by ID"""
    cached_doc = enhanced_document_cache.get(document_id)
    if cached_doc:
        return JSONResponse(content={
            "document_id": document_id,
            "content": cached_doc,
            "content_length": len(cached_doc)
        })

    db_document = await get_document_from_db(document_id)
    if db_document:
        enhanced_document_cache.set(document_id, db_document)
        return JSONResponse(content={
            "document_id": document_id,
            "content": db_document,
            "content_length": len(db_document)
        })

    raise HTTPException(status_code=404, detail="Document not found")

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a cached document"""
    if document_id in enhanced_document_cache.cache:
        del enhanced_document_cache.cache[document_id]

    if document_id in enhanced_document_cache.access_times:
        del enhanced_document_cache.access_times[document_id]

    return JSONResponse(content={
        "message": f"Document {document_id} deleted from cache if it existed"
    })

@app.get("/api/analysis/status/{request_id}")
async def get_analysis_status(request_id: str):
    """Get real analysis progress"""
    if request_id not in analysis_progress:
        return {"status": "not_found", "request_id": request_id}

    progress = analysis_progress[request_id]
    completion_percentage = int((progress["completed_chunks"] / progress["total_chunks"]) * 100)

    return {
        "status": progress.get("status", "in_progress"),
        "request_id": request_id,
        "completion_percentage": completion_percentage,
        "completed_chunks": progress["completed_chunks"],
        "total_chunks": progress["total_chunks"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "connected" if db_pool else "not configured"
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "cached_documents": len(enhanced_document_cache.cache),
        "cached_chat_sessions": len(chat_history_cache),
        "version": "2.0.0"
    })

@app.post("/api/analyze/bulk")
async def analyze_bulk_documents(request: PolicyAnalysisFileRequest):
    """Bulk analyze up to 1000 documents with RAG pipeline"""
    try:
        api_key = get_api_key(request.api_key)
        if not api_key:
            raise HTTPException(status_code=400, detail="API key required")
        if not request.storj_urls:
            raise HTTPException(status_code=400, detail="No URLs provided")

        if len(request.storj_urls) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 documents allowed")

        request_id = generate_unique_id()

        # Initialize progress tracking
        analysis_progress[request_id] = {
            'status': 'starting',
            'total_documents': len(request.storj_urls),
            'processed_documents': 0,
            'failed_documents': 0,
            'batch_size': 20,
            'start_time': time.time()
        }

        # Process in background
        asyncio.create_task(process_bulk_documents_async(
            request.storj_urls, request, api_key, request_id
        ))

        return JSONResponse(content={
            "request_id": request_id,
            "status": "processing_started",
            "total_documents": len(request.storj_urls),
            "estimated_time_minutes": len(request.storj_urls) * 0.5,  # Rough estimate
            "check_status_url": f"/api/bulk/status/{request_id}"
        })

    except Exception as e:
        logger.error(f"Bulk analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bulk/status/{request_id}")
async def get_bulk_processing_status(request_id: str):
    """Get status of bulk processing"""
    if request_id not in analysis_progress:
        raise HTTPException(status_code=404, detail="Request ID not found")

    progress = analysis_progress[request_id]

    # Calculate progress percentage
    if progress['total_documents'] > 0:
        progress_percentage = (progress['processed_documents'] / progress['total_documents']) * 100
    else:
        progress_percentage = 0

    response = {
        "request_id": request_id,
        "status": progress['status'],
        "progress_percentage": round(progress_percentage, 2),
        "processed_documents": progress['processed_documents'],
        "total_documents": progress['total_documents'],
        "failed_documents": progress['failed_documents']
    }

    if 'start_time' in progress:
        elapsed_time = time.time() - progress['start_time']
        response['elapsed_time_seconds'] = round(elapsed_time, 2)

        if progress['processed_documents'] > 0:
            avg_time_per_doc = elapsed_time / progress['processed_documents']
            remaining_docs = progress['total_documents'] - progress['processed_documents']
            estimated_remaining = remaining_docs * avg_time_per_doc
            response['estimated_remaining_seconds'] = round(estimated_remaining, 2)

    if progress['status'] == 'error':
        response['error'] = progress.get('error', 'Unknown error')

    return JSONResponse(content=response)

@app.post("/api/chat/rag", response_model=ChatResponse)
async def rag_enhanced_chat(request: ChatRequest):
    """RAG-enhanced chat endpoint"""
    api_key = get_api_key(request.api_key)
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")

    session_id = request.session_id or generate_session_id()

    # Use RAG for better context retrieval
    rag_result = await enhanced_rag_chat(
        query=request.message,
        document_ids=[request.document_id] if request.document_id else None,
        session_id=session_id,
        api_key=api_key
    )

    # Get or create chat history
    chat_history = await get_chat_history_from_db(session_id)
    if not chat_history:
        chat_history = {
            "conversation_context": [],
            "assistant_response": [],
            "system_role": create_system_role(
                request.sector, request.subsector,
                request.jurisdictions, request.roles, request.use_cases
            ),
            "document_id": request.document_id,
            "updated_at": time.time()
        }

    # Add user message
    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": request.message}]
    }
    chat_history["conversation_context"].append(user_message)

    # Add assistant response
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": rag_result['response']}]
    }
    chat_history["conversation_context"].append(assistant_message)
    chat_history["assistant_response"] = [{"type": "text", "text": rag_result['response']}]

    # Save to database
    await save_chat_history_to_db(session_id, chat_history)

    return JSONResponse(content={
        "response": rag_result['response'],
        "session_id": session_id,
        "document_id": request.document_id,
        "conversation_context": chat_history["conversation_context"],
        "assistant_response": chat_history["assistant_response"],
        "rag_info": {
            "sources_used": rag_result['sources_used'],
            "relevant_chunks": rag_result.get('relevant_chunks', [])
        }
    })

@app.post("/api/search")
async def search_documents(query: str, top_k: int = 10, document_ids: List[str] = None):
    """Search across all documents using RAG"""
    try:
        relevant_chunks = await retrieve_relevant_chunks(query, top_k=top_k)

        if document_ids:
            relevant_chunks = [
                chunk for chunk in relevant_chunks
                if chunk['document_id'] in document_ids
            ]

        return JSONResponse(content={
            "query": query,
            "total_results": len(relevant_chunks),
            "results": [
                {
                    "document_id": chunk['document_id'],
                    "score": chunk['score'],
                    "text_preview": chunk['text'][:200] + "...",
                    "chunk_index": chunk['chunk_index']
                } for chunk in relevant_chunks
            ]
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cleanup")
async def cleanup_cache():
    """Cleanup old cache entries"""
    try:
        # Clean document cache
        enhanced_document_cache.clear_old_entries(max_age_hours=24)

        # Clean progress tracking
        current_time = time.time()
        old_progress_keys = [
            key for key, progress in analysis_progress.items()
            if current_time - progress.get('start_time', current_time) > 3600  # 1 hour
        ]

        for key in old_progress_keys:
            del analysis_progress[key]

        return JSONResponse(content={
            "message": "Cache cleanup completed",
            "active_documents": len(enhanced_document_cache.cache),
            "active_progress_items": len(analysis_progress)
        })

    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
