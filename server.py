from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
import os
from rag import RAG, call_gpt
from agents import Agents

from config import *


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def save_files(files: List[UploadFile], dir: str) -> List[str]:
    """
    Saves uploaded files to a given directory and returns their paths.

    Args:
        files: Files to save.
        dir: Path to save files.

    Returns:
        List of paths to saves files.
    """
    file_paths = []
    for file in files:
        try:
            file_path = os.path.join(dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_paths.append(file_path)
        except Exception as e:
            print(str(e))
            continue
    return file_paths


def run_agent(agent_id: str, new: bool = True):
    """
    Runs selected agent.
    If agent is new its data is parsed and chunked first.
    If agent is not new it is ingested from backup.

    Args:
        agent_id: The ID of agent to add.
        new: Whether agent's data is already processed.

    Returns:
        agent: Object of corresponding agent to call.
    """
    global agents_db
    global agents_objects
    agent = RAG()
    if new:
        # parse files to get documents for top level keyword search
        # and chunks for low level keyword and semantic search
        files_dir = agents_db.get_agent_field(agent_id, "files")
        documents, chunks = agent.load_and_split_documents(files_dir)

        # save parsed documents for future loading on agent restart
        documents_dir = DOCUMENTS_DIR_GLOB.format(agent_id=agent_id)
        agent.save_documents(documents, documents_dir)
        agents_db.set_agent_field(agent_id, "documents", documents_dir)
        
        # save parsed chunks for future loading on agent restart
        chunks_dir = CHUNKS_DIR_GLOB.format(agent_id=agent_id)
        agent.save_chunks(chunks, chunks_dir)
        agents_db.set_agent_field(agent_id, "chunks", chunks_dir)

        # create all stores
        agent.create_keywords_store_docs(documents)
        agent.create_keywords_store_chunks(chunks)
        agent.create_vector_store(chunks)

        # save vectors for future loading on agent restart
        vectors_dir = VECTORS_DIR_GLOB.format(agent_id=agent_id)
        agent.save_vector_store(vectors_dir)
        agents_db.set_agent_field(agent_id, "vectors", vectors_dir)
        agents_db.to_json()
    else:
        try:
            # load top level keyword store from backup
            documents_dir = agents_db.get_agent_field(agent_id, "documents")
            documents = agent.load_documents(documents_dir)
            agent.create_keywords_store_docs(documents)

            # load low level keyword store from backup
            chunks_dir = agents_db.get_agent_field(agent_id, "chunks")
            chunks = agent.load_chunks(chunks_dir)
            agent.create_keywords_store_chunks(chunks)

            # load vector store from backup
            vectors_dir = agents_db.get_agent_field(agent_id, "vectors")
            agent.load_vector_store(vectors_dir)
        except Exception as e:
            print(str(e))
    agents_objects[agent_id] = agent


class AgentQuery(BaseModel):
    query: str
    max_attempts: int = 1
    query_expansion: bool = False
    keyword_docs_top_k: int = 2
    keyword_chunks_top_k: int = 20
    vector_top_k: int = 15
    rerank_top_k: int = 10
    weights: list = [0.4, 0.6]
    model: str = "gpt-4.1"
    max_tokens: int = 32768
    temperature: float = 0.2


class LLMQuery(BaseModel):
    input: str
    instruction: str
    model: str = "gpt-4.1"
    max_tokens: int = 32768
    temperature: float = 0.2


@app.post("/llm")
def query_llm(query: LLMQuery):
    if not (query.input or query.instruction):
        raise HTTPException(status_code=400, detail="Both system and user prompts should be given.")
    try:
        response = call_gpt(
            prompt_system=query.instruction, 
            prompt_user=query.input, 
            model=query.model, 
            max_tokens=query.max_tokens,
            temperature=query.temperature
        )
        return {"response": response.strip(), "error": ""}
    except Exception as e:
        return {"response": "", "error": str(e)}


@app.post("/agents")
def add_agent(agent_id: str):
    global agents_db
    agent_added = agents_db.add_agent(agent_id)
    if not agent_added:
        raise HTTPException(status_code=400, detail="Agent with such id already exists.")
    agents_db.to_json()
    return f"Agent {agent_id} created successfully."


@app.get("/agents/{agent_id}")
def show_agent(agent_id: str):
    global agents_db
    files_dir = agents_db.get_agent_field(agent_id, "files")
    if files_dir == "":
        raise HTTPException(status_code=404, detail="Agent not found or it was not ingested with files.")
    try:
        file_names = os.listdir(files_dir)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Agent has no files. {e}")
    return f"Agent {agent_id} is ingested with the following files: {file_names}"


@app.post("/agents/{agent_id}/ingest")
async def ingest_agent(agent_id: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    global agents_db
    global agents_objects
    if not files:
        raise HTTPException(status_code=400, detail="No files transfered.")
    files_dir = FILES_DIR_GLOB.format(agent_id=agent_id)
    os.makedirs(files_dir, exist_ok=True)
    file_paths = await save_files(files, files_dir)
    if len(file_paths) > 0:
        agents_db.set_agent_field(agent_id, "files", files_dir)
        background_tasks.add_task(run_agent, agent_id, True)
        file_names = os.listdir(files_dir)
        return f"Agent {agent_id} is ingested with the following files: {file_names}"
    else:
        raise HTTPException(status_code=400, detail="No files saved.")
    

@app.post("/agents/{agent_id}/query")
def query_agent(agent_id: str, query: AgentQuery):
    global agents_objects
    if agent_id not in agents_objects:
        raise HTTPException(status_code=404, detail="Agent not found.")
    print(query)
    result = agents_objects[agent_id].process_query(
        query=query.query, 
        max_attempts=query.max_attempts,
        query_expansion=query.query_expansion,
        keyword_docs_top_k=query.keyword_docs_top_k,
        keyword_chunks_top_k=query.keyword_chunks_top_k,
        vector_top_k=query.vector_top_k,
        rerank_top_k=query.rerank_top_k,
        weights=query.weights,
        model=query.model, 
        max_tokens=query.max_tokens,
        temperature=query.temperature
    )
    
    print(result)
    sources = []
    if "retrieved_documents" in result and result["retrieved_documents"]:
        for doc, score in result["retrieved_documents"]:
            sources.append({
                "source": doc.metadata.get("source", "Неизвестный источник"),
                "content_preview": doc.page_content, # doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "relevance_score": round(score, 3)
            })
    return {
        "response": result["response"], 
        "attempted_queries": result.get("attempted_queries", []),
        "sources": sources,
        "total_attempts": len(result.get("attempted_queries", [])),
        "final_query": result.get("final_query", query.query)
    }


os.makedirs(AGENTS_DIR_GLOB, exist_ok=True)

agents_db = Agents.from_json(f"{AGENTS_DIR_GLOB}/{AGENTS_FILE_GLOB}")

agents_objects = {}

for agent_id in agents_db.db:
    vectors_dir = agents_db.get_agent_field(agent_id, "vectors")
    files_dir = agents_db.get_agent_field(agent_id, "files")
    if vectors_dir != "":
        run_agent(agent_id, new=False)
    elif files_dir != "":
        run_agent(agent_id, new=True)
print(agents_objects)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
