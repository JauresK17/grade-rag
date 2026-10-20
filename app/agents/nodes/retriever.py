import logfire 
from app.agents.state import AgentState 
from app.services.retrieval.qdrant_service import search_entreprise_knowledge 
from app.services.retrieval.ranking_service import reranker_documents 

def retrieve_node(state:AgentState):
    """
    Performs vector search and semantic reranking for technical queries"""

    query = state["current_query"]

    #Standard Retrieval Logic 
    with logfire.span("Knowledge Retrieval "):
        logfire.info(f"Searching Qdrant for : {query}")
        raw_results = search_entreprise_knowledge(query,limit=15)
        logfire.info(f"Retrieval {len(raw_results)} candidates from Vector DB")

        doc_content = [doc['content'] for doc in raw_results]

        with logfire.span("Semantic Reranking"):
            reranked_contents = reranker_documents(query, doc_content, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks")

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

    return {
        "documents":formatted_docs,
        "status":f"Found technical context.",
        "plan":state["plan"] + ["content Retrieved"]
    }