import os
import logging
import argparse
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def query_entity(entity_name: str):
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        logger.error("Missing Neo4j configuration.")
        return

    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        
        print(f"\n=== Investigating Entity: '{entity_name}' ===\n")

        # 1. Who is this? (Neighbors)
        print("--- Direct Connections (Who/What are they linked to?) ---")
        # Find everything connected to a node with this ID (case-insensitive fuzzy match might be better, but exact ID is safer if known)
        # We'll try exact match on ID first.
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.id CONTAINS $name
        RETURN type(r) as rel, labels(m)[0] as type, m.id as target, startNode(r) = n as is_source
        LIMIT 20
        """
        results = graph.query(query, params={"name": entity_name})
        
        if not results:
            print(f"No direct connections found for '{entity_name}'.")
        else:
            for res in results:
                direction = "-->" if res['is_source'] else "<--"
                print(f"(Self) {direction} [{res['rel']}] {direction} {res['target']} ({res['type']})")

        # 2. What documents mention them?
        print("\n--- Mentioned in Documents ---")
        doc_query = """
        MATCH (d:Document)-[:MENTIONS]->(n)
        WHERE n.id CONTAINS $name
        RETURN d.title as title, d.topic as topic, d.text as text
        LIMIT 3
        """
        docs = graph.query(doc_query, params={"name": entity_name})
        
        if not docs:
            print("No documents explicitly link to this entity via MENTIONS relationship.")
        else:
            for doc in docs:
                print(f"Title: {doc['title']}")
                print(f"Topic: {doc['topic']}")
                print(f"Context: {doc['text'][:200]}...")
                print("-" * 40)

    except Exception as e:
        logger.error(f"Failed to query entity: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investigate a specific entity in the graph.")
    parser.add_argument("name", help="Name of the entity to investigate")
    args = parser.parse_args()
    
    query_entity(args.name)

