import os
import logging
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def inspect_graph():
    from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        logger.error("Missing Neo4j configuration.")
        return

    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        # Refresh schema to ensure we have the latest view
        graph.refresh_schema()
        
        print("\n=== Neo4j Graph Summary ===\n")

        # 1. Schema Info
        print("--- Node Labels & Relationships ---")
        # graph.schema is a string representation of the schema
        print(graph.schema) 

        # 2. Counts
        print("\n--- Counts ---")
        count_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->()
        RETURN count(distinct n) as nodes, count(r) as relationships
        """
        counts = graph.query(count_query)
        if counts:
            print(f"Nodes: {counts[0]['nodes']}")
            print(f"Relationships: {counts[0]['relationships']}")
        else:
            print("No data found.")

        # 3. Top Connected Entities
        print("\n--- Top 5 Most Connected Entities ---")
        top_nodes_query = """
        MATCH (n)-[r]-()
        RETURN n.id as id, labels(n)[0] as type, count(r) as degree
        ORDER BY degree DESC
        LIMIT 5
        """
        top_nodes = graph.query(top_nodes_query)
        if top_nodes:
            for node in top_nodes:
                node_id = node.get('id', 'Unknown ID')
                node_type = node.get('type', 'Unknown Type')
                degree = node.get('degree', 0)
                print(f"{node_id} ({node_type}): {degree} connections")
        else:
            print("No connected nodes found.")

        # 4. Sample Relationship
        print("\n--- Sample Relationship ---")
        sample_query = """
        MATCH (a)-[r]->(b)
        RETURN a.id, type(r), b.id
        LIMIT 1
        """
        sample = graph.query(sample_query)
        if sample:
            s = sample[0]
            print(f"{s['a.id']} --[{s['type(r)']}]--> {s['b.id']}")
        else:
            print("No relationships found.")

    except Exception as e:
        logger.error(f"Failed to inspect graph: {e}")

if __name__ == "__main__":
    inspect_graph()

