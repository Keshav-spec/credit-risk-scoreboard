import sqlite3
import pandas as pd

#    
# CONNECT TO DATABASE
#    

conn = sqlite3.connect("data/credit_risk.db")

#    
# READ SQL FILE
#    

with open("sql/analysis_queries.sql", "r") as file:
    sql_script = file.read()

# Split queries by semicolon
queries = sql_script.split(";")

#    
# EXECUTE QUERIES
#    

for i, query in enumerate(queries):

    query = query.strip()

    if query:

        print("\n" + "=" * 60)
        print(f"QUERY {i+1}")
        print("=" * 60)

        try:
            result = pd.read_sql_query(query, conn)

            print(result)

        except Exception as e:
            print(f"Error: {e}")

#    
# CLOSE CONNECTION
#    

conn.close()

print("\nAll SQL queries executed successfully.")