from sqlalchemy import create_engine, inspect

# Use the same connection string you constructed earlier
DB_CONNECTION_STRING = "postgresql+psycopg2://sql_agent_user:N0cturn3112@localhost:5432/adventureworks"

print("Attempting to connect to the database...")

try:
    # Create a database engine
    engine = create_engine(DB_CONNECTION_STRING)

    # Connect and inspect the database
    with engine.connect() as connection:
        print("✅ Connection successful!")
        
        inspector = inspect(engine)
        
        print("\nFound tables in the database:")
        
        # Get all schema names
        schema_names = inspector.get_schema_names()
        
        # Loop through each schema to find tables
        for schema in schema_names:
            # We want to ignore the system schemas
            if not schema.startswith('pg_') and schema != 'information_schema':
                print(f"\n--- Tables in schema: {schema} ---")
                table_names = inspector.get_table_names(schema=schema)
                if not table_names:
                    print(" (No tables found)")
                else:
                    for name in table_names:
                        print(f"- {name}")

except Exception as e:
    print(f"❌ Connection failed. Error: {e}")