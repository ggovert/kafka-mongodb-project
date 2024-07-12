from pymongo import MongoClient

def test_mongo_connection():
    try:
        # Replace 'admin' and 'password' with your actual username and password
        client = MongoClient('mongodb://admin:password@localhost:27017/')
        db = client.admin
        
        # The 'ping' command is a way to check if the server is reachable
        server_status = db.command("ping")
        print("MongoDB connection successful:", server_status)

        # Optionally, you can list databases
        databases = client.list_database_names()
        print("Databases:", databases)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    test_mongo_connection()
