import json
from kafka import KafkaConsumer

if __name__ == "__main__":
    consumer = KafkaConsumer("ftde01-project4", bootstrap_servers='localhost')
    print("Starting the consumer")
    
    for msg in consumer:
        print(f"Records = {json.loads(msg.value)}")
        
        # Parsing pesan Kafka
        data = json.loads(msg.value)