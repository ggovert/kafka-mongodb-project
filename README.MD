1. Create networks for Docker
```bash
docker network create -d bridge dataninja-networks
```

Running Kafka
```bash
docker-compose -f ./docker/docker-compose-kafka-dc.yml --env-file .env up -d
```

Running MongoDB
```bash
docker-compose -f ./docker/docker-compose-mongo-dc.yml --env-file .env up -d
```