#!/bin/bash

# Deployment script for PEX-A on Cardano

echo "🚀 Deploying PEX-A to Cardano Network..."

# Load environment variables
source .env

# Build and compile Aiken contracts
echo "📝 Compiling Aiken contracts..."
cd contracts
aiken build
cd ..

# Build Docker images
echo "🐳 Building Docker images..."
docker-compose build

# Deploy to server
echo "📤 Deploying to production..."
docker-compose up -d

# Wait for services to start
sleep 30

# Run database migrations
echo "🗃️ Running database migrations..."
docker-compose exec backend alembic upgrade head

# Verify deployment
echo "✅ Verifying deployment..."
curl -f http://localhost:3000 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "🎉 PEX-A successfully deployed!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
else
    echo "❌ Deployment failed!"
    exit 1
fi