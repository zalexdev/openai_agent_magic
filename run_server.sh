#!/bin/bash
# Script to run the OpenAI-compatible API server

echo "Starting OpenAI-Compatible API with Tavily Search..."
echo "API will be available at http://localhost:8000"
echo "Docs available at http://localhost:8000/docs"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the server
python -m src.main
