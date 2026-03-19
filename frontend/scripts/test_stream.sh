#!/bin/bash
#frontend/scripts/test_stream.sh
echo "Testing Streaming Endpoint (/ask?stream=true)..."
echo "You should see multiple 'data:' lines followed by '[DONE_JSON]'"
echo "------------------------------------------------------------"

curl -N -X POST "http://127.0.0.1:8000/ask?stream=true" \
     -H "Content-Type: application/json" \
     -d '{
           "user_query": "cheap flight delhi to mumbai tomorrow",
           "origin": "DEL",
           "destination": "BOM",
           "date": "2026-03-15"
         }'
echo -e "\n------------------------------------------------------------"
echo "Stream test complete."