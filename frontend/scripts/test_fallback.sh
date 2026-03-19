#!/bin/bash
#frontend/scripts/test_fallback.sh
echo "Testing Standard Fallback Endpoint (/ask)..."
echo "You should see a single, complete JSON response (no 'data:' prefixes)."
echo "------------------------------------------------------------"

curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "user_query": "cheap flight delhi to mumbai tomorrow",
           "origin": "DEL",
           "destination": "BOM",
           "date": "2026-03-15"
         }'
echo -e "\n------------------------------------------------------------"
echo "Fallback test complete."