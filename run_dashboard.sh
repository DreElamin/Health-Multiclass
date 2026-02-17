#!/bin/bash

# Health ML Dashboard Launch Script

echo "🏥 Multi-Class Health Condition Classification Dashboard"
echo "========================================================"
echo ""
echo "Starting dashboard..."
echo ""
echo "The dashboard will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed."
    echo "Installing required dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Run the dashboard
streamlit run dashboard.py
