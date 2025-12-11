import os
import sys
from app import app

# Run the app in debug mode to see errors
print("Starting app with error capture...")
app.run(debug=True, use_reloader=False)
