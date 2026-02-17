"""Fix corrupted width parameters in dashboard.py"""
import re

# Read the file
with open('dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the malformed patterns
# Pattern 1: width=" stretch\)
content = re.sub(r'width="\s*stretch\\+\)', 'width="stretch")', content)

# Pattern 2: Any other malformed width=" patterns
content = re.sub(r'width="\s*stretch\\*(?!")', 'width="stretch"', content)

# Write back
with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed all corrupted width parameters")
