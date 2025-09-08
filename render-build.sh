#!/usr/bin/env bash
# Install Linux packages needed by WeasyPrint
apt-get update
apt-get install -y \
  libpango-1.0-0 \
  libpangocairo-1.0-0 \
  libgdk-pixbuf2.0-0 \
  libcairo2 \
  libffi-dev

# Now install Python packages
pip install -r requirements.txt
