#!/bin/bash
# Fix dpkg and re-run installation

echo "Fixing broken dpkg..."
sudo dpkg --configure -a || true

echo ""
echo "Re-running installation script..."
bash /home/roch/install_pi_ncnn.sh
