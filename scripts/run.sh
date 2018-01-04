# Note: all paths referenced here are relative to the Docker container.
#
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
source /tools/config.sh
source activate py27
cd /storage/home/karthikt/image_registration
python scripts/train_dqn.py

