# Note: all paths referenced here are relative to the Docker container.
#
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/tools/caffes/dependencies/others:$LD_LIBRARY_PATH"
export PYTHONPATH="/storage/home/karthikt/baselines":"/storage/home/karthikt/image_registration/imgreg":$PYTHONPATH
source /tools/config.sh
source activate py35
cd /storage/home/karthikt/image_registration
python -u scripts/train_dqn_OpenAI.py &> logs > output5.1
