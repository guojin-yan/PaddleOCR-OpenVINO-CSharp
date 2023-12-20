DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/Modelv4/ppstructure_mobile_v2.0_SLANet_infer.zip
wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar
# Extract the data.
mkdir paddle
echo "Extracting..."
tar -xvf ppstructure_mobile_v2.0_SLANet_infer.tar -C ./paddle
tar -xvf picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar -C ./paddle