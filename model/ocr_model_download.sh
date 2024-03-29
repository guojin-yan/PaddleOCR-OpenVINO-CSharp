DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# Download the data.
echo "Downloading..."
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
# Extract the data.
mkdir paddle
echo "Extracting..."
tar -xvf ch_PP-OCRv4_det_infer.tar -C ./paddle
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar -C ./paddle
tar -xvf ch_PP-OCRv4_rec_infer.tar -C ./paddle