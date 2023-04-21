using OpenCvSharp;

namespace paddleocr
{
    public class PaddleOCR
    {
        OcrDet ocrDet;
        public PaddleOCR(Dictionary<string, string> model_path) 
        {
            ocrDet = new OcrDet(model_path["det_model"], "CPU", "x", "sigmoid_0.tmp_0",
            new ulong[] { 1, 3, 640, 640 }, EnumDataType.Affine_Standard_Deviation, 0.3, 0.5);
        }

        public void predict(Mat image) 
        {
            List<List<List<int>>> boxes = ocrDet.predict(image);
        }

    }
}