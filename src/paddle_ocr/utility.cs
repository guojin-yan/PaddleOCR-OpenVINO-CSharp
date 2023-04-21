using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace paddleocr
{
    public struct OCRPredictResult
    {
        List<Rect> box = new List<Rect>();
        string text = "";
        float score = -1.0f;
        float cls_score = -1.0f;
        int cls_label = -1;

        public OCRPredictResult()
        {
        }
    }
    struct StructurePredictResult
    {
        List<float> box = new List<float>();
        List<Rect> cell_box = new List<Rect>();
        string type = "";
        List<OCRPredictResult> text_res = new List<OCRPredictResult>();
        string html = "";
        float html_score = -1.0f;
        float confidence = -1.0f;

        public StructurePredictResult()
        {
        }
    }

    public enum EnumDataType
    {
        Normal_Standard_Deviation = 0,
        Normal_Normalization = 1,
        Normal_Non = 2,
        Affine_Standard_Deviation = 3,
        Affine_Normalization = 4,
        Affine_Non = 5,
    }
}
