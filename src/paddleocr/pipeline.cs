using OpenCvSharp;
using OpenVinoSharp.Extensions.utility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace OpenVinoSharp.Extensions.model.PaddleOCR
{

    public static class Pipeline
    {
        public static async Task<OnlineOcr> GetOnlineOCR(Language language = Language.ch_PP_OCRv4, bool det = true, bool rec = true, bool cls = true)
        {
            OcrModel model = await OcrModel.GetOnlineOcrModel(language, det, cls, rec);
            return new OnlineOcr(model);
        }
    }
    public class OnlineOcr
    {
        OCRPredictor predictor;
        public OnlineOcr(OcrModel model)
        {
            OcrConfig config = new OcrConfig(model);
            predictor = new OCRPredictor(config);
        }

        public List<OCRPredictResult> predict(Mat img, bool det = true, bool rec = true, bool cls = true)
        {
            return predictor.ocr(img, det, rec, cls);
        }

        public Tuple<List<OCRPredictResult>, Mat> ocr_test()
        {
            string url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/image/demo_1.jpg";
            Uri uri = new Uri(url);
            string file_name = System.IO.Path.GetFileName(uri.LocalPath);
            string file_path = Path.Combine("./", file_name);
            if (!File.Exists(file_path))
                _ = Download.download_file_async(url, file_path).Result;
            Mat img = Cv2.ImRead(file_path);
            List<OCRPredictResult> result = predict(img);
            return new Tuple<List<OCRPredictResult>, Mat>(result, img);
        }
    }
}
