using OpenCvSharp;

namespace paddleocr
{
    public class PaddleOCR
    {
        OcrDet ocrDet;
        OcrCls ocrCls;
        OcrRec ocrRec;
        public PaddleOCR(Dictionary<string, string> model_path) 
        {
            ocrDet = new OcrDet(model_path["det_model"], "CPU", "x", "sigmoid_0.tmp_0",
            new ulong[] { 1, 3, 640, 640 }, EnumDataType.Normal_Standard_Deviation, 0.3, 0.5);

            ocrCls = new OcrCls(model_path["cls_model"], "CPU", "x", "softmax_0.tmp_0",
                new ulong[] { 1, 3, 640, 640 }, EnumDataType.Normal_Standard_custom_Deviation, 0.9);

            ocrRec = new OcrRec(model_path["rec_model"], "CPU", "x", "softmax_5.tmp_0",
            new ulong[] { 1, 3, 48, 320 }, EnumDataType.Normal_Standard_custom_Deviation, @".\..\..\..\..\..\model\ppocr_keys_v1.txt");
        }

        public List<OCRPredictResult> predict_det(Mat image) 
        {
            List<OCRPredictResult> ocr_results = new List<OCRPredictResult>();
            // 文字区域识别
            List<List<List<int>>> boxes = ocrDet.predict(image);

            for (int i = 0; i < boxes.Count; i++)
            {
                OCRPredictResult res = new OCRPredictResult();
                res.box = boxes[i];
                ocr_results.Add(res);
            }
            ocr_results = ocr_results.OrderBy(t => t.box[0][1]).ThenBy(t => t.box[0][0]).ToList();
            return ocr_results;
        }
        public void predict_cls(List<Mat> img_list, List<OCRPredictResult> ocr_results) 
        {
            List<int> lables = new List<int>();
            List<float> scores = new List<float>();
            ocrCls.predict(img_list, lables, scores);
            // output cls results
            for (int i = 0; i < lables.Count; i++)
            {
                ocr_results[i].cls_label = lables[i];
                ocr_results[i].cls_score = scores[i];
            }
        }

        public void predict_rec(List<Mat> img_list, List<OCRPredictResult> ocr_results) 
        {
            List<string> rec_texts = new List<string>();
            List<float> rec_text_scores = new List<float>();
            ocrRec.predict(img_list, rec_texts, rec_text_scores);
            for (int i = 0; i < rec_texts.Count; i++)
            {
                ocr_results[i].text = rec_texts[i];
                ocr_results[i].score = rec_text_scores[i];
            }
        }

        public void predict(Mat image) 
        {
            DateTime start = DateTime.Now;

            // 文本区域识别
            List<OCRPredictResult> ocr_results = predict_det(image);
            // crop image
            List<Mat> img_list = new List<Mat>();
            for (int j = 0; j < ocr_results.Count; j++)
            {
                Mat crop_img = Utility.GetRotateCropImage(image, ocr_results[j].box);
                img_list.Add(crop_img);
                //Cv2.ImShow("resize_img", crop_img);
                //Cv2.WaitKey(0);
            }
            // 文字方向判断
            predict_cls(img_list, ocr_results);
            for (int i = 0; i < img_list.Count; i++)
            {
                if (ocr_results[i].cls_label % 2 == 1 &&
                    ocr_results[i].cls_score > ocrCls.m_cls_thresh) { }
                else
                {
                    Cv2.Rotate(img_list[i], img_list[i], RotateFlags.Rotate180);
                }
            }

            predict_rec(img_list, ocr_results);

            DateTime end = DateTime.Now;
            TimeSpan timeSpan = end - start;
            Console.WriteLine("图片预测时间：{0} ms",timeSpan.TotalMilliseconds);

            Utility.print_result(ocr_results);

            // 将文字区域标注出来
            for (int r = 0; r < ocr_results.Count; r++)
            {
                var ocr_result = ocr_results[r];
                var box = ocr_result.box;

                //Console.WriteLine("1({0}, {1})", box[0][0], box[0][1]);
                //Console.WriteLine("2({0}, {1})", box[1][0], box[1][1]);
                //Console.WriteLine("3({0}, {1})", box[2][0], box[2][1]);
                //Console.WriteLine("4({0}, {1})", box[3][0], box[3][1]);

                Rect rect = new Rect(box[0][0], box[0][1], box[3][0] - box[0][0], box[1][1] - box[0][1]);
                Cv2.Rectangle(image, rect, new Scalar(0, 0, 255), 2);
            }



            Cv2.ImShow("image_rect", image);

            Cv2.WaitKey(0);
        }

    }
}