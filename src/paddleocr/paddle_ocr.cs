using OpenCvSharp;

namespace PaddleOCR
{
    public class OCRPredictor
    {
        OcrDet ocrDet;
        OcrCls ocrCls;
        OcrRec ocrRec;
        public OCRPredictor(string det_model, string cls_model, string rec_model)
        {
            ocrDet = new OcrDet(det_model);
            ocrCls = new OcrCls(cls_model);
            ocrRec = new OcrRec(rec_model);
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
            List<string> rec_texts = new List<string>(new string[img_list.Count]);
            List<float> rec_text_scores = new List<float>(new float[img_list.Count]);
            ocrRec.predict(img_list, rec_texts, rec_text_scores);
            for (int i = 0; i < rec_texts.Count; i++)
            {
                ocr_results[i].text = rec_texts[i];
                ocr_results[i].score = rec_text_scores[i];
            }
        }

        public List<OCRPredictResult> predict(Mat image)
        {
            DateTime start = DateTime.Now;

            // 文本区域识别
            List<OCRPredictResult> ocr_results = predict_det(image);
            // crop image
            List<Mat> img_list = new List<Mat>();
            for (int j = 0; j < ocr_results.Count; j++)
            {
                Mat crop_img = Utility.get_rotate_crop_image(image, ocr_results[j].box);
                img_list.Add(crop_img);
                //Cv2.ImShow("resize_img", crop_img);
                //Cv2.WaitKey(0);
            }
            // 文字方向判断
            predict_cls(img_list, ocr_results);
            for (int i = 0; i < img_list.Count; i++)
            {
                if (ocr_results[i].cls_label % 2 == 0 &&
                    ocr_results[i].cls_score > ocrCls.m_cls_thresh) { }
                else
                {
                    Cv2.Rotate(img_list[i], img_list[i], RotateFlags.Rotate180);
                }
                //Cv2.ImShow("ss", img_list[i]);
                //Cv2.WaitKey(0);
            }

            predict_rec(img_list, ocr_results);

            //DateTime end = DateTime.Now;
            //TimeSpan timeSpan = end - start;
            //Console.WriteLine("图片预测时间：{0} ms", timeSpan.TotalMilliseconds);

            return ocr_results;
        }

    }
}
