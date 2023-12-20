using OpenCvSharp;

namespace PaddleOCR
{
    public class OCRPredictor
    {
        OcrDet ocrDet;
        OcrCls ocrCls;
        OcrRec ocrRec;
        bool flag_det = false;
        bool flag_rec = false;
        bool flag_cls = false;

        public OCRPredictor(string det_model = null, string cls_model = null, string rec_model = null)
        {
            if (det_model != null) 
            {
                flag_det = true;
                ocrDet = new OcrDet(det_model);
            }

            if (cls_model != null)
            {
                flag_cls = true;
                ocrCls = new OcrCls(cls_model);
            }
            if (rec_model != null)
            {
                flag_rec = true;
                ocrRec = new OcrRec(rec_model);
            }

            
        }

        public List<OCRPredictResult> predict_det(Mat image)
        {
            // 文本区域识别
            List<OCRPredictResult> ocr_results = new List<OCRPredictResult>();
            predict_det(image, ocr_results);
            return ocr_results;
        }

        public List<OCRPredictResult> predict_det(Mat image, List<OCRPredictResult> ocr_results) 
        {
            // 文字区域识别
            List<List<List<int>>> boxes = ocrDet.predict(image);

            for (int i = 0; i < boxes.Count; i++)
            {
                OCRPredictResult res = new OCRPredictResult();
                res.box = boxes[i];
                ocr_results.Add(res);
            }
            return  PaddleOcrUtility.sorted_boxes(ocr_results);
        }

        public List<OCRPredictResult> predict_cls(List<Mat> img_list)
        {
            List<OCRPredictResult> ocr_results = new List<OCRPredictResult>();
            // 文字方向判断
            predict_cls(img_list, ocr_results);
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


        public List<OCRPredictResult> predict_rec(List<Mat> img_list)
        {
            List<OCRPredictResult> ocr_results = new List<OCRPredictResult>();
            predict_rec(img_list, ocr_results);
            return ocr_results;
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
        public List<OCRPredictResult> predict_all(Mat image)
        {
            DateTime start = DateTime.Now;

            // 文本区域识别
            List<OCRPredictResult> ocr_results = new List<OCRPredictResult>();
            // 文字区域识别
            List<List<List<int>>> boxes = ocrDet.predict(image);

            for (int i = 0; i < boxes.Count; i++)
            {
                OCRPredictResult res = new OCRPredictResult();
                res.box = boxes[i];
                ocr_results.Add(res);
            }


            // crop image
            List<Mat> img_list = new List<Mat>();
            for (int j = 0; j < ocr_results.Count; j++)
            {
                Mat crop_img = PaddleOcrUtility.get_rotate_crop_image(image, ocr_results[j].box);
                img_list.Add(crop_img);
                //Cv2.ImShow("resize_img", crop_img);
                //Cv2.WaitKey(0);
            }


            // 文字方向判断
            List<int> lables = new List<int>();
            List<float> scores = new List<float>();
            ocrCls.predict(img_list, lables, scores);
            // output cls results
            for (int i = 0; i < lables.Count; i++)
            {
                ocr_results[i].cls_label = lables[i];
                ocr_results[i].cls_score = scores[i];
            }

            // 图片翻转
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

            List<string> rec_texts = new List<string>(new string[img_list.Count]);
            List<float> rec_text_scores = new List<float>(new float[img_list.Count]);
            ocrRec.predict(img_list, rec_texts, rec_text_scores);
            for (int i = 0; i < rec_texts.Count; i++)
            {
                ocr_results[i].text = rec_texts[i];
                ocr_results[i].score = rec_text_scores[i];
            }

            //DateTime end = DateTime.Now;
            //TimeSpan timeSpan = end - start;
            //Console.WriteLine("图片预测时间：{0} ms", timeSpan.TotalMilliseconds);

            return ocr_results;
        }

        public List<OCRPredictResult> predict(Mat image) 
        {
            if (flag_det && flag_cls && flag_rec)
            {
                return predict_all(image);
            }
            else if (flag_det && !flag_cls && !flag_rec)
            {
                return predict_det(image);
            }
            else if (!flag_det && flag_cls && !flag_rec)
            {
                List<Mat> img_list = new List<Mat>();
                return predict_cls(img_list);
            }
            else if (!flag_det && !flag_cls && flag_rec)
            {
                List<Mat> img_list = new List<Mat>();
                return predict_rec(img_list);
            }
            else if (!flag_det && flag_cls && flag_rec)
            {
                List<OCRPredictResult> ocr_results = new List<OCRPredictResult>();
                List<Mat> img_list = new List<Mat>();
                img_list.Add(image);
                // 文字方向判断
                List<int> lables = new List<int>();
                List<float> scores = new List<float>();
                ocrCls.predict(img_list, lables, scores);
                // output cls results
                for (int i = 0; i < lables.Count; i++)
                {
                    ocr_results[i].cls_label = lables[i];
                    ocr_results[i].cls_score = scores[i];
                }

                // 图片翻转
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

                List<string> rec_texts = new List<string>(new string[img_list.Count]);
                List<float> rec_text_scores = new List<float>(new float[img_list.Count]);
                ocrRec.predict(img_list, rec_texts, rec_text_scores);
                for (int i = 0; i < rec_texts.Count; i++)
                {
                    ocr_results[i].text = rec_texts[i];
                    ocr_results[i].score = rec_text_scores[i];
                }

                return ocr_results;
            }
            else if (flag_det && !flag_cls && flag_rec)
            {
                // 文本区域识别
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


                // crop image
                List<Mat> img_list = new List<Mat>();
                for (int j = 0; j < ocr_results.Count; j++)
                {
                    Mat crop_img = PaddleOcrUtility.get_rotate_crop_image(image, ocr_results[j].box);
                    img_list.Add(crop_img);
                }

                List<string> rec_texts = new List<string>(new string[img_list.Count]);
                List<float> rec_text_scores = new List<float>(new float[img_list.Count]);
                ocrRec.predict(img_list, rec_texts, rec_text_scores);
                for (int i = 0; i < rec_texts.Count; i++)
                {
                    ocr_results[i].text = rec_texts[i];
                    ocr_results[i].score = rec_text_scores[i];
                }

                return ocr_results;
            }
            else if (flag_det && flag_cls && !flag_rec)
            {
                // 文本区域识别
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


                // crop image
                List<Mat> img_list = new List<Mat>();
                for (int j = 0; j < ocr_results.Count; j++)
                {
                    Mat crop_img = PaddleOcrUtility.get_rotate_crop_image(image, ocr_results[j].box);
                    img_list.Add(crop_img);
                }

                // 文字方向判断
                List<int> lables = new List<int>();
                List<float> scores = new List<float>();
                ocrCls.predict(img_list, lables, scores);
                // output cls results
                for (int i = 0; i < lables.Count; i++)
                {
                    ocr_results[i].cls_label = lables[i];
                    ocr_results[i].cls_score = scores[i];
                }

                // 图片翻转
                for (int i = 0; i < img_list.Count; i++)
                {
                    if (ocr_results[i].cls_label % 2 == 0 &&
                        ocr_results[i].cls_score > ocrCls.m_cls_thresh) { }
                    else
                    {
                        Cv2.Rotate(img_list[i], img_list[i], RotateFlags.Rotate180);
                    }
                }

                return ocr_results;
            }
            else
            {
                return new List<OCRPredictResult>();
            }
        }
        public List<List<OCRPredictResult>> prdict(List<Mat> images) 
        {
            List<List<OCRPredictResult>> ocr_results = new List<List<OCRPredictResult>>();
            foreach (Mat image in images) 
            {
                ocr_results.Add(predict(image));
            }
            return ocr_results;
        }
    }
}
