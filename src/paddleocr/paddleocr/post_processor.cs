using iTextSharp.text.pdf.parser.clipper;
using OpenCvSharp;
using PaddleOCR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace PaddleOCR
{
    public static class PostProcessor
    {

        static int clampi(int x, int min, int max)
        {
            if (x > max)
                return max;
            if (x < min)
                return min;
            return x;
        }

        static float clampf(float x, float min, float max)
        {
            if (x > max)
                return max;
            if (x < min)
                return min;
            return x;
        }

        static void get_contour_area(List<List<float>> box, float unclip_ratio, out float distance)
        {
            int pts_num = 4;
            float area = 0.0f;
            float dist = 0.0f;
            for (int i = 0; i < pts_num; i++)
            {
                area += box[i][0] * box[(i + 1) % pts_num][1] -
                        box[i][1] * box[(i + 1) % pts_num][0];
                dist += (float)Math.Sqrt((box[i][0] - box[(i + 1) % pts_num][0]) *
                                  (box[i][0] - box[(i + 1) % pts_num][0]) +
                              (box[i][1] - box[(i + 1) % pts_num][1]) *
                                  (box[i][1] - box[(i + 1) % pts_num][1]));
            }
            
            area = Math.Abs((float)(area / 2.0));

            distance = area * unclip_ratio / dist;
            
        }

        static RotatedRect unclip(List<List<float>> box, float unclip_ratio)
        {
            float distance = 1.0f;

            get_contour_area(box, unclip_ratio, out distance);
            

            ClipperOffset offset = new ClipperOffset();

            List<IntPoint> path = new List<IntPoint> { new IntPoint((int)box[0][0], (int)box[0][1]),
            new IntPoint((int)box[1][0], (int)box[1][1]), new IntPoint((int)box[2][0], (int)box[2][1]),
            new IntPoint((int)box[3][0], (int)box[3][1])};

            offset.AddPath(path, JoinType.jtRound, EndType.etClosedPolygon);
            List<List<IntPoint>> paths = new List<List<IntPoint>>();
            offset.Execute(ref paths, distance);
            List<Point2f> points = new List<Point2f>();

            for (int j = 0; j < paths.Count(); j++)
            {
                for (int i = 0; i < paths[paths.Count() - 1].Count(); i++)
                {
                    points.Add(new Point2f(paths[j][i].X, paths[j][i].Y));
                }
            }
            RotatedRect res;
            if (points.Count() <= 0)
            {
                
                res = new RotatedRect(new Point2f(0, 0), new Size2f(1, 1), 0);
            }
            else
            {
                res = Cv2.MinAreaRect(points);
            }
            return res;
        }


        static List<List<int>> order_points_clockwise(List<List<int>> pts)
        {
            List<List<int>> box = pts;
            box = box.OrderBy(t => t[0]).ToList();
            List<List<int>> leftmost = new List<List<int>> { box[0], box[1] };
            List<List<int>> rightmost = new List<List<int>> { box[2], box[3] };

            List<List<int>> rect = new List<List<int>>();
            if (leftmost[0][1] > leftmost[1][1])
            {
                rect.Add(leftmost[1]);
                rect.Add(leftmost[0]);
            }
            else 
            {
                rect.Add(leftmost[0]);
                rect.Add(leftmost[1]);
            }

            if (rightmost[0][1] > rightmost[1][1])
            {
                rect.Add(rightmost[1]);
                rect.Add(rightmost[0]);
            }
            else 
            {
                rect.Add(rightmost[0]);
                rect.Add(rightmost[1]);
            }
            return rect;
        }

        static List<List<float>> mat_to_list(Mat mat)
        {
            List<List<float>> img_vec = new List<List<float>>();

            for (int i = 0; i < mat.Rows; ++i)
            {
                List<float> tmp = new List<float>();
                for (int j = 0; j < mat.Cols; ++j)
                {
                    tmp.Add(mat.At<float>(i, j));
                }
                img_vec.Add(tmp);
            }
            return img_vec;
        }


        static List<List<float>> get_mini_boxes(RotatedRect box, out float ssid)
        {
            ssid = Math.Max(box.Size.Width, box.Size.Height);

            Mat points = new Mat();
            Cv2.BoxPoints(box, points);


            var array = mat_to_list(points);

            array = array.OrderBy(t => t[0]).ToList();//升序

            List<float> idx1 = array[0], idx2 = array[1], idx3 = array[2], idx4 = array[3];
            if (array[3][1] <= array[2][1])
            {
                idx2 = array[3];
                idx3 = array[2];
            }
            else
            {
                idx2 = array[2];
                idx3 = array[3];
            }
            if (array[1][1] <= array[0][1])
            {
                idx1 = array[1];
                idx4 = array[0];
            }
            else
            {
                idx1 = array[0];
                idx4 = array[1];
            }

            array[0] = idx1;
            array[1] = idx2;
            array[2] = idx3;
            array[3] = idx4;

            return array;
        }

        static float polygon_score_acc(Point[] contour, Mat pred)
        {
            int width = pred.Cols;
            int height = pred.Rows;
            List<float> box_x = new List<float>();
            List<float> box_y = new List<float>();
            for (int i = 0; i < contour.Length; ++i)
            {
                box_x.Add(contour[i].X);
                box_y.Add(contour[i].Y);
            }

            int xmin = clampi((int)Math.Floor(box_x.Min()), 0, width - 1);
            int xmax = clampi((int)Math.Ceiling(box_x.Max()), 0, width - 1);
            int ymin = clampi((int)Math.Floor(box_y.Min()), 0, height - 1);
            int ymax = clampi((int)Math.Ceiling(box_y.Max()), 0, height - 1);

            Mat mask = new Mat();
            mask = Mat.Zeros(ymax - ymin + 1, xmax - xmin + 1, MatType.CV_8UC1);

            Point[] rook_point = new Point[contour.Length];

            for (int i = 0; i < contour.Length; ++i)
            {
                rook_point[i] = new Point((int)box_x[i] - xmin, (int)box_y[i] - ymin);
            }
            Point[][] ppt = new Point[1][] { rook_point };


            Cv2.FillPoly(mask, ppt, new Scalar(1));

            Mat croppedImg = new Mat(pred.Clone(), new Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
            float score = (float)Cv2.Mean(croppedImg, mask)[0];
            return score;
        }
        static float box_score_fast(List<List<float>> box_array, Mat pred)
        {
            var array = box_array;
            int width = pred.Cols;
            int height = pred.Rows;

            List<float> box_x = new List<float> { array[0][0], array[1][0], array[2][0], array[3][0] };
            List<float> box_y = new List<float> { array[0][1], array[1][1], array[2][1], array[3][1] };

            int xmin = clampi((int)Math.Floor(box_x.Min()), 0, width - 1);
            int xmax = clampi((int)Math.Ceiling(box_x.Max()), 0, width - 1);
            int ymin = clampi((int)Math.Floor(box_y.Min()), 0, height - 1);
            int ymax = clampi((int)Math.Ceiling(box_y.Max()), 0, height - 1);

            Mat mask = Mat.Zeros(ymax - ymin + 1, xmax - xmin + 1, MatType.CV_8UC1);

            Point[] root_point = new Point[4];
            root_point[0] = new Point((int)array[0][0] - xmin, (int)array[0][1] - ymin);
            root_point[1] = new Point((int)array[1][0] - xmin, (int)array[1][1] - ymin);
            root_point[2] = new Point((int)array[2][0] - xmin, (int)array[2][1] - ymin);
            root_point[3] = new Point((int)array[3][0] - xmin, (int)array[3][1] - ymin);
            Point[][] ppt = { root_point };

            Cv2.FillPoly(mask, ppt, new Scalar(1));

            Mat croppedImg = new Mat(pred.Clone(), new Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));


            float score = (float)Cv2.Mean(croppedImg, mask)[0];
            return score;
        }



        public static List<List<List<int>>> boxes_from_bitmap(Mat pred, Mat bitmap, float box_thresh, float det_db_unclip_ratio, string det_db_score_mode)
        {
            const int min_size = 3;
            const int max_candidates = 1000;

            int width = bitmap.Cols;
            int height = bitmap.Rows;

            Point[][] contours;
            HierarchyIndex[] hierarchy;

            Cv2.FindContours(bitmap, out contours, out hierarchy, RetrievalModes.List,
                ContourApproximationModes.ApproxSimple);

            int num_contours = contours.Length >= max_candidates ? max_candidates : contours.Length;

            List<List<List<int>>> boxes = new List<List<List<int>>>();

            for (int _i = 0; _i < num_contours; _i++)
            {
  
                if (contours[_i].Length <= 2)
                {
                    continue;
                }
                
                float ssid;
                RotatedRect box = Cv2.MinAreaRect(contours[_i]);
                var array = get_mini_boxes(box, out ssid);

                var box_for_unclip = array;
                // end get_mini_box
                if (ssid < min_size)
                {
                    continue;
                }

                float score;
                if (det_db_score_mode == "slow")
                    /* compute using polygon*/
                    score = polygon_score_acc(contours[_i], pred);
                else
                    score = box_score_fast(array, pred);
                if (score < box_thresh)
                    continue;

                // start for unclip
                RotatedRect points = unclip(box_for_unclip, det_db_unclip_ratio);
                //Console.WriteLine("points.Size  {0}", points.Size);
                if (points.Size.Height < 1.001 && points.Size.Width < 1.001)
                {
                    continue;
                }
                // end for unclip

                RotatedRect clipbox = points;
                var cliparray = get_mini_boxes(clipbox, out ssid);
                //Console.WriteLine("ssid  {0}", ssid);
                if (ssid < min_size + 2)
                    continue;

                int dest_width = pred.Cols;
                int dest_height = pred.Rows;
                List<List<int>> intcliparray = new List<List<int>>();
                for (int num_pt = 0; num_pt < 4; num_pt++)
                {
                    List<int> a = new List<int>{
                    (int)clampf((float)Math.Round(cliparray[num_pt][0] / (float)(width) *(float)(dest_width)), 0, (float)(dest_width)),
                    (int)clampf((float)Math.Round(cliparray[num_pt][1] /(float)(height) * (float)(dest_height)), 0, (float)(dest_height))
                    };
                    intcliparray.Add(a);
                }
                boxes.Add(intcliparray);

            } // end for
            
            return boxes;
        }


        public static List<List<List<int>>> filter_tag_det_res(List<List<List<int>>> boxes, float ratio_h, float ratio_w, Mat srcimg)
        {
            int oriimg_h = srcimg.Rows;
            int oriimg_w = srcimg.Cols;
            List<List<List<int>>> root_points = new List<List<List<int>>>();
            for (int n = 0; n < boxes.Count(); n++)
            {
                boxes[n] = order_points_clockwise(boxes[n]);
                for (int m = 0; m < boxes[0].Count(); m++)
                {
                    boxes[n][m][0] = (int)(boxes[n][m][0] / ratio_h);
                    boxes[n][m][1] = (int)(boxes[n][m][1] / ratio_w);

                    boxes[n][m][0] = Math.Min(Math.Max(boxes[n][m][0], 0), oriimg_w - 1);
                    boxes[n][m][1] = Math.Min(Math.Max(boxes[n][m][1], 0), oriimg_h - 1);
                }
            }

            for (int n = 0; n < boxes.Count(); n++)
            {
                int rect_width, rect_height;
                rect_width = (int)(Math.Sqrt(Math.Pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                                      Math.Pow(boxes[n][0][1] - boxes[n][1][1], 2)));
                rect_height = (int)(Math.Sqrt(Math.Pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                                       Math.Pow(boxes[n][0][1] - boxes[n][3][1], 2)));
                if (rect_width <= 4 || rect_height <= 4)
                    continue;
                root_points.Add(boxes[n]);
            }
            return root_points;
        }

    }



    class TablePostProcessor
    {
        public TablePostProcessor(string label_path, bool merge_no_span_structure = true)
        {
            this.label_list_ = PaddleOcrUtility.read_dict(label_path);
            if (merge_no_span_structure)
            {
                this.label_list_.Add("<td></td>");
                for (int count = 0; count < label_list_.Count;)
                {
                    if (label_list_[count] == "<td>")
                    {
                        label_list_.RemoveAt(count);
                    }
                    else
                    {
                        ++count;
                    }
                }
            }
            // add_special_char
            this.label_list_.Insert(0, this.beg);
            this.label_list_.Add(this.end);
        }
        public void Run(List<float> loc_preds, List<float> structure_probs,
                 List<float> rec_scores, List<int> loc_preds_shape,
                 List<int> structure_probs_shape,
                 List<List<string>> rec_html_tag_batch,
                 List<List<List<int>>> rec_boxes_batch,
                 List<int> width_list, List<int> height_list)
        {
            for (int batch_idx = 0; batch_idx < structure_probs_shape[0]; batch_idx++)
            {
                // image tags and boxs
                List<string> rec_html_tags = new List<string>();
                List<List<int>> rec_boxes = new List<List<int>>();

                float score = 0.0f;
                int count = 0;
                float char_score = 0.0f;
                int char_idx = 0;

                // step
                for (int step_idx = 0; step_idx < structure_probs_shape[1]; step_idx++)
                {
                    string html_tag;
                    List<int> rec_box = new List<int>();
                    // html tag
                    int step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) *
                                         structure_probs_shape[2];
                    char_idx = (int)(PaddleOcrUtility.argmax(structure_probs, step_start_idx, step_start_idx + structure_probs_shape[2]));
                    char_score = Math.Max(
                        structure_probs[step_start_idx],
                        structure_probs[step_start_idx + structure_probs_shape[2]]);
                    html_tag = this.label_list_[char_idx];

                    if (step_idx > 0 && html_tag == this.end)
                    {
                        break;
                    }
                    if (html_tag == this.beg)
                    {
                        continue;
                    }
                    count += 1;
                    score += char_score;
                    rec_html_tags.Add(html_tag);

                    // box
                    if (html_tag == "<td>" || html_tag == "<td" || html_tag == "<td></td>")
                    {
                        for (int point_idx = 0; point_idx < loc_preds_shape[2]; point_idx++)
                        {
                            step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) *
                                                 loc_preds_shape[2] +
                                             point_idx;
                            float point = loc_preds[step_start_idx];
                            if (point_idx % 2 == 0)
                            {
                                point = (int)(point * width_list[batch_idx]);
                            }
                            else
                            {
                                point = (int)(point * height_list[batch_idx]);
                            }
                            rec_box.Add((int)point);
                        }
                        rec_boxes.Add(rec_box);
                    }
                }
                score /= count;
                if (float.IsNaN(score) || rec_boxes.Count == 0)
                {
                    score = -1;
                }
                rec_scores.Add(score);
                rec_boxes_batch.Add(rec_boxes);
                rec_html_tag_batch.Add(rec_html_tags);
            }

        }

        private List<string> label_list_ = new List<string>();
        private string end = "eos";
        private string beg = "sos";
    };

    class PicodetPostProcessor
    {
        private List<string> label_list_;
        private double score_threshold_ = 0.4;
        private double nms_threshold_ = 0.5;
        private int num_class_ = 5;
        public List<int> fpn_stride_ = new List<int> { 8, 16, 32, 64 };
        public PicodetPostProcessor(string label_path, List<int> fpn_stride,double score_threshold = 0.4, double nms_threshold = 0.5)
        {
            this.label_list_ = PaddleOcrUtility.read_dict(label_path);
            this.score_threshold_ = score_threshold;
            this.nms_threshold_ = nms_threshold;
            this.num_class_ = label_list_.Count;
            this.fpn_stride_ = fpn_stride;
        }


        public void Run(List<StructurePredictResult> results, List<List<float>> outs, List<int> ori_shape, List<int> resize_shape, int reg_max)
        {
            int in_h = resize_shape[0];
            int in_w = resize_shape[1];
            float scale_factor_h = resize_shape[0] / (float)(ori_shape[0]);
            float scale_factor_w = resize_shape[1] / (float)(ori_shape[1]);

            List<List<StructurePredictResult>> bbox_results = new List<List<StructurePredictResult>>();
            for (int i = 0; i < num_class_; ++i) bbox_results.Add(new List<StructurePredictResult>());
            for (int i = 0; i < this.fpn_stride_.Count; ++i)
            {
                int feature_h = (int)Math.Ceiling((double)in_h / this.fpn_stride_[i]);
                int feature_w = (int)Math.Ceiling((double)in_w / this.fpn_stride_[i]);
                for (int idx = 0; idx < feature_h * feature_w; idx++)
                {
                    // score and label
                    float score = 0;
                    int cur_label = 0;
                    for (int label = 0; label < this.num_class_; label++)
                    {
                        if (outs[i][idx * this.num_class_ + label] > score)
                        {
                            score = outs[i][idx * this.num_class_ + label];
                            cur_label = label;
                        }
                    }
                    // bbox
                    if (score > this.score_threshold_)
                    {
                        int row = idx / feature_w;
                        int col = idx % feature_w;
                        List<float> bbox_pred = outs[i + this.fpn_stride_.Count].GetRange(idx * 4 * reg_max, 1 * 4 * reg_max);
                        bbox_results[cur_label].Add(
                            this.disPred2Bbox(bbox_pred, cur_label, score, col, row,
                                               this.fpn_stride_[i], resize_shape, reg_max));
                    }
                }
            }
            for (int i = 0; i < bbox_results.Count; i++)
            {
                bool flag = bbox_results[i].Count <= 0;
            }
            for (int i = 0; i < bbox_results.Count; i++)
            {
                bool flag = bbox_results[i].Count <= 0;
                if (bbox_results[i].Count <= 0)
                {
                    continue;
                }
                bbox_results[i] = this.nms(bbox_results[i], (float)this.nms_threshold_);
                foreach (var box in bbox_results[i])
                {
                    box.box[0] = box.box[0] / scale_factor_w;
                    box.box[2] = box.box[2] / scale_factor_w;
                    box.box[1] = box.box[1] / scale_factor_h;
                    box.box[3] = box.box[3] / scale_factor_h;
                    results.Add(box);
                }
            }
        }

        StructurePredictResult disPred2Bbox(List<float> bbox_pred, int label, float score, int x, int y, int stride, List<int> im_shape, int reg_max)
        {
            float ct_x = (x + 0.5f) * stride;
            float ct_y = (y + 0.5f) * stride;
            List<float> dis_pred = new List<float>(4);
            for (int i = 0; i < 4; ++i) dis_pred.Add(0.0f);

            for (int i = 0; i < 4; i++)
            {
                float dis = 0;
                List<float> bbox_pred_i = bbox_pred.GetRange(i * reg_max, reg_max);
                List<float> dis_after_sm = PaddleOcrUtility.activation_function_softmax(bbox_pred_i);
                for (int j = 0; j < reg_max; j++)
                {
                    dis += j * dis_after_sm[j];
                }
                dis *= stride;
                dis_pred[i] = dis;
            }

            float xmin = Math.Max(ct_x - dis_pred[0], .0f);
            float ymin = Math.Max(ct_y - dis_pred[1], .0f);
            float xmax = Math.Min(ct_x + dis_pred[2], (float)im_shape[1]);
            float ymax = Math.Min(ct_y + dis_pred[3], (float)im_shape[0]);

            StructurePredictResult result_item = new StructurePredictResult();
            result_item.box = new List<float> { xmin, ymin, xmax, ymax };
            result_item.type = this.label_list_[label];
            result_item.confidence = score;

            return result_item;
        }


        List<StructurePredictResult> nms(List<StructurePredictResult> input_boxes, float nms_threshold)
        {

            input_boxes.Sort((x, y) => x.confidence.CompareTo(y.confidence));
            List<int> picked = new List<int>(); 
            for (int i = 0; i < input_boxes.Count; ++i) picked.Add(1);
            for (int i = 0; i < input_boxes.Count; ++i)
            {
                if (picked[i] == 0)
                {
                    continue;
                }
                for (int j = i + 1; j < input_boxes.Count; ++j)
                {
                    if (picked[j] == 0)
                    {
                        continue;
                    }
                    float iou = PaddleOcrUtility.iou(input_boxes[i].box, input_boxes[j].box);
                    if (iou > nms_threshold)
                    {
                        picked[j] = 0;
                    }
                }
            }
            List<StructurePredictResult> input_boxes_nms = new List<StructurePredictResult>();
            for (int i = 0; i < input_boxes.Count; ++i)
            {
                if (picked[i] == 1)
                {
                    input_boxes_nms.Add(input_boxes[i]);
                }
            }
            return input_boxes_nms;
        }
    };
}
