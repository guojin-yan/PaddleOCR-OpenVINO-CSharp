using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using iTextSharp.text.pdf.parser.clipper;

namespace paddleocr
{
    public class PostProcessor
    {

        int clampi(int x, int min, int max)
        {
            if (x > max)
                return max;
            if (x < min)
                return min;
            return x;
        }

        float clampf(float x, float min, float max)
        {
            if (x > max)
                return max;
            if (x < min)
                return min;
            return x;
        }

        void GetContourArea(List<List<float>> box, float unclip_ratio, float distance)
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

        RotatedRect UnClip(List<List<float>> box, float unclip_ratio)
        {
            double distance = 1.0;

            GetContourArea(box, unclip_ratio, (float)distance);

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


        List<List<int>> OrderPointsClockwise(List<List<int>> pts)
        {
            List<List<int>> box = pts;
            box.OrderBy(t => t[0]).ToList();

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

        List<List<float>> mat_to_list(Mat mat)
        {
            List<List<float>> img_vec = new List<List<float>>();
            List<float> tmp = new List<float>();

            for (int i = 0; i < mat.Rows; ++i)
            {
                tmp.Clear();
                for (int j = 0; j < mat.Cols; ++j)
                {
                    tmp.Add(mat.At<float>(i, j));
                }
                img_vec.Add(tmp);
            }
            return img_vec;
        }


        List<List<float>> get_mini_boxes(RotatedRect box, out float ssid)
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

        float PolygonScoreAcc(Point[] contour, Mat pred)
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
        float BoxScoreFast(List<List<float>> box_array, Mat pred)
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



        public List<List<List<int>>> BoxesFromBitmap(Mat pred, Mat bitmap, float box_thresh, float det_db_unclip_ratio, string det_db_score_mode)
        {
            const int min_size = 3;
            const int max_candidates = 1000;

            int width = bitmap.Cols;
            int height = bitmap.Rows;

            Point[][] contours;
            HierarchyIndex[] hierarchy;

            Cv2.FindContours(bitmap, out contours, out hierarchy, RetrievalModes.List,
                ContourApproximationModes.ApproxSimple);

            int num_contours =
              contours.Length >= max_candidates ? max_candidates : contours.Length;

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
                    score = PolygonScoreAcc(contours[_i], pred);
                else
                    score = BoxScoreFast(array, pred);

                if (score < box_thresh)
                    continue;

                // start for unclip
                RotatedRect points = UnClip(box_for_unclip, det_db_unclip_ratio);
                if (points.Size.Height < 1.001 && points.Size.Width < 1.001)
                {
                    continue;
                }
                // end for unclip

                RotatedRect clipbox = points;
                var cliparray = get_mini_boxes(clipbox, out ssid);

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


        public List<List<List<int>>> FilterTagDetRes(List<List<List<int>>> boxes, float ratio_h, float ratio_w, Mat srcimg)
        {
            int oriimg_h = srcimg.Rows;
            int oriimg_w = srcimg.Cols;

            List<List<List<int>>> root_points = new List<List<List<int>>>();
            for (int n = 0; n < boxes.Count(); n++)
            {
                for (int m = 0; m < boxes[0].Count(); m++)
                {
                    boxes[n][m][0] = (int)(boxes[n][m][0] / ratio_w);
                    boxes[n][m][1] = (int)(boxes[n][m][0] / ratio_h);

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
}
