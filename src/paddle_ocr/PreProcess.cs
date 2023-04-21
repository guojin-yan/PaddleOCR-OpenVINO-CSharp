using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace paddleocr
{
    public class PreProcess
    {

        public Mat ResizeImgType0(Mat img, string limit_type, int limit_side_len,
            out float ratio_h, out float ratio_w)
        {
            int w = img.Cols;
            int h = img.Rows;
            float ratio = 1.0f;
            if (limit_type == "min")
            {
                int min_wh = Math.Min(h, w);
                if (min_wh < limit_side_len)
                {
                    if (h < w)
                    {
                        ratio = (float)limit_side_len / (float)h;
                    }
                    else
                    {
                        ratio = (float)limit_side_len / (float)w;
                    }
                }
            }
            else
            {
                int max_wh = Math.Max(h, w);
                if (max_wh > limit_side_len)
                {
                    if (h > w)
                    {
                        ratio = (float)(limit_side_len) / (float)(h);
                    }
                    else
                    {
                        ratio = (float)(limit_side_len) / (float)(w);
                    }
                }
            }

            int resize_h = (int)((float)(h) * ratio);
            int resize_w = (int)((float)(w) * ratio);

            resize_h = Math.Max((int)(Math.Round((float)(resize_h) / 32) * 32), 32);
            resize_w = Math.Max((int)(Math.Round((float)(resize_w) / 32) * 32), 32);

            Mat resize_img = new Mat();
            Cv2.Resize(img, resize_img, new Size(resize_w, resize_h));
            ratio_h = (float)(resize_h) / (float)(h);
            ratio_w = (float)(resize_w) / (float)(w);
            return resize_img;
        }
    }
}
