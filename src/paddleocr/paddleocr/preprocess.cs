using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Size = OpenCvSharp.Size;

namespace PaddleOCR
{
    public static class PreProcess
    {
        public static float[] permute(Mat im)
        {
            int rh = im.Rows;
            int rw = im.Cols;
            int rc = im.Channels();
            float[] res = new float[rh * rw * rc];

            GCHandle resultHandle = default;
            try
            {
                resultHandle = GCHandle.Alloc(res, GCHandleType.Pinned);
                IntPtr resultPtr = resultHandle.AddrOfPinnedObject();
                for (int i = 0; i < rc; ++i)
                {
                    using Mat dest = new(rh, rw, MatType.CV_32FC1, resultPtr + i * rh * rw * sizeof(float));
                    Cv2.ExtractChannel(im, dest, i);
                }
            }
            finally
            {
                resultHandle.Free();
            }
            return res;
        }

        public static float[] permute_batch(List<Mat> imgs)
        {
            int rh = imgs[0].Rows;
            int rw = imgs[0].Cols;
            int rc = imgs[0].Channels();
            float[] res = new float[rh * rw * rc * imgs.Count];

            GCHandle resultHandle = default;
            resultHandle = GCHandle.Alloc(res, GCHandleType.Pinned);
            IntPtr resultPtr = resultHandle.AddrOfPinnedObject();
            try
            {
                for (int j = 0; j < imgs.Count; j++)
                {
                    for (int i = 0; i < rc; ++i)
                    {
                        using Mat dest = new(rh, rw, MatType.CV_32FC1, resultPtr + (j * rc + i) * rh * rw * sizeof(float));
                        Cv2.ExtractChannel(imgs[j], dest, i);
                    }
                }
            }
            finally
            {
                resultHandle.Free();
            }
            return res;
        }

        public static Mat normalize(Mat im, float[] mean, float[] scale, bool is_scale)
        {
            double e = 1.0;
            if (is_scale)
            {
                e /= 255.0;
            }
            im.ConvertTo(im, MatType.CV_32FC3, e);
            Mat[] bgr_channels = new Mat[3];
            Cv2.Split(im, out bgr_channels);
            for (var i = 0; i < bgr_channels.Length; i++)
            {
                bgr_channels[i].ConvertTo(bgr_channels[i], MatType.CV_32FC1, 1.0 * scale[i],
                    (0.0 - mean[i]) * scale[i]);
            }
            Mat re = new Mat();
            Cv2.Merge(bgr_channels, re);
            return re;
        }

        public static Mat resize_imgtype0(Mat img, string limit_type, int limit_side_len,
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

            //int resize_h = 960;
            //int resize_w = 960;

            resize_h = Math.Max((int)(Math.Round((float)(resize_h) / 32.0f) * 32), 32);
            resize_w = Math.Max((int)(Math.Round((float)(resize_w) / 32.0f) * 32), 32);

            Mat resize_img = new Mat();
            Cv2.Resize(img, resize_img, new Size(resize_w, resize_h));
            ratio_h = (float)(resize_h) / (float)(h);
            ratio_w = (float)(resize_w) / (float)(w);
            return resize_img;
        }

        public static Mat cls_resize_img(Mat img, List<int> cls_image_shape)
        {
            int imgC, imgH, imgW;
            imgC = cls_image_shape[0];
            imgH = cls_image_shape[1];
            imgW = cls_image_shape[2];

            float ratio = (float)img.Cols / (float)img.Rows;
            int resize_w, resize_h;
            if (Math.Ceiling(imgH * ratio) > imgW)
                resize_w = imgW;
            else
                resize_w = (int)(Math.Ceiling(imgH * ratio));
            Mat resize_img = new Mat();
            Cv2.Resize(img, resize_img, new Size(resize_w, imgH), 0.0f, 0.0f, InterpolationFlags.Linear);
            return resize_img;
        }


        public static Mat crnn_resize_img(Mat img, float wh_ratio, int[] rec_image_shape)
        {
            int imgC, imgH, imgW;
            imgC = rec_image_shape[0];
            imgH = rec_image_shape[1];
            imgW = rec_image_shape[2];

            imgW = (int)(imgH * wh_ratio);

            float ratio = (float)(img.Cols) / (float)(img.Rows);
            int resize_w, resize_h;

            if (Math.Ceiling(imgH * ratio) > imgW)
                resize_w = imgW;
            else
                resize_w = (int)(Math.Ceiling(imgH * ratio));
            Mat resize_img = new Mat();
            Cv2.Resize(img, resize_img, new Size(resize_w, imgH), 0.0f, 0.0f, InterpolationFlags.Linear);
            Cv2.CopyMakeBorder(resize_img, resize_img, 0, 0, 0,(int)(imgW - resize_img.Cols), BorderTypes.Constant, new Scalar( 127, 127, 127));
            return resize_img;
        }


        public static void TableResizeImg(Mat img, Mat resize_img, int max_len = 488)
        {
            int w = img.Cols;
            int h = img.Rows;

            int max_wh = w >= h ? w : h;
            float ratio = w >= h ? (float)(max_len) / (float)(w) : (float)(max_len) / (float)(h);

            int resize_h = (int)((float)(h) * ratio);
            int resize_w = (int)((float)(w) * ratio);

            Cv2.Resize(img, resize_img, new Size(resize_w, resize_h));
        }
        public static Mat TablePadImg(Mat img, int max_len = 488)
        {
            int w = img.Cols;
            int h = img.Rows;
            Mat resize_img = new Mat();
            Cv2.CopyMakeBorder(img, resize_img, 0, max_len - h, 0, max_len - w, BorderTypes.Constant, new Scalar(0, 0, 0));
            return resize_img;
        }

        public static Mat Resize(Mat img, int h, int w)
        {
            Mat resize_img = new Mat();
            Cv2.Resize(img, resize_img, new Size(w, h));
            return resize_img;
        }
    }
}
