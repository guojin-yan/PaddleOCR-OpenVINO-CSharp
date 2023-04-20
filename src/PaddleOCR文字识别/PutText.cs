using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace OpenVinoSharpPaddleOCR
{
    internal class PutText
    {
        public static Mat put_text(Mat image, Rect[] rects, string[] texts) {
            Bitmap bitmap_image = BitmapConverter.ToBitmap(image);
            Graphics g = Graphics.FromImage(bitmap_image);
            for (int s = 0; s < texts.Length; s++)
            {
                String str = texts[s];
                Font font = new Font("微软雅黑", 10);
                SolidBrush sbrush = new SolidBrush(System.Drawing.Color.Black);
                g.DrawString(str, font, sbrush, new System.Drawing.Point(rects[s].X, rects[s].Y));
            }

            image = BitmapConverter.ToMat(bitmap_image);

            return image;

        }
    }
}
