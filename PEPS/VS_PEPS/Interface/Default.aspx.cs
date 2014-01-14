using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Runtime;
using Wrapper;

namespace Interface
{
    public partial class _Default : System.Web.UI.Page
    {
        public void price(object sender, EventArgs e)
        {
            // Récupérer les valeurs des paramètres dans les différentes TextBox
            WrapperClass wc = new WrapperClass();
            /*double strike = double.Parse(strike_tb.Text);
            int size = int.Parse(size_tb.Text);
            double r = double.Parse(r_tb.Text);
            double he = double.Parse(h_tb.Text);
            int sample = int.Parse(samples_tb.Text);*/


            //byte[] abyte1 = new byte[type.Text.Length];
            //System.Text.ASCIIEncoding encoding = new System.Text.ASCIIEncoding();
            //abyte1 = encoding.GetBytes(type.Text);
            //sbyte[] baske = Array.ConvertAll(abyte1, q => Convert.ToSByte(q));
            //double[] sigma = new double[5];
            //sigm[0] = double.Parse(sigma.Text);
            //double[] rho = new double[1];
            //ro[0] = double.Parse(rho.Text);

            // Valeurs manuels
            const int size = 1;
            double S0 = 100;
            double T = 1;
            double K = 100;
            int M = 50000;
            double sigma = .2;
            double r = .05;
            int timeStep = 1;

            wc.getPriceOption(M, T, S0, K, sigma, r, size, timeStep);

            prix_label.Text = wc.getPrice().ToString();
            ic_label.Text = wc.getIC().ToString();
        }
    }
}
