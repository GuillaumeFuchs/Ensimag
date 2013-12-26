using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Wrapper;

namespace WindowsFormsApplication1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            // Récupérer les valeurs des paramètres dans les différentes TextBox
            WrapperClass wc = new WrapperClass();
            double S0 = 100;
            double sigma = 0.2;
            double r = 0.095;
            double T = 1;
            double K = 100;
            int J = 12;
            int M = 50000;

            wc.getPriceOptionMC(S0, sigma, r, T, K, J, M);
            PrixMC_Lab.Text = wc.getPrice().ToString();
            ICMC_Lab.Text = wc.getIC().ToString();

            wc.getPriceOptionMCC(S0, sigma, r, T, K, J, M);
            PrixMCC_Lab.Text = wc.getPrice().ToString();
            ICMCC_Lab.Text = wc.getIC().ToString();
        }

    }
}
